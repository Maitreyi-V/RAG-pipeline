#!/usr/bin/env python3
"""
Run a detection layer over annotation UNITS rather than whole videos.

Video-level detection ("BG 2.1 appears somewhere in this video") collapses many
gold mentions into one scorable item and gives false positives no location.
Unit-level detection is the actual grounding task: for each ~100-word unit, which
verse (if any) is being grounded here?

Output: results/units_layer{N}.csv  ->  unit_id, detected_verses (semicolon list)

Usage:
    python scripts/detect_units.py --layer 1 --language kn
    python scripts/detect_units.py --layer 3 --language en --threshold 0.65
"""
import argparse
import csv
import io
import contextlib
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

UNITS = os.path.join(HERE, "data", "units_v2.csv")


def refs_from(result):
    """Detector outputs vary; pull verse refs out of whatever shape came back."""
    out = []
    for d in result or []:
        if isinstance(d, dict):
            r = d.get("verse_ref") or d.get("ref") or d.get("verse")
        elif isinstance(d, (list, tuple)) and len(d) >= 2:
            r = d[1]
        else:
            r = d
        if r:
            out.append(str(r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, choices=[1, 3, 4], required=True)
    ap.add_argument("--provider", choices=["openai", "ollama"], default="openai",
                    help="layer 4 only")
    ap.add_argument("--model", default=None, help="layer 4 only")
    ap.add_argument("--only-annotated", action="store_true",
                    help="restrict to units that already have gold labels "
                         "(saves API calls on layer 4)")
    ap.add_argument("--units", default=UNITS)
    ap.add_argument("--language", choices=["kn", "en"])
    ap.add_argument("--video")
    ap.add_argument("--threshold", type=float, default=0.65, help="layer 3 only")
    ap.add_argument("--top-k", type=int, default=3,
                    help="layer 3: max verses returned per unit (default 3)")
    ap.add_argument("--sanskrit-threshold", type=float, default=0.12,
                    help="layer 1: min Sanskrit density for a window (default 0.12)")
    ap.add_argument("--match-threshold", type=int, default=55,
                    help="layer 1: min fuzzy score 0-100 (default 55)")
    ap.add_argument("--script", choices=["native", "roman"], default="native",
                    help="layer 1: 'native' matches Kannada-script verses; "
                         "'roman' matches romanised Sanskrit (English transcripts)")
    ap.add_argument("--out")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>.partial, skipping units already done")
    args = ap.parse_args()

    with open(args.units, newline="", encoding="utf-8") as f:
        units = list(csv.DictReader(f))
    if args.language:
        units = [u for u in units if u["language"] == args.language]
    if args.video:
        units = [u for u in units if u["video_file"] == args.video]
    if args.only_annotated:
        spans = os.path.join(HERE, "data", "mention_spans.csv")
        if os.path.exists(spans):
            with open(spans, newline="", encoding="utf-8") as f:
                labelled = {r["unit_id"] for r in csv.DictReader(f)}
            units = [u for u in units if u["unit_id"] in labelled]
    if not units:
        sys.exit("No units matched.")

    if args.layer == 1 and args.script == "roman":
        from roman_detector import RomanShlokaDetector
        det = RomanShlokaDetector(match_threshold=args.match_threshold)
        print(f"  layer 1 (roman): match_threshold={args.match_threshold}")

        def predict(text):
            return refs_from(det.detect(text))
    elif args.layer == 1:
        from shloka_detector import ShlokaDetector
        det = ShlokaDetector(sanskrit_threshold=args.sanskrit_threshold,
                             match_threshold=args.match_threshold)
        print(f"  layer 1: sanskrit_threshold={args.sanskrit_threshold} "
              f"match_threshold={args.match_threshold}")

        def predict(text):
            with contextlib.redirect_stdout(io.StringIO()):
                return refs_from(det.detect(text))
    elif args.layer == 4:
        from dotenv import load_dotenv
        load_dotenv()
        from layer4_llm_detector import detect_verse_llm
        model = args.model or ("gpt-4o-mini" if args.provider == "openai"
                               else "llama3.1:8b")
        print(f"  layer 4: {args.provider} / {model}")

        def predict(text):
            nums = detect_verse_llm(text, args.provider, model)
            return [f"BG 2.{n}" for n in nums]
    else:
        from layer3_semantic import Layer3SemanticMatcher
        m = Layer3SemanticMatcher(threshold=args.threshold)
        with contextlib.redirect_stdout(io.StringIO()):
            m.build_index()

        def predict(text):
            # top-k, so Layer 3 can return several verses like Layer 1 can.
            # detect() is top-1 only, which caps recall on multi-verse units.
            with contextlib.redirect_stdout(io.StringIO()):
                hits = m.detect_top_k(text, k=args.top_k)
            return [ref for ref, score in hits
                    if ref and score >= args.threshold]

    out = args.out or os.path.join(HERE, "results", f"units_layer{args.layer}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # Checkpointing: each successful unit is appended to <out>.partial and
    # flushed immediately, so a crash, a killed terminal or a dead ollama server
    # costs at most one unit. The partial file is deliberately NOT the output
    # path -- a half-finished run must never look like a scoreable result.
    # It is deleted once the full run completes.
    partial = out + ".partial"
    done = {}
    if args.resume and os.path.exists(partial):
        with open(partial, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done[r["unit_id"]] = r["detected_verses"]
        print(f"  resuming: {len(done)} units already done, "
              f"{len(units) - len(done)} to go")
    elif os.path.exists(partial):
        os.remove(partial)

    todo = [u for u in units if u["unit_id"] not in done]
    n_err = 0
    first_err = None

    pf = open(partial, "a", newline="", encoding="utf-8")
    pw = csv.writer(pf)
    if not done:
        pw.writerow(["unit_id", "video_file", "detected_verses"])
        pf.flush()

    try:
        for i, u in enumerate(todo, 1):
            try:
                refs = sorted(set(predict(u["text"])))
            except Exception as e:
                n_err += 1
                if first_err is None:
                    first_err = f"{type(e).__name__}: {e}"
                    print(f"  !! {u['unit_id']}: {first_err}")
                if n_err >= 5 and n_err == i:
                    sys.exit(f"\nABORTED: first {n_err} units all failed.\n"
                             f"  {first_err}\n  Progress kept in {os.path.basename(partial)} "
                             f"-- fix the cause and re-run with --resume.")
                continue                      # not checkpointed: retried on resume
            done[u["unit_id"]] = ";".join(refs)
            pw.writerow([u["unit_id"], u["video_file"], ";".join(refs)])
            pf.flush()
            if i % 25 == 0:
                print(f"  ... {i}/{len(todo)}")
    finally:
        pf.close()

    n_hit = sum(1 for v in done.values() if v)
    print(f"\n  layer {args.layer} over {len(units)} units "
          f"({len(done)} completed, {len(units)-len(done)} failed)")
    print(f"  units with a detection: {n_hit} ({n_hit/max(len(done),1):.1%})")
    if n_err:
        print(f"  !! errors: {n_err} — first was {first_err}")

    missing = len(units) - len(done)
    if missing and missing / len(units) > 0.02:
        sys.exit(f"\nABORTED: {missing}/{len(units)} units incomplete "
                 f"({missing/len(units):.1%}). Failed units are indistinguishable "
                 f"from true negatives, so the result is unscoreable.\n"
                 f"  Progress kept in {os.path.basename(partial)} "
                 f"-- re-run with --resume to finish them.")

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "video_file", "detected_verses"])
        for u in units:
            if u["unit_id"] in done:
                w.writerow([u["unit_id"], u["video_file"], done[u["unit_id"]]])
    os.remove(partial)
    print(f"  wrote -> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
