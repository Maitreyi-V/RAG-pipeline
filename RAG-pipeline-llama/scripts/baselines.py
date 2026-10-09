#!/usr/bin/env python3
"""
Carry baselines and oracles for discourse propagation.

All rows share one mechanism: an ANCHOR sets the active verse, and a CARRY
POLICY decides how long it stays active on units where no anchor fires.
Only the two ingredients change:

  string only         detected anchors, never carry
  carry max_gap=k     detected anchors, carry at most k silent units
  carry unbounded     detected anchors, carry until the next anchor
  monotonic           unbounded, but an anchor may not move to an earlier verse
  oracle gate (drop)  detected anchors; the first carried unit where gold no
                      longer grounds the verse ends the run for good
  oracle gate (resume) detected anchors; the verse stays active until the next
                      anchor, but is only emitted on units where gold grounds
                      it -- so a speaker who digresses and returns is recovered
                      These two are the CEILINGS for the two ways a gate can work.
  oracle anchors      gold recitations (T1+T2) as anchors, unbounded carry --
                      what a perfect recitation detector would buy
  oracle both         gold anchors and the resume gate -- the ceiling of the
                      whole anchor-and-carry family

The max_gap rows together are the precision-recall curve of the distance
heuristic. On the TEST split they are for reporting only: never choose a
setting from them.

Usage:
    python scripts/baselines.py --language en --base results/final_l1_en.csv
    python scripts/baselines.py --language kn --base results/final_l1_kn.csv --split dev
"""
import argparse
import csv
import os

from evalkit import (HERE, SPANS, TIERS, UNITS, bootstrap, load_gold, load_pred,
                     read_split, score, verse_number)

GAPS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]
UNBOUNDED = 10 ** 9


def carry(us, anchors, max_gap=UNBOUNDED, monotonic=False, keep=None, resume=False):
    """Anchor-and-carry over one lecture.

    us       : unit ids in spoken order
    anchors  : {unit_id: [verse_ref, ...]}
    max_gap  : carry the active verse over at most this many silent units
    monotonic: ignore anchors that would move to an earlier verse number
    keep     : optional gate (unit_id, verse_ref) -> bool, applied to every
               carried label
    resume   : what a failed gate does. False: the verse is dropped for good.
               True: it is withheld on this unit only and stays active
    """
    out, active, gap = {}, [], 0
    for u in us:
        hits = list(anchors.get(u, ()))
        if monotonic and hits and active:
            floor = max(verse_number(v) or 0 for v in active)
            hits = [v for v in hits if (verse_number(v) or 0) >= floor]
        emit = hits
        if hits:
            active, gap = hits, 0
        elif active and gap < max_gap:
            gap += 1
            emit = [v for v in active if keep(u, v)] if keep else active
            if keep and not resume:
                active = emit
        else:
            active = emit = []
        out[u] = list(emit)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", choices=["kn", "en"], required=True)
    ap.add_argument("--base", required=True, help="string-matching predictions CSV")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--spans", default=SPANS)
    ap.add_argument("--units", default=UNITS)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", help="CSV (default results/baselines_<lang>_<split>.csv)")
    args = ap.parse_args()

    lectures, gold, _ = load_gold(args.spans, args.units)
    videos = [v for v in read_split(args.language, args.split) if v in lectures]
    detected = load_pred(args.base)
    recited = {u: [v for v, t in g.items() if t in ("T1", "T2")] for u, g in gold.items()}

    def in_gold(u, v):
        return v in gold[u]

    systems = [("string only", dict(anchors=detected, max_gap=0))]
    systems += [(f"carry max_gap={g}", dict(anchors=detected, max_gap=g)) for g in GAPS]
    systems += [
        ("carry unbounded", dict(anchors=detected)),
        ("monotonic unbounded", dict(anchors=detected, monotonic=True)),
        ("oracle gate (drop)", dict(anchors=detected, keep=in_gold)),
        ("oracle gate (resume)", dict(anchors=detected, keep=in_gold, resume=True)),
        ("oracle anchors (T1+T2)", dict(anchors=recited)),
        ("oracle both", dict(anchors=recited, keep=in_gold, resume=True)),
    ]

    n_units = sum(len(lectures[v]) for v in videos)
    n_empty = sum(1 for v in videos for u in lectures[v] if not gold[u])
    print("=" * 110)
    print(f"BASELINES — {args.language} / {args.split}   lectures {len(videos)}   "
          f"units {n_units}   empty {n_empty}")
    print("=" * 110)
    print(f"  {'system':24} {'P':>6} {'R':>6} {'F1':>6}  {'F1 95% CI':13} {'fire':>5}  "
          + "  ".join(f"{t:>8}" for t in TIERS))
    print("  " + "-" * 106)

    rows = []
    for name, kw in systems:
        pred = {}
        for v in videos:
            pred.update(carry(lectures[v], **kw))
        s = score(gold, pred, lectures, videos)
        ci = bootstrap(gold, pred, lectures, videos, n=args.n_boot)
        tiers = [f"{h}/{n}" for h, n in (s["tiers"][t] for t in TIERS)]
        print(f"  {name:24} {s['P']:>6.3f} {s['R']:>6.3f} {s['F1']:>6.3f}  "
              f"[{ci['F1'][0]:.3f},{ci['F1'][1]:.3f}] {s['fired']:>5}  "
              + "  ".join(f"{t:>8}" for t in tiers))
        rows.append([name, f"{s['P']:.4f}", f"{s['R']:.4f}", f"{s['F1']:.4f}",
                     f"{ci['P'][0]:.4f}", f"{ci['P'][1]:.4f}",
                     f"{ci['R'][0]:.4f}", f"{ci['R'][1]:.4f}",
                     f"{ci['F1'][0]:.4f}", f"{ci['F1'][1]:.4f}",
                     s["tp"], s["fp"], s["fn"], s["fired"], s["units"]] + tiers)

    print(f"\n  CI: {args.n_boot} bootstrap resamples over the {len(videos)} lectures. "
          f"Tier cells are hit/n.")
    if args.split == "test":
        print("  TEST split: report these rows, do not select a setting from them.")

    out = args.out or os.path.join(HERE, "results", f"baselines_{args.language}_{args.split}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "P", "R", "F1", "P_lo", "P_hi", "R_lo", "R_hi", "F1_lo", "F1_hi",
                    "tp", "fp", "fn", "fired", "units"] + TIERS)
        w.writerows(rows)
    print(f"  wrote -> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
