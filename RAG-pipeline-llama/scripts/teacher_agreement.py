#!/usr/bin/env python3
"""
Declared teacher analysis (pre-registered 21 Oct 2026): agreement of the silver
teacher (gpt-5-mini, config B) with GOLD on off-verse, per language and, on the
new-speaker split, per speaker. Run once per split, after the v2 test run.

Gold off-verse = passage with no Chapter 2 verse. Teacher flag = label DIGRESSION.
Per group: passages, gold off-verse share, teacher flag share, accuracy, off-verse
recall, false flags on verse passages, Cohen's kappa, AUC of the score
(+confidence for DIGRESSION, -confidence for VERSE; only when both classes occur).
Original-speaker lectures have no speaker names in data/lecture_meta.csv, so E2
is grouped by language.

Writes results/teacher_agreement_<split>.txt.

Usage:
    python scripts/teacher_agreement.py --split test
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import LECTURES_NEW, load_csv, load_gold, read_split, verse_number  # noqa: E402


def stats(us, gold, silver):
    from sklearn.metrics import cohen_kappa_score, roc_auc_score
    y = [int(not any(verse_number(r) for r in gold[u])) for u in us]
    f = [int(silver[u]["label"] == "DIGRESSION") for u in us]
    sc = [float(silver[u]["score"]) for u in us]
    n, n_off = len(y), sum(y)
    acc = sum(a == b for a, b in zip(y, f)) / n
    rec = sum(b for a, b in zip(y, f) if a) / n_off if n_off else float("nan")
    ff = sum(b for a, b in zip(y, f) if not a)
    kap = cohen_kappa_score(y, f) if len(set(y)) > 1 or len(set(f)) > 1 else float("nan")
    auc = roc_auc_score(y, sc) if 0 < n_off < n else float("nan")
    return (f"{n:>6} {n_off / n:>8.3f} {sum(f) / n:>8.3f} {acc:>6.3f} {rec:>7.3f} "
            f"{ff:>5}/{n - n_off:<5} {ff / (n - n_off) if n > n_off else float('nan'):>6.3f} {kap:>6.3f} {auc:>6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "heldout"], required=True)
    args = ap.parse_args()
    lectures, gold, _ = load_gold()
    silver = {r["unit_id"]: r for r in csv.DictReader(open(os.path.join(HERE, "results/silver/silver_labels.csv"),
                                                           encoding="utf-8"))}
    speaker = {r["video_file"]: r["speaker"] for r in load_csv(LECTURES_NEW)}
    groups = []
    for L, name in (("kn", "Kannada"), ("en", "English")):
        vids = [v for v in read_split(L, args.split) if v in lectures]
        if args.split == "heldout":
            for sp in sorted({speaker[v] for v in vids}):
                groups.append((f"{name}: {sp}", [v for v in vids if speaker[v] == sp]))
            groups.append((f"{name} new speakers (pooled)", vids))
        else:
            groups.append((f"{name} {'tuning' if args.split == 'dev' else 'test'}", vids))
    lines = [f"Teacher (gpt-5-mini, config B) vs gold off-verse — split = {args.split}"
             + ("   (DRY RUN on tuning lectures; the teacher prompt was chosen here)" if args.split == "dev" else ""),
             f"  {'group':40} {'lect':>4} {'pass':>6} {'gold off':>8} {'flagged':>8} {'acc':>6} {'recall':>7} "
             f"{'false flags':>11} {'rate':>6} {'kappa':>6} {'AUC':>6}"]
    for name, vids in groups:
        us = [u for v in vids for u in lectures[v]]
        lines.append(f"  {name:40} {len(vids):>4} " + stats(us, gold, silver))
    out = "\n".join(lines)
    print(out)
    open(os.path.join(HERE, "results", f"teacher_agreement_{args.split}.txt"), "w").write(out + "\n")


if __name__ == "__main__":
    main()
