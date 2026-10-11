"""
build_units_heldout.py — unit grid for the NEW-SPEAKER held-out lectures.

build_units.py rebuilds units_v2.csv from scratch and hard-codes
"Kannada = Dvaita, English = Advaita", so new speakers must not go through it.
This builds a separate grid with the same 100-word segmentation:

  reads   data/lectures_new.csv          one row per lecture (speaker, tradition, ...)
          data/transcripts_new/<video_file>.txt
  writes  data/units_heldout_v1.csv      same columns as units_v2.csv
          data/normalized/<video_file>.txt

Only lectures with status=ready are built. Use status=hold for a transcript
that is incomplete: annotating it would be wasted when the text changes.

Annotations key on unit_id, so a re-run may add lectures and may rebuild a
lecture nobody has labelled yet (a corrected transcript), but it must reproduce
every LABELLED unit exactly. If a labelled unit would change or disappear,
nothing is written.

Usage:
    python build_units_heldout.py
"""
import csv
import glob
import os
import sys

from build_units import BASE, FIELDS, NORMALIZED, text_units

MANIFEST = os.path.join(BASE, "data", "lectures_new.csv")
SOURCE = os.path.join(BASE, "data", "transcripts_new")
OUT_CSV = os.path.join(BASE, "data", "units_heldout_v1.csv")


def main():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        lectures = list(csv.DictReader(f))

    rows = []
    for lec in lectures:
        vk = lec["video_file"]
        if lec["status"] != "ready":
            print(f"  skip  {vk:28} (status={lec['status']})")
            continue
        path = os.path.join(SOURCE, vk + ".txt")
        if not os.path.exists(path):
            sys.exit(f"missing transcript: {path}")
        full, units = text_units(path)
        with open(os.path.join(NORMALIZED, vk + ".txt"), "w", encoding="utf-8") as f:
            f.write(full)
        for i, u in enumerate(units):
            rows.append({"unit_id": f"{vk}_{i:04d}", "video_file": vk,
                         "language": lec["language"], "tradition": lec["tradition"],
                         "unit_index": i, **u})
        print(f"  build {vk:28} {len(units):4} units  {lec['speaker']}")

    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="", encoding="utf-8") as f:
            old = {r["unit_id"]: r["text"] for r in csv.DictReader(f)}
        new = {r["unit_id"]: r["text"] for r in rows}
        changed = [u for u, t in old.items() if new.get(u) != t]
        labelled = set()
        for p in (glob.glob(os.path.join(BASE, "data", "annotations*", "*.csv"))
                  + [os.path.join(BASE, "data", "mention_spans.csv")]):
            with open(p, newline="", encoding="utf-8") as f:
                labelled |= {r["unit_id"] for r in csv.DictReader(f)}
        broken = [u for u in changed if u in labelled]
        if broken:
            sys.exit(f"\nABORTED: {len(broken)} LABELLED unit(s) would change or vanish "
                     f"(first: {broken[0]}).\n  Annotations key on unit_id. "
                     f"Restore the transcript, or write units_heldout_v2.csv.")
        if changed:
            vids = sorted({u.rsplit("_", 1)[0] for u in changed})
            print(f"\n  rebuilt {len(changed)} unlabelled unit(s) in: {', '.join(vids)}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  {len({r['video_file'] for r in rows})} lectures, {len(rows)} units "
          f"-> {os.path.relpath(OUT_CSV, BASE)}")


if __name__ == "__main__":
    main()
