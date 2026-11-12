#!/usr/bin/env python3
"""
Tuning B: 2 of the 7 E2 English test lectures and 3 of the 23 E2 Kannada test lectures, picked
with a fixed seed (random.Random(SEED).sample over the sorted split lists). Tuning B is used only
to check and choose cascade settings; scorer weights, transition counts and profiles are not refit.

-> results/tuning_b.txt  (the picked lecture ids)

Usage:
    python scripts/pick_tuning_b.py
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from evalkit import read_split  # noqa: E402

SEED = 20261112
PICK = {"en": 2, "kn": 3}


def tuning_b():
    rng = random.Random(SEED)
    return {L: sorted(rng.sample(sorted(read_split(L, "test")), PICK[L])) for L in ("en", "kn")}


def main():
    tb = tuning_b()
    text = f"tuning B, seed {SEED}\n" + "\n".join(f"{L}\t{','.join(vs)}" for L, vs in tb.items())
    print(text)
    with open(os.path.join(HERE, "results", "tuning_b.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
