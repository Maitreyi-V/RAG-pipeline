#!/usr/bin/env python3
"""
Declare and fit the three systems for the v2 test run (pre-registered in
docs/experiment_log.md). Settings are fitted on the 9 TUNING lectures only.

  v2_teacher145   primary     teacher P_off (silver soft_target), 145-state resume
  v2_teacher73    ablation    teacher P_off, 73-state
  v2_local73      zero-API    old off-verse scorer + student logit (6th feature), 73-state

Writes results/<name>.pkl (read by step4_viterbi.predict_locked(path=...)) and
results/<name>.json (settings, human-readable).

Usage:
    python scripts/freeze_v2_systems.py
"""
import json
import math
import os
import pickle
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from phase2_decoder import column  # noqa: E402
from step4_viterbi import run  # noqa: E402

TEACHER = "results/silver/silver_labels.csv soft_target = clip((score+5)/10, 0.02, 0.98); gpt-5-mini-2025-08-07, config B"
STUDENT = "logit of results/student/student_poff.csv p_final (v1, 5-fold cross-fitted on silver), clipped to [0.02, 0.98]"


def main():
    teacher = column("results/silver/silver_labels.csv", "soft_target")
    student = column("results/student/student_poff.csv", "p_final")
    student_logit = {u: math.log(p / (1 - p)) for u, p in student.items()}
    systems = {
        "v2_teacher145": ("primary", dict(resume=True, off_override=teacher), TEACHER, 0.731),
        "v2_teacher73": ("ablation: does the 145-state decoder matter?", dict(off_override=teacher), TEACHER, 0.728),
        "v2_local73": ("zero-API variant", dict(off_extra=student_logit), "old off-verse scorer + " + STUDENT, 0.707),
    }
    for name, (role, kw, p_off, tune_en) in systems.items():
        path = os.path.join(HERE, "results", f"{name}.pkl")
        run(alphas=("a05",), verbose=False, save=path, **kw)
        m = pickle.load(open(path, "rb"))["a05"]
        cfg = dict(system=name, role=role, decoder="145-state resume" if m["resume"] else "73-state",
                   p_off=p_off, verse_scorer="locked Step 3 verse scorer (alpha 0.5, last)",
                   gamma=m["gamma"], beta_off=m["beta"], delta=m["delta"], tau=m["tau"],
                   counts=m["counts"], trained_on=m["trained_on"], tuning_lolo_en_strict=tune_en)
        json.dump(cfg, open(os.path.join(HERE, "results", f"{name}.json"), "w"), indent=1)
        print(f"  {name:14} {cfg['decoder']:17} gamma={m['gamma']} beta_off={m['beta']} tau={m['tau']}")


if __name__ == "__main__":
    main()
