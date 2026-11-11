"""Score calib + test lectures of one run with the fine-tuned models.

Writes scores/<run>/<lecture_id>.npz with
  verse_logit : [T, 72]  cross-encoder logit for every (passage, verse 2.1..2.72)
  off_logit   : [T]      off-verse classifier logit (sigmoid -> P_off before calibration)
"""
import argparse, os
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common import load_passages, load_verses, verse_text, off_context, load_plan, V

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--models_root", default="models")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--max_len", type=int, default=384)
ap.add_argument("--out_root", default="scores")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
lect, verses, plan = load_passages(), load_verses(), load_plan()
run = next(r for r in plan["runs"] if r["name"] == a.run)

def load(task):
    path = f"{a.models_root}/{a.run}_{task}"
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path).to(dev).eval()
    if dev == "cuda":
        m.half()
    return tok, m

@torch.no_grad()
def predict(tok, m, pairs):
    out = []
    for i in range(0, len(pairs), a.bs):
        chunk = pairs[i:i + a.bs]
        enc = tok([x for x, _ in chunk], [y for _, y in chunk], truncation="longest_first",
                  max_length=a.max_len, padding=True, return_tensors="pt").to(dev)
        out.append(m(**enc).logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)

vt, vm = load("verse"); ot, om = load("off")
os.makedirs(f"{a.out_root}/{a.run}", exist_ok=True)
for lid in run["calib"] + run["test"]:
    ps = lect[lid]; T = len(ps); lang = ps[0]["lang"]
    pairs = [(p["text"], verse_text(verses, v, lang)) for p in ps for v in range(1, V + 1)]
    verse_logit = predict(vt, vm, pairs).reshape(T, V)
    off_logit = predict(ot, om, [(p["text"], off_context(ps, t)) for t, p in enumerate(ps)])
    np.savez(f"{a.out_root}/{a.run}/{lid}.npz", verse_logit=verse_logit, off_logit=off_logit)
    print(f"{lid}: {T} passages scored", flush=True)
