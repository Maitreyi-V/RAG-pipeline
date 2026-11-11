"""Fine-tune a cross-encoder for one run and task (verse | off). Fixed epochs, no test data."""
import argparse, json, os, random, time
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--task", choices=["verse", "off"], required=True)
ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--max_len", type=int, default=384)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--examples_dir", default="examples")
ap.add_argument("--out_root", default="models")
a = ap.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ex = [json.loads(l) for l in open(f"{a.examples_dir}/{a.run}_{a.task}.jsonl", encoding="utf-8")]
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1).to(dev)
if dev == "cuda":
    model.gradient_checkpointing_enable()

def collate(batch):
    enc = tok([e["a"] for e in batch], [e["b"] for e in batch], truncation="longest_first",
              max_length=a.max_len, padding=True, return_tensors="pt")
    enc["labels"] = torch.tensor([float(e["y"]) for e in batch])
    enc["weights"] = torch.tensor([float(e["w"]) for e in batch])
    return enc

g = torch.Generator().manual_seed(a.seed)
dl = DataLoader(ex, batch_size=a.bs, shuffle=True, collate_fn=collate, generator=g)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
steps = max(1, (len(dl) // a.accum) * a.epochs)
sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
use_amp = dev == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
lossf = torch.nn.BCEWithLogitsLoss(reduction="none")
log, t0 = [], time.time()
model.train()
for ep in range(a.epochs):
    run_loss, n = 0.0, 0
    opt.zero_grad()
    for i, b in enumerate(dl):
        y, w = b.pop("labels").to(dev), b.pop("weights").to(dev)
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logit = model(**b).logits.squeeze(-1)
        loss = (lossf(logit.float(), y) * w).sum() / w.sum()
        scaler.scale(loss / a.accum).backward()
        if (i + 1) % a.accum == 0 or i + 1 == len(dl):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
        run_loss += loss.item(); n += 1
        if (i + 1) % 200 == 0:
            print(f"epoch {ep} step {i+1}/{len(dl)} loss {run_loss/n:.4f} {time.time()-t0:.0f}s", flush=True)
    log.append({"epoch": ep, "mean_loss": run_loss / max(1, n)})
    print(f"epoch {ep} done, mean loss {run_loss/max(1,n):.4f}", flush=True)

out = f"{a.out_root}/{a.run}_{a.task}"
os.makedirs(out, exist_ok=True)
model.save_pretrained(out); tok.save_pretrained(out)
json.dump({**vars(a), "n_examples": len(ex), "n_pos": sum(e["y"] for e in ex),
           "loss": log, "seconds": round(time.time() - t0, 1), "device": dev,
           "torch": torch.__version__}, open(f"{out}/train_meta.json", "w"), indent=1)
print("saved", out)
