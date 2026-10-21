#!/usr/bin/env python3
"""
Student off-verse detector (work plan, Phase 1): learns the silver teacher's
soft_target (results/silver/silver_labels.csv) from frozen BGE-M3 embeddings.

Input per passage i: embeddings of passages i-2 .. i+2 of the same lecture
(5 x 1024), zero vectors at lecture edges plus 4 "missing" flags.
Target: soft_target in [0, 1].

Models
  v1   L2 logistic regression. Soft targets via duplicated rows (label 1 with
       weight p, label 0 with weight 1-p). C chosen by inner cross-validation
       grouped by lecture, on silver labels only (soft log-loss).
  v1b  MLP, one hidden layer of 256, dropout 0.3, BCE on soft targets, Adam;
       the number of epochs is chosen by early stopping on one inner fold of
       lectures (silver only).
  narrative  logistic regression on hand-made narrative features (ablation)
  stacker    logistic regression on [student v1, old off-verse scorer, narrative],
             trained on out-of-fold predictions, itself cross-fitted
Cross-fitting: data/folds_v1.csv; a lecture is never scored by a model trained
on its own silver labels.
Inductive: v1 trained on the 9 tuning lectures' silver labels only
(leave-one-lecture-out among them; all other lectures scored by the model on all 9).

Evaluation uses gold only here, on the TUNING lectures (English for decisions).
Writes results/student/student_poff.csv and results/student/student_eval.txt.

Usage:
    python scripts/student_detector.py
"""
import csv
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from embed_scores import unit_cache  # noqa: E402
from evalkit import UNITS, UNITS_HELDOUT, load_csv, load_gold, read_split, verse_number  # noqa: E402

OUT = os.path.join(HERE, "results", "student")
CTX = 2
CS = [0.001, 0.01, 0.1, 1.0]
SEED = 2026


# ---- data ----------------------------------------------------------------------
def load():
    units, E = [], []
    for p in (UNITS, UNITS_HELDOUT):
        units += load_csv(p)
        E.append(np.load(unit_cache(p))["emb"])
    E = np.concatenate(E).astype(np.float32)
    row = {u["unit_id"]: i for i, u in enumerate(units)}
    lectures = {}
    for u in sorted(units, key=lambda u: (u["video_file"], int(u["unit_index"]))):
        lectures.setdefault(u["video_file"], []).append(u["unit_id"])
    text = {u["unit_id"]: u["text"] for u in units}
    soft = {r["unit_id"]: float(r["soft_target"]) for r in
            load_csv(os.path.join(HERE, "results", "silver", "silver_labels.csv"))}
    fold = {r["video_file"]: int(r["fold"]) for r in load_csv(os.path.join(HERE, "data", "folds_v1.csv"))}
    return E, row, lectures, text, soft, fold


def window_features(E, row, lectures):
    X = {}
    for v, us in lectures.items():
        for k, u in enumerate(us):
            parts, flags = [], []
            for d in range(-CTX, CTX + 1):
                j = k + d
                if 0 <= j < len(us):
                    parts.append(E[row[us[j]]])
                    if d:
                        flags.append(0.0)
                else:
                    parts.append(np.zeros(E.shape[1], dtype=np.float32))
                    flags.append(1.0)
            X[u] = np.concatenate(parts + [np.array(flags, dtype=np.float32)])
    return X


STORY = re.compile(r"\bonce\b|\bthere was\b|\bthere were\b|\bone day\b|\bstory\b|\banecdote\b|ಒಂದು ಕಥೆ|ಕಥೆ", re.I)
DIALOGUE = re.compile(r"\bsaid\b|\bsays\b|\basked\b|\btold\b|\breplied\b|ಹೇಳಿದ|ಕೇಳಿದ", re.I)
PAST = {"was", "were", "had", "did", "went", "came", "told", "saw", "took", "gave", "asked", "thought", "said",
        "made", "got", "knew", "found", "left", "became"}


def narrative_features(text, catalogue_words):
    X = {}
    for u, t in text.items():
        words = re.findall(r"[A-Za-z]+", t)
        n = max(len(words), 1)
        past = sum(1 for w in words if w.lower() in PAST or (len(w) > 4 and w.lower().endswith("ed")))
        caps = [w for i, w in enumerate(words) if w[0].isupper() and i > 0 and w.lower() not in catalogue_words]
        X[u] = np.array([past / n, len(caps) / n, len(STORY.findall(t)), len(DIALOGUE.findall(t)),
                         float(len(words) == 0)], dtype=np.float32)
    return X


# ---- models --------------------------------------------------------------------
def soft_logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_lr(X, y, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xd = np.vstack([X, X])
    yd = np.r_[np.ones(len(y)), np.zeros(len(y))]
    w = np.r_[y, 1 - y]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    m.fit(Xd, yd, logisticregression__sample_weight=w)
    return m


def choose_C(X, y, groups):
    """C with the lowest soft log-loss under inner leave-group-out (silver only)."""
    best = None
    for C in CS:
        losses = []
        for g in sorted(set(groups)):
            tr, te = groups != g, groups == g
            losses.append(soft_logloss(fit_lr(X[tr], y[tr], C).predict_proba(X[te])[:, 1], y[te]))
        loss = float(np.mean(losses))
        if best is None or loss < best[1]:
            best = (C, loss)
    return best[0]


def fit_mlp(X, y, Xval, yval, max_epochs=200, patience=10):
    import torch
    torch.manual_seed(SEED)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    f = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)  # noqa: E731
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                              torch.nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    Xt, yt, Xv, yv = f(X), torch.tensor(y, dtype=torch.float32), f(Xval), torch.tensor(yval, dtype=torch.float32)
    best, best_state, bad = None, None, 0
    for _ in range(max_epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(net(Xt[b]).squeeze(1), yt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = loss_fn(net(Xv).squeeze(1), yv).item()
        if best is None or vl < best - 1e-4:
            best, best_state, bad = vl, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(A):
        with torch.no_grad():
            return torch.sigmoid(net(f(A)).squeeze(1)).numpy()
    return predict


def old_scorer_probs(lectures):
    """the locked off-verse scorer's P_off: leave-one-lecture-out on tuning lectures,
    the locked model elsewhere"""
    from step4_viterbi import predict_locked
    from step3_scorer import FEATS, model, off_X
    from step4_viterbi import carried_cos
    z = np.load(FEATS)
    F = {k: z[k] for k in z.files}
    refs = list(F["refs"])
    row = {u: i for i, u in enumerate(F["unit_ids"])}
    _, gold, _ = load_gold()
    tune = read_split("kn", "dev") + read_split("en", "dev")
    out = {}
    dec = predict_locked({v: us for v, us in lectures.items() if v not in tune})
    for v, d in dec.items():
        out.update({u: p_off for u, (_, _, _, p_off) in zip(lectures[v], d)})
    X = {v: np.column_stack([off_X(F, [row[u] for u in lectures[v]], "a05"),
                             carried_cos(F, row, refs, lectures[v], "a05")]) for v in tune}
    Y = {v: np.array([int(not any(verse_number(r) for r in gold[u])) for u in lectures[v]]) for v in tune}
    for v in tune:
        tr = [w for w in tune if w != v]
        m = model().fit(np.concatenate([X[w] for w in tr]), np.concatenate([Y[w] for w in tr]))
        out.update(zip(lectures[v], m.predict_proba(X[v])[:, 1]))
    return out


# ---- main ----------------------------------------------------------------------
def main():
    E, row, lectures, text, soft, fold = load()
    from roman_detector import RomanShlokaDetector
    from verse_translations import VERSE_TRANSLATIONS
    cat = {w.lower() for ts in VERSE_TRANSLATIONS.values() for t in ts for w in re.findall(r"[A-Za-z]+", t)}
    cat |= {w for v in RomanShlokaDetector().verses.values() for w in v.split()}
    XW = window_features(E, row, lectures)
    XN = narrative_features(text, cat)
    units = [u for v in sorted(lectures) for u in lectures[v]]
    lec = {u: v for v, us in lectures.items() for u in us}
    y = np.array([soft[u] for u in units])
    f_of = np.array([fold[lec[u]] for u in units])
    lec_arr = np.array([lec[u] for u in units])
    A_w = np.stack([XW[u] for u in units])
    A_n = np.stack([XN[u] for u in units])
    P = {k: np.zeros(len(units)) for k in ("v1", "v1b", "narrative")}
    chosen = []
    for k in range(5):
        tr, te = f_of != k, f_of == k
        C = choose_C(A_w[tr], y[tr], f_of[tr])
        chosen.append(C)
        P["v1"][te] = fit_lr(A_w[tr], y[tr], C).predict_proba(A_w[te])[:, 1]
        val_fold = sorted(set(f_of[tr]))[0]                      # early-stopping fold (silver only)
        fit_part, val_part = tr & (f_of != val_fold), tr & (f_of == val_fold)
        P["v1b"][te] = fit_mlp(A_w[fit_part], y[fit_part], A_w[val_part], y[val_part])(A_w[te])
        Cn = choose_C(A_n[tr], y[tr], f_of[tr])
        P["narrative"][te] = fit_lr(A_n[tr], y[tr], Cn).predict_proba(A_n[te])[:, 1]
        print(f"  fold {k}: v1 C={C}, narrative C={Cn}, held-out passages {int(te.sum())}")

    old = old_scorer_probs(lectures)
    P["old scorer"] = np.array([old[u] for u in units])
    S = np.column_stack([P["v1"], P["old scorer"], P["narrative"]])
    P["stacker"] = np.zeros(len(units))
    for k in range(5):
        tr, te = f_of != k, f_of == k
        P["stacker"][te] = fit_lr(S[tr], y[tr], 1.0).predict_proba(S[te])[:, 1]

    tune = read_split("kn", "dev") + read_split("en", "dev")
    is_t = np.isin(lec_arr, tune)
    P["inductive"] = np.zeros(len(units))
    groups_t = lec_arr[is_t]
    Ct = choose_C(A_w[is_t], y[is_t], groups_t)
    full = fit_lr(A_w[is_t], y[is_t], Ct)
    P["inductive"][~is_t] = full.predict_proba(A_w[~is_t])[:, 1]
    for v in tune:
        tr, te = is_t & (lec_arr != v), lec_arr == v
        P["inductive"][te] = fit_lr(A_w[tr], y[tr], Ct).predict_proba(A_w[te])[:, 1]
    P["teacher direct"] = y.copy()

    os.makedirs(OUT, exist_ok=True)
    cols = ["v1", "v1b", "inductive", "narrative", "stacker"]
    with open(os.path.join(OUT, "student_poff.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "video_file", "fold"] + [f"p_{c}" for c in cols])
        for i, u in enumerate(units):
            w.writerow([u, lec[u], f_of[i]] + [f"{P[c][i]:.4f}" for c in cols])

    # ---- evaluation: gold, tuning lectures only ---------------------------------
    from sklearn.metrics import roc_auc_score
    _, gold, _ = load_gold()
    lines = ["Student off-verse detector — evaluation against GOLD on the TUNING lectures only",
             f"v1 C per outer fold: {chosen}; inductive C: {Ct}", ""]
    order = ["old scorer", "teacher direct", "v1", "v1b", "inductive", "narrative", "stacker"]
    for L in ("en", "kn"):
        vids = read_split(L, "dev")
        idx = np.array([i for i, u in enumerate(units) if lec[u] in vids])
        g = np.array([int(not any(verse_number(r) for r in gold[units[i]])) for i in idx])
        lines.append(f"{'ENGLISH' if L == 'en' else 'KANNADA (1 off-verse passage; not for decisions)'} tuning: "
                     f"{len(idx)} passages, {int(g.sum())} gold off-verse")
        lines.append(f"  {'P_off source':16} {'AUC':>6} {'recall@0.5':>11} {'false flags@0.5':>16}   per-lecture AUC")
        for name in order:
            p = P[name][idx]
            auc = roc_auc_score(g, p) if 0 < g.sum() < len(g) else float("nan")
            rec = ((p >= 0.5) & (g == 1)).sum() / max(g.sum(), 1)
            ff = int(((p >= 0.5) & (g == 0)).sum())
            per = []
            if L == "en":
                for v in vids:
                    j = np.array([k for k, i in enumerate(idx) if lec[units[i]] == v])
                    per.append(f"{v} {roc_auc_score(g[j], p[j]):.3f}")
            lines.append(f"  {name:16} {auc:>6.3f} {rec:>11.3f} {ff:>7}/{int((g == 0).sum()):<8}   " + ", ".join(per))
        lines.append("")
    lines.append("teacher direct = the silver soft_target itself (its prompt was chosen on these lectures).")
    text_out = "\n".join(lines)
    print(text_out)
    open(os.path.join(OUT, "student_eval.txt"), "w", encoding="utf-8").write(text_out + "\n")
    print(f"\n  wrote -> results/student/student_poff.csv, results/student/student_eval.txt")


if __name__ == "__main__":
    main()
