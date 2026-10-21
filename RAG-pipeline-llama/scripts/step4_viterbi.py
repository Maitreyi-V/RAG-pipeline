#!/usr/bin/env python3
"""
Step 4: 73-state Viterbi over each lecture, TUNING lectures only.

States: 72 verses (BG 2.1 .. 2.72) + off-verse.

emission    verse v : log P_verse(v | passage)            (Step 3 verse scorer)
            off     : log P_off(passage) + beta_off       (Step 3 off-verse scorer)
            beta_off corrects the base-rate gap: P_verse is trained on pairs
            (about 1% positive), P_off on passages (about 32% off-verse).
transition  five tied buckets, counted on the GOLD state sequences of the
            training lectures, +1 smoothing, each bucket's mass shared equally
            over its targets:
                stay (v -> v), next (v -> v+1, v+2), jump (any other verse),
                to_off (v -> off), stay_off (off -> off), from_off (off -> any verse)
            gold state of a passage = its lowest Chapter 2 verse, or off.
            No "resume the last verse after off-verse" bucket (needs memory of
            the last verse; not in this first version).
path score  sum emission + gamma * sum log transition
second verse after decoding, a passage on verse y also outputs any other verse
            with P_verse >= tau (two verses in one passage, passing quotes)

Protocol on the 9 tuning lectures, leave-one-lecture-out: for each held-out
lecture, transitions are counted and (gamma, beta_off, tau) are chosen on the
other 8, maximising (F1_kn + F1_en) / 2 (strict). Emissions are the Step 3
leave-one-lecture-out probabilities (pooled scorer), so no lecture is scored by
a model that saw it -- except that the 8 training lectures' probabilities came
from models that saw the held-out one; this stays inside the tuning set.

Reported per language: strict P/R/F1 and within-1-verse F1 (a predicted verse
counts if gold has a verse within +-1 on that passage; a gold verse counts if a
predicted verse is within +-1), and theme F1 (verses mapped to the domain
expert's 9 themes in data/verse_themes.csv).

LOCKED system (16 Oct 2026, chosen on the tuning lectures only): `last` in the
verse scorer, decoupled emissions, cosine to the carried verse in the off-verse
scorer, no delta. These are run()'s defaults; main() also fits the final models
on all 9 tuning lectures and saves them to results/step4_locked.pkl.

Usage:
    python scripts/step4_viterbi.py
"""
import itertools
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from baselines import carry  # noqa: E402
from evalkit import load_gold, read_split, score, verse_number  # noqa: E402
from step3_scorer import FEATS, VERSE_COLS, model, off_X, verse_X  # noqa: E402

OUT = os.path.join(HERE, "results", "step4_locked.pkl")
V = 72
OFF = V                                    # state index of off-verse
GAMMAS = [0.25, 0.5, 1, 2, 4]
BETAS = [-5, -4, -3, -2, -1, 0, 1, 2, 3]
DELTAS = [0, 1, 2, 4]
TAUS = [0.3, 0.5, 0.7, 0.9, 1.01]          # 1.01 = never add a second verse


def near_f1(gold, pred, lectures, vids, k=1):
    """Within-k-verse P/R/F1 (reproduces results_v4 plain carry 0.831 kn test)."""
    n_p = h_p = n_g = h_g = 0
    for v in vids:
        for u in lectures[v]:
            g, p = gold[u], pred.get(u, ())
            gn = [x for x in map(verse_number, g) if x]
            pn = [x for x in map(verse_number, p) if x]
            for r in p:
                x = verse_number(r)
                n_p += 1
                h_p += r in g or (x is not None and any(abs(x - y) <= k for y in gn))
            for r in g:
                x = verse_number(r)
                n_g += 1
                h_g += r in p or (x is not None and any(abs(x - y) <= k for y in pn))
    P = h_p / n_p if n_p else 0.0
    R = h_g / n_g if n_g else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def carried_cos(F, row, refs, us, a, key="cos"):
    """cosine (key "cos") or z-score (key "z"), alpha a, of each passage to the
    verse(s) plain carry holds; 0 before the first anchor"""
    held = carry(us, {u: [refs[j] for j in np.flatnonzero(F["anchor"][row[u]])] for u in us})
    return np.array([max((F[f"{key}_{a}"][row[u], refs.index(r)] for r in held[u]),
                         default=0.0) for u in us])


def predict_locked(lecture_units, alpha="a05", path=OUT, model=None):
    """Apply the saved LOCKED Step 4 system (no gold labels used).

    lecture_units : {video: [unit_id, ...] in spoken order}
    -> {video: [(state, [extra verse indices], p_state, p_off), ...]}
    state 0..71 = verse, 72 = off; p_state = the decoded emission probability
    of that state ((1 - P_off) * share of P_verse when decoupled), p_off = P_off
    """
    if model is None:
        with open(path, "rb") as f:
            m = pickle.load(f)[alpha]
    else:
        m = model                              # e.g. the E4 model from fit_restricted
    z = np.load(FEATS)
    F = {k: z[k] for k in z.files}
    refs = list(F["refs"])
    row = {u: i for i, u in enumerate(F["unit_ids"])}
    out = {}
    for v, us in lecture_units.items():
        rows = [row[u] for u in us]
        pv = m["verse_model"].predict_proba(verse_X(F, rows, alpha, m["verse_cols"]))[:, 1]
        pv = pv.reshape(len(rows), V)
        Xo = off_X(F, rows, alpha)[:, m.get("keep_off", list(range(len(OFF_NAMES))))]
        if m["off_active"]:
            Xo = np.column_stack([Xo, carried_cos(F, row, refs, us, alpha)])
        if m.get("held_z"):
            Xo = np.column_stack([Xo, carried_cos(F, row, refs, us, alpha, "z")])
        po = m["off_model"].predict_proba(Xo)[:, 1]
        e = (1 - po)[:, None] * pv / pv.sum(1, keepdims=True) if m["decouple"] else pv
        emit = np.log(np.clip(np.column_stack([e, po]), 1e-12, 1))
        emit[:, OFF] += m["beta"]
        if not m.get("off_after_anchor", True):
            emit[np.cumsum(F["anchor"][rows].any(1)) > 0, OFF] = -1e9
        if m.get("resume"):
            states = viterbi_resume(emit, m["logA"], m["gamma"])
        else:
            states = viterbi(emit, m["logA"], m["gamma"], m["delta"], F["anchor"][rows])
        p_all = np.column_stack([e, po])
        out[v] = [(s_, [j for j in np.flatnonzero(pv[k] >= m["tau"]) if j != s_],
                float(p_all[k, s_]), float(po[k]))
                for k, s_ in enumerate(states)]
    return out


def fit_restricted(max_verse, alpha="a05", path=OUT):
    """E4: the locked system re-fit on the 9 tuning lectures using ONLY verses
    2.1 .. 2.<max_verse>. gamma, beta_off and tau are kept from the locked model
    (they are not verse-specific); the scorer weights and transition counts are
    re-estimated:
      verse scorer     (passage, verse) pairs with verse <= max_verse
      off-verse scorer passages whose gold has no verse above max_verse
      transitions      gold sequences cut wherever a verse above max_verse occurs
    Returns a model dict for predict_locked(model=...)."""
    with open(path, "rb") as f:
        m = dict(pickle.load(f)[alpha])
    z = np.load(FEATS)
    F = {k: z[k] for k in z.files}
    refs = list(F["refs"])
    row = {u: i for i, u in enumerate(F["unit_ids"])}
    lectures, gold, _ = load_gold()
    vids = [v for L in ("kn", "en") for v in read_split(L, "dev") if v in lectures]
    keep_v = np.arange(V) < max_verse
    Xv, yv, Xo, yo, seqs = [], [], [], [], []
    for v in vids:
        rows = [row[u] for u in lectures[v]]
        Y = np.zeros((len(rows), V), dtype=int)
        for k, u in enumerate(lectures[v]):
            for r in gold[u]:
                if r in refs:
                    Y[k, refs.index(r)] = 1
        X = verse_X(F, rows, alpha, m["verse_cols"]).reshape(len(rows), V, -1)
        Xv.append(X[:, keep_v].reshape(-1, X.shape[2]))
        yv.append(Y[:, keep_v].ravel())
        ok = ~Y[:, ~keep_v].any(1)
        Xo_ = off_X(F, rows, alpha)
        if m["off_active"]:
            Xo_ = np.column_stack([Xo_, carried_cos(F, row, refs, lectures[v], alpha)])
        Xo.append(Xo_[ok])
        yo.append((Y.sum(1) == 0).astype(int)[ok])
        cur = []
        for st in gold_states(Y):
            if st != OFF and st >= max_verse:
                if cur:
                    seqs.append(cur)
                cur = []
            else:
                cur.append(st)
        if cur:
            seqs.append(cur)
    m["verse_model"] = model().fit(np.concatenate(Xv), np.concatenate(yv))
    m["off_model"] = model().fit(np.concatenate(Xo), np.concatenate(yo))
    m["logA"], m["counts"] = transitions(seqs)
    m["trained_on"] = f"tuning lectures, verses 2.1-2.{max_verse} only"
    return m


def theme_f1(gold, pred, lectures, vids):
    """Strict F1 after mapping every Chapter 2 verse to its theme
    (data/verse_themes.csv); refs outside Chapter 2 are dropped."""
    from catalogue import load_themes
    th = load_themes()

    def m(refs):
        return {th[n] for n in map(verse_number, refs) if n in th}
    tg = {u: {t: "" for t in m(gold[u])} for v in vids for u in lectures[v]}
    tp = {u: sorted(m(pred.get(u, ()))) for u in tg}
    return score(tg, tp, lectures, vids)["F1"]


def gold_states(Y):
    """(T, 72) gold matrix -> state per passage: lowest verse, or OFF."""
    return [int(np.flatnonzero(y)[0]) if y.any() else OFF for y in Y]


def transitions(seqs):
    """Log transition matrix (73 x 73) from the five tied buckets."""
    c = dict(stay=0, next=0, jump=0, to_off=0, stay_off=0, from_off=0)
    for s in seqs:
        for a, b in zip(s, s[1:]):
            if a == OFF:
                c["stay_off" if b == OFF else "from_off"] += 1
            elif b == OFF:
                c["to_off"] += 1
            elif b == a:
                c["stay"] += 1
            elif b - a in (1, 2):
                c["next"] += 1
            else:
                c["jump"] += 1
    c = {k: x + 1 for k, x in c.items()}
    A = np.zeros((V + 1, V + 1))
    for a in range(V):
        nxt = [b for b in (a + 1, a + 2) if b < V]
        jmp = [b for b in range(V) if b != a and b not in nxt]
        A[a, a] = c["stay"]
        A[a, nxt] = c["next"] / len(nxt) if nxt else 0
        A[a, jmp] = c["jump"] / len(jmp)
        A[a, OFF] = c["to_off"]
    A[OFF, :V] = c["from_off"] / V
    A[OFF, OFF] = c["stay_off"]
    A /= A.sum(1, keepdims=True)
    return np.log(A), c


PRE = V                 # resume model: off-verse before the lecture's first verse
HOLD = V + 1            # resume model: HOLD + v = off-verse while holding verse v
S_RESUME = 2 * V + 1    # 145 states


def transitions_resume(seqs):
    """Log transition matrix (145 x 145) for the resume model, from GOLD sequences.

    verse v      stay, next (v+1, v+2), jump (any other verse), to_off (-> HOLD+v)
    PRE          pre_stay (PRE -> PRE), pre_exit (-> any verse)
    HOLD+u       stay_off, resume (-> u, u+1, u+2), from_off_other (-> any other verse)
    The held verse u is the last gold verse before the off-verse stretch.
    """
    c = dict(stay=0, next=0, jump=0, to_off=0, pre_stay=0, pre_exit=0,
             stay_off=0, resume=0, from_off_other=0)
    for s in seqs:
        held = None
        for a, b in zip(s, s[1:]):
            if a != OFF:
                held = a
                if b == OFF:
                    c["to_off"] += 1
                elif b == a:
                    c["stay"] += 1
                elif b - a in (1, 2):
                    c["next"] += 1
                else:
                    c["jump"] += 1
            elif held is None:
                c["pre_stay" if b == OFF else "pre_exit"] += 1
            elif b == OFF:
                c["stay_off"] += 1
            else:
                c["resume" if b - held in (0, 1, 2) else "from_off_other"] += 1
    c = {k: x + 1 for k, x in c.items()}
    A = np.zeros((S_RESUME, S_RESUME))
    for a in range(V):
        nxt = [b for b in (a + 1, a + 2) if b < V]
        jmp = [b for b in range(V) if b != a and b not in nxt]
        A[a, a] = c["stay"]
        A[a, nxt] = c["next"] / len(nxt) if nxt else 0
        A[a, jmp] = c["jump"] / len(jmp)
        A[a, HOLD + a] = c["to_off"]
        res = [b for b in (a, a + 1, a + 2) if b < V]
        other = [b for b in range(V) if b not in res]
        A[HOLD + a, HOLD + a] = c["stay_off"]
        A[HOLD + a, res] = c["resume"] / len(res)
        A[HOLD + a, other] = c["from_off_other"] / len(other)
    A[PRE, PRE] = c["pre_stay"]
    A[PRE, :V] = c["pre_exit"] / V
    A /= A.sum(1, keepdims=True)
    with np.errstate(divide="ignore"):
        return np.log(A), c


def viterbi_resume(emit, logA, gamma):
    """emit: (T, 73) as for the 73-state model. Every off state (PRE and HOLD+v)
    uses the off emission. A lecture starts in a verse or in PRE. Returns states in
    the 73-state coding (0..71 verse, 72 = OFF)."""
    E = np.column_stack([emit[:, :V], np.repeat(emit[:, [OFF]], V + 1, axis=1)])
    T = len(E)
    score_ = E[0].copy()
    score_[HOLD:] = -np.inf
    back = np.zeros((T, S_RESUME), dtype=int)
    for t in range(1, T):
        cand = score_[:, None] + gamma * logA
        back[t] = cand.argmax(0)
        score_ = cand.max(0) + E[t]
    path = [int(score_.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return [x if x < V else OFF for x in path[::-1]]


# NEXT[a, b] = 1 where a -> b is in the "next" bucket
NEXT = np.zeros((V + 1, V + 1))
for _a in range(V):
    NEXT[_a, [b for b in (_a + 1, _a + 2) if b < V]] = 1


def viterbi(emit, logA, gamma, delta=0.0, anchor=None):
    """delta: penalty on a next-verse step into a verse whose anchor did not
    fire on this passage (anchor: (T, 72) detector flags)."""
    T = len(emit)
    score_ = emit[0].copy()
    back = np.zeros((T, V + 1), dtype=int)
    for t in range(1, T):
        cand = score_[:, None] + gamma * logA          # from x to
        if delta:
            silent = np.append(1 - anchor[t], 0)        # off column never penalised
            cand = cand - delta * NEXT * silent[None, :]
        back[t] = cand.argmax(0)
        score_ = cand.max(0) + emit[t]
    path = [int(score_.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


OFF_NAMES = ("max_string", "max_cos", "max_recency", "any_anchor")   # off_X columns


def run(use_last=True, use_delta=False, alphas=("a05", "a0"), verbose=True,
        decouple=True, off_active=True, save=None,
        drop_verse=(), drop_off=(), off_after_anchor=True,
        resume=False, held_z=False, info=None, off_override=None):
    """LOLO on the tuning lectures -> (systems, gold, lectures, tune).

    use_last   : add the 2C feature `last` to the verse scorer
    use_delta  : choose delta (unanchored next-step penalty) from DELTAS, else 0
    decouple   : verse emission = log(1 - P_off) + log(P_verse(v) / sum over verses),
                so the off-verse scorer alone decides on vs off and the verse
                scorer only ranks the 72 verses
    off_active : add to the off-verse scorer the cosine to the verse plain carry
                holds (0 before the first anchor)
    save       : path; also fit the final models on ALL tuning lectures (settings
                chosen on the leave-one-lecture-out probabilities) and pickle them
    drop_verse : verse-scorer features to leave out (ablation, E6)
    drop_off   : off-verse-scorer features to leave out, names from OFF_NAMES
    off_after_anchor : False = the off-verse state is allowed only before the
                lecture's first detection (ablation, E6)
    resume     : 145-state model with "off-verse while holding verse v" states that
                 can resume v, v+1, v+2 (plan Step 4.1, bucket 5)
    held_z     : add to the off-verse scorer the z-score of the carried verse
    info       : optional dict, filled with "counts" (all 9 lectures) and "chosen"
    off_override : optional {unit_id: P_off}; replaces the off-verse scorer's
                 probability (e.g. the silver teacher's soft_target)
    The defaults are the LOCKED Step 4 system (16 Oct 2026).
    """
    trans = transitions_resume if resume else transitions
    cols = tuple(c for c in VERSE_COLS + (("last",) if use_last else ())
                if c not in drop_verse)
    keep_off = [i for i, n in enumerate(OFF_NAMES) if n not in drop_off]
    deltas = DELTAS if use_delta else [0]
    z = np.load(FEATS)
    F = {k: z[k] for k in z.files}
    refs = list(F["refs"])
    row = {u: i for i, u in enumerate(F["unit_ids"])}
    lectures, gold, _ = load_gold()

    tune = {lang: [v for v in read_split(lang, "dev") if v in lectures] for lang in ("kn", "en")}
    lang_of = {v: lang for lang, vs in tune.items() for v in vs}
    rows, Y = {}, {}
    for v in lang_of:
        rows[v] = [row[u] for u in lectures[v]]
        Y[v] = np.zeros((len(rows[v]), V), dtype=int)
        for k, u in enumerate(lectures[v]):
            for r in gold[u]:
                if r in refs:
                    Y[v][k, refs.index(r)] = 1
    seq = {v: gold_states(Y[v]) for v in lang_of}

    def active_cos(v, a):
        return carried_cos(F, row, refs, lectures[v], a)

    def oX(v, a):
        X = off_X(F, rows[v], a)[:, keep_off]
        if off_active:
            X = np.column_stack([X, active_cos(v, a)])
        if held_z:
            X = np.column_stack([X, carried_cos(F, row, refs, lectures[v], a, "z")])
        return X

    def lolo_probs(a):
        """Step 3 pooled scorers, leave-one-lecture-out -> {lecture: (Pv, Poff)}"""
        out = {}
        for v in lang_of:
            pool = [w for w in lang_of if w != v]
            Xv = np.concatenate([verse_X(F, rows[w], a, cols) for w in pool])
            yv = np.concatenate([Y[w].ravel() for w in pool])
            Xo = np.concatenate([oX(w, a) for w in pool])
            yo = np.concatenate([(Y[w].sum(1) == 0).astype(int) for w in pool])
            pv = model().fit(Xv, yv).predict_proba(verse_X(F, rows[v], a, cols))[:, 1]
            po = model().fit(Xo, yo).predict_proba(oX(v, a))[:, 1]
            if off_override is not None:
                po = np.array([off_override[u] for u in lectures[v]])
            out[v] = (pv.reshape(len(rows[v]), V), po)
        return out

    def decode(v, probs, logA, gamma, beta, delta):
        pv, po = probs[v]
        if decouple:
            pv = (1 - po)[:, None] * pv / pv.sum(1, keepdims=True)
        emit = np.log(np.clip(np.column_stack([pv, po]), 1e-12, 1))
        emit[:, OFF] += beta
        if not off_after_anchor:
            seen = np.cumsum(F["anchor"][rows[v]].any(1)) > 0
            emit[seen, OFF] = -1e9
        if resume:
            return viterbi_resume(emit, logA, gamma)
        return viterbi(emit, logA, gamma, delta, F["anchor"][rows[v]])

    def to_pred(v, probs, path, tau):
        pv = probs[v][0]
        out = {}
        for k, (u, s) in enumerate(zip(lectures[v], path)):
            if s == OFF:
                out[u] = []
                continue
            extra = [j for j in np.flatnonzero(pv[k] >= tau) if j != s]
            out[u] = [refs[s]] + [refs[j] for j in extra]
        return out

    def macro(pred, vids):
        f = [score(gold, pred, lectures, [v for v in vids if lang_of[v] == L])["F1"]
            for L in ("kn", "en") if any(lang_of[v] == L for v in vids)]
        return sum(f) / len(f)

    def choose(train, probs, logA):
        """(gamma, beta, delta, tau) maximising macro F1 over the train lectures"""
        best = None
        for gamma, beta, delta in itertools.product(GAMMAS, BETAS, deltas):
            paths = {w: decode(w, probs, logA, gamma, beta, delta) for w in train}
            for tau in TAUS:
                p = {}
                for w in train:
                    p.update(to_pred(w, probs, paths[w], tau))
                m = macro(p, train)
                if best is None or m > best[0]:
                    best = (m, gamma, beta, delta, tau)
        return best[1:]

    results, final = {}, {}
    for a in alphas:
        probs = lolo_probs(a)
        pred, chosen = {}, []
        for held in lang_of:
            train = [w for w in lang_of if w != held]
            logA, _ = trans([seq[w] for w in train])
            gamma, beta, delta, tau = choose(train, probs, logA)
            chosen.append((held, gamma, beta, delta, tau))
            pred.update(to_pred(held, probs,
                                decode(held, probs, logA, gamma, beta, delta), tau))
        if save:
            allv = list(lang_of)
            logA, counts = trans([seq[w] for w in allv])
            gamma, beta, delta, tau = choose(allv, probs, logA)
            final[a] = dict(
                verse_cols=cols, decouple=decouple, off_active=off_active,
                resume=resume, held_z=held_z, keep_off=keep_off,
                off_after_anchor=off_after_anchor,
                verse_model=model().fit(np.concatenate([verse_X(F, rows[w], a, cols) for w in allv]),
                                        np.concatenate([Y[w].ravel() for w in allv])),
                off_model=model().fit(np.concatenate([oX(w, a) for w in allv]),
                                    np.concatenate([(Y[w].sum(1) == 0).astype(int) for w in allv])),
                logA=logA, counts=counts, gamma=gamma, beta=beta, delta=delta, tau=tau,
                trained_on=allv)
            print(f"  FINAL alpha {a}: gamma={gamma} beta_off={beta} delta={delta} tau={tau}")
        results[f"Viterbi (alpha {'0.5' if a == 'a05' else '0'})"] = pred
        if a == "a05":
            scorer = {}
            for v in lang_of:
                pv = probs[v][0]
                for k, u in enumerate(lectures[v]):
                    scorer[u] = [refs[j] for j in np.flatnonzero(pv[k] >= 0.5)]
            results["scorer alone (alpha 0.5)"] = scorer
        if verbose:
            print(f"  chosen settings per held-out lecture, alpha {a}, last={use_last}:")
            for held, g, b, d, t in chosen:
                print(f"    {held:18} gamma={g:<5} beta_off={b:<3} delta={d:<2} tau={t}")

    anchors = {u: [refs[j] for j in np.flatnonzero(F["anchor"][row[u]])]
            for v in lang_of for u in lectures[v]}
    S = F["string"]
    best_anchor = {u: ([max(a_, key=lambda r: S[row[u], refs.index(r)])] if a_ else [])
                for u, a_ in anchors.items()}
    string_only, plain, plain1 = {}, {}, {}
    for v in lang_of:
        string_only.update({u: anchors[u] for u in lectures[v]})
        plain.update(carry(lectures[v], anchors))
        plain1.update(carry(lectures[v], best_anchor))
    order = [("string only", string_only), ("plain carry", plain),
            ("carry, best anchor only", plain1),
            ("scorer alone (alpha 0.5)", results["scorer alone (alpha 0.5)"])]
    order += [(k, p) for k, p in results.items() if k.startswith("Viterbi")]

    if verbose:
        _, counts = trans([seq[w] for w in lang_of])
        print(f"\n  transition bucket counts, all 9 tuning lectures (+1 smoothing): {counts}")
    if info is not None:
        info["counts"] = trans([seq[w] for w in lang_of])[1]
        info["chosen"] = chosen
    if save:
        with open(save, "wb") as f:
            pickle.dump(final, f)
        print(f"  wrote -> {os.path.relpath(save, HERE)}")
    return order, gold, lectures, tune


def main():
    order, gold, lectures, tune = run(save=OUT)
    print("\n" + "=" * 84)
    print("STEP 4 on the TUNING lectures (leave-one-lecture-out).  Themes: "
        "data/verse_themes.csv (domain expert)")
    print("=" * 84)
    print(f"  {'system':26} {'lang':4} {'P':>6} {'R':>6} {'strict F1':>10} "
        f"{'within-1 F1':>12} {'theme F1':>9}")
    for name, pred in order:
        for L, vids in tune.items():
            s = score(gold, pred, lectures, vids)
            n = near_f1(gold, pred, lectures, vids)
            t = theme_f1(gold, pred, lectures, vids)
            print(f"  {name:26} {L:4} {s['P']:>6.3f} {s['R']:>6.3f} {s['F1']:>10.3f} "
                f"{n[2]:>12.3f} {t:>9.3f}")


if __name__ == "__main__":
    main()
