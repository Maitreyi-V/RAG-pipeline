"""
GitaGround core: the LOCKED Step 6 retrieval and the three display levels,
read from the Step 5 index. The app (gitaground_app.py) and the evaluation
(scripts/step6_retrieval.py) both call this, so what is shown is what is scored.

Input -> verse
  verse number    "2.45", "BG 2.45", "Gita 2.45", "verse 45" -> that verse directly
  pasted shloka   the existing string matchers (Kannada script ShlokaDetector;
                  Devanagari is first transliterated to Kannada script; Roman
                  letters RomanShlokaDetector). If they fire, those verses
                  come first.
  anything else   rank the 72 verses by cosine to the Step 2 meaning profile
                  (alpha 0.5, built from detections in the 39 original lectures)

Verse -> 5 passages (LOCKED 16 Oct 2026, chosen on the 74 dev questions)
  plain vector search's top 2, then 3 passages whose Step 4 primary verse is
  the top verse, ranked by cosine * p_primary.

Levels
  1  matched verses, theme, related verses (verses that co-occur with it in the
     index as primary / secondary on the same passage)
  2  the top verse's segments grouped by speaker and tradition, with dwell and
     effective dwell; passages that only mention it as a secondary verse are
     "passing"
  3  the 5 passages, lecture, position in the lecture (approximate: share of
     the lecture's words; no durations yet), plus an extractive summary (the 3
     sentences closest to the input) or an LLM summary restricted to them

Abstention: when the input names no verse and matches no shloka, the app
abstains if its abstention score is below the threshold in
results/e7_abstain.json (signal and threshold set by scripts/freeze_e7.py on the
DEVELOPMENT questions only). Without that file it never abstains.
Not built: calibrated confidence bands.
"""
import json
import contextlib
import csv
import io
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from embed_scores import unit_cache  # noqa: E402
from evalkit import UNITS, UNITS_HELDOUT, load_csv  # noqa: E402
from step2_features import anchor_profile_sums, normalise, verse_docs  # noqa: E402

ALPHA = 0.5
K = 5
N_VECTOR = 2
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
KANNADA = re.compile(r"[ಀ-೿]")
VERSE_REF = [re.compile(r"(?<![\d.])2\s*[.:]\s*(\d{1,2})(?![\d.])"),
             re.compile(r"\b(?:verse|shloka|sloka|śloka)\s+(\d{1,2})\b", re.I)]
MIN_LETTERS = 15          # chosen by hand, not measured
ABSTAIN = "I could not find sufficient support for this question in the available lectures."


class GitaGround:
    def __init__(self, split="dev"):
        from embeddings import embed_texts
        from roman_detector import RomanShlokaDetector
        from shloka_detector import ShlokaDetector
        from verse_database import VERSES_BG_CH2
        from verse_translations import VERSE_TRANSLATIONS

        self.embed = lambda texts: normalise(np.asarray(embed_texts(texts), dtype=np.float32))
        self.verses = {v["ref"]: v for v in VERSES_BG_CH2}
        self.refs = list(self.verses)
        self.translations = VERSE_TRANSLATIONS
        self.kn_det = ShlokaDetector(sanskrit_threshold=0.12, match_threshold=45)
        self.en_det = RomanShlokaDetector(match_threshold=55)
        self.themes = {"BG " + r["verse_id"]: (r["theme_id"], r["theme_name"])
                       for r in load_csv(os.path.join(HERE, "data", "verse_themes.csv"))}

        # unit grids, embeddings, Step 2 anchors -> meaning profiles
        units, E = [], []
        for path in (UNITS, UNITS_HELDOUT):
            units += load_csv(path)
            E.append(np.load(unit_cache(path))["emb"])
        E = np.concatenate(E)
        f2 = np.load(os.path.join(HERE, "results", "step2_features.npz"))
        assert list(f2["unit_ids"]) == [u["unit_id"] for u in units]
        row = {u["unit_id"]: i for i, u in enumerate(units)}
        original = {u["video_file"] for u in load_csv(UNITS)}
        lectures = defaultdict(list)
        for u in sorted(units, key=lambda u: (u["video_file"], int(u["unit_index"]))):
            lectures[u["video_file"]].append(row[u["unit_id"]])
        sums, _ = anchor_profile_sums(E, lectures, original, f2["anchor"])
        corpus = normalise(sum(sums.values()).astype(np.float32))
        self.mu = normalise((1 - ALPHA) * corpus + ALPHA * self.embed(verse_docs()))
        self.text = {u["unit_id"]: u["text"] for u in units}

        # Step 5 index
        res = os.path.join(HERE, "results")
        self.index = load_csv(os.path.join(res, f"step5_index_{split}.csv"))
        self.segments = load_csv(os.path.join(res, f"step5_segments_{split}.csv"))
        self.P = np.stack([E[row[r["unit_id"]]] for r in self.index])
        self.pp = np.array([float(r["p_primary"]) for r in self.index])
        path = os.path.join(res, "e7_abstain.json")
        self.abstain_rule = json.load(open(path)) if os.path.exists(path) else None
        r = self.abstain_rule                   # a passage-based threshold depends on the
        if r and r["signal"] == "passage" and r.get("index") != split:   # index; a profile
            self.abstain_rule = None            # one does not
        self.by_verse = defaultdict(list)
        for k, r in enumerate(self.index):
            if r["primary_verse"] != "OFF":
                self.by_verse[r["primary_verse"]].append(k)

    # ---- input -> verses ------------------------------------------------------
    def verse_numbers(self, text):
        """Explicit references to Chapter 2 verses in the input, in order."""
        found = []
        for pat in VERSE_REF:
            for m in pat.finditer(text):
                ref = f"BG 2.{int(m.group(1))}"
                if ref in self.verses and ref not in found:
                    found.append(ref)
        return found

    def shloka_matches(self, text):
        """Verses the string matchers find in the input, best first. Verse
        references are removed first, and inputs with fewer than MIN_LETTERS
        letters left are not matched (fuzzy partial matching of a 2-letter
        string like "bg" against a verse is meaningless)."""
        for pat in VERSE_REF:
            text = pat.sub(" ", text)
        text = re.sub(r"\b(?:bg|gita|bhagavad)\b", " ", text, flags=re.I)
        if sum(c.isalpha() for c in text) < MIN_LETTERS:
            return []
        if DEVANAGARI.search(text):
            from indic_transliteration.sanscript import DEVANAGARI as D, KANNADA as KN, transliterate
            text = transliterate(text, D, KN)
        if KANNADA.search(text):
            with contextlib.redirect_stdout(io.StringIO()):
                hits = self.kn_det.detect(text)
            hits = sorted(hits, key=lambda d: -d["match_score"])
        else:
            hits = self.en_det.detect(text)
        return list(dict.fromkeys(d["verse_ref"] for d in hits))

    def rank_verses(self, text, q=None):
        """[(ref, how), ...] all 72 verses: verse numbers, then string matches,
        then the rest by meaning profile."""
        q = self.embed([text])[0] if q is None else q
        first = [(r, "verse number") for r in self.verse_numbers(text)]
        first += [(r, "shloka match") for r in self.shloka_matches(text)
                  if r not in dict(first)]
        seen = dict(first)
        rest = [self.refs[j] for j in np.argsort(-(self.mu @ q)) if self.refs[j] not in seen]
        return first + [(r, "meaning profile") for r in rest]

    # ---- verses -> passages ---------------------------------------------------
    def retrieve(self, text):
        """LOCKED: plain top 2 + 3 from the top verse that has passages.
        Returns (index rows, top verse)."""
        q = self.embed([text])[0]
        sim = self.P @ q
        out = list(np.argsort(-sim)[:N_VECTOR])
        top = next((r for r, _ in self.rank_verses(text, q) if r in self.by_verse), None)
        if top:
            ks = sorted(self.by_verse[top], key=lambda k: -sim[k] * self.pp[k])
            out += [k for k in ks if k not in out][:K - len(out)]
        out += [k for k in np.argsort(-sim) if k not in out][:K - len(out)]
        return [int(k) for k in out[:K]], top

    def abstain_scores(self, text, q=None):
        """the two candidate abstention signals: best passage cosine in the index,
        best meaning-profile cosine over the 72 verses"""
        q = self.embed([text])[0] if q is None else q
        return {"passage": float((self.P @ q).max()), "profile": float((self.mu @ q).max())}

    def abstains(self, text):
        """True if the input names no verse, matches no shloka, and its abstention
        score is below the locked threshold"""
        if not self.abstain_rule or self.verse_numbers(text) or self.shloka_matches(text):
            return False
        r = self.abstain_rule
        return self.abstain_scores(text)[r["signal"]] < r["threshold"]

    def answer(self, text):
        """What the app shows. Never substitutes a verse silently.

        Returns dict(ranked, verse, passages, note, query_for_summary):
          verse     the verse Levels 2-3 are about, or None
          passages  index rows for Level 3 ([] when nothing honest to show)
          note      a message to show the user, or ""
        """
        ranked = self.rank_verses(text)
        if self.abstains(text):
            return dict(ranked=ranked, verse=None, passages=[], query_for_summary=None,
                        note=ABSTAIN)
        first, how = ranked[0]
        if how in ("verse number", "shloka match"):
            if first not in self.by_verse:
                return dict(ranked=ranked, verse=None, passages=[], query_for_summary=None,
                            note=f"{first} is not discussed in any indexed lecture "
                                 f"({len(self.by_verse)} of 72 verses are covered so far).")
            if how == "verse number":           # the input has no content to match on
                ks = sorted(self.by_verse[first], key=lambda k: -self.pp[k])[:K]
                return dict(ranked=ranked, verse=first, passages=ks, note="",
                            query_for_summary=self.translations[first][0])
        ks, top = self.retrieve(text)
        note = "" if top == first else (
            f"{first} (best match) is not discussed in any indexed lecture; the passages "
            f"below are for {top}, the closest verse that is.")
        return dict(ranked=ranked, verse=top, passages=ks, note=note, query_for_summary=text)

    # ---- levels ---------------------------------------------------------------
    def related(self, ref):
        """Verses on the same passages as ref (primary <-> secondary), most frequent first."""
        c = Counter()
        for r in self.index:
            sec = [s for s in r["secondary_verses"].split(";") if s]
            if r["primary_verse"] == ref:
                c.update(sec)
            elif ref in sec and r["primary_verse"] != "OFF":
                c[r["primary_verse"]] += 1
        return c.most_common()

    def coverage(self, ref):
        """Level 2: {(speaker, tradition): {"segments": [...], "passing": n}}"""
        meta = {r["video_file"]: (r["speaker"], r["tradition"]) for r in self.index}
        out = defaultdict(lambda: {"segments": [], "passing": 0})
        for s in self.segments:
            if s["primary_verse"] == ref:
                out[meta[s["video_file"]]]["segments"].append(s)
        for r in self.index:
            if ref in r["secondary_verses"].split(";") and r["primary_verse"] != ref:
                out[(r["speaker"], r["tradition"])]["passing"] += 1
        for g in out.values():
            g["segments"].sort(key=lambda s: -float(s["effective_dwell"]))
        return dict(out)

    def extractive(self, text, ks, n=3):
        """The n sentences of the retrieved passages closest to the input (no LLM)."""
        sents = [(k, s.strip()) for k in ks
                 for s in re.split(r"(?<=[.?!।])\s+", self.text[self.index[k]["unit_id"]])
                 if len(s.split()) >= 5]
        if not sents:
            return []
        S = self.embed([s for _, s in sents])
        q = self.embed([text])[0]
        return [sents[j] for j in np.argsort(-(S @ q))[:n]]

    def llm_summary(self, text, ks):
        """Summary restricted to the retrieved passages; abstains if they do not answer."""
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        ctx = "\n\n".join(f"[{i + 1}] ({self.index[k]['speaker']}, {self.index[k]['video_file']})\n"
                          f"{self.text[self.index[k]['unit_id']]}" for i, k in enumerate(ks))
        prompt = ("Answer the question using ONLY the lecture passages below. Cite passages as "
                  "[1]..[5]. Do not add any verse, fact or interpretation that is not in the "
                  f"passages. If they do not answer the question, reply exactly: \"{ABSTAIN}\"\n\n"
                  f"Question: {text}\n\nPassages:\n{ctx}")
        r = OpenAI().chat.completions.create(model="gpt-4o-mini", temperature=0,
                                             messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content
