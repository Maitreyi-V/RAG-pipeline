"""
Tests for gitaground_app.py / gitaground_ui.py that need NO model and NO embeddings.

The real GitaGround class is used for answer(), related(), coverage() and
verse_numbers(); only the embedding-dependent pieces (rank_verses, retrieve,
abstains, extractive, llm_summary) are replaced by tiny stand-ins, so this checks
the app's flow and display rules, not retrieval quality. The spec's acceptance
tests (section 8) that depend on the real model still have to be run once on the
machine that has the embedding caches.

    python -m pytest test_gitaground_app.py -q        (or: python test_gitaground_app.py)
"""
import csv
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))
csv.field_size_limit(10 ** 8)

import gitaground as real  # noqa: E402
import gitaground_ui as ui  # noqa: E402


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class StubGitaGround(real.GitaGround):
    """Real index, real coverage/related/answer logic; fake embeddings."""

    def __init__(self, split="all"):
        from verse_database import VERSES_BG_CH2
        from verse_translations import VERSE_TRANSLATIONS
        self.verses = {v["ref"]: v for v in VERSES_BG_CH2}
        self.refs = list(self.verses)
        self.translations = VERSE_TRANSLATIONS
        self.themes = {"BG " + r["verse_id"]: (r["theme_id"], r["theme_name"])
                       for r in _rows(os.path.join(HERE, "data", "verse_themes.csv"))}
        self.text = {}
        for name in ("units_v2.csv", "units_heldout_v1.csv"):
            for u in _rows(os.path.join(HERE, "data", name)):
                self.text[u["unit_id"]] = u["text"]
        res = os.path.join(HERE, "results")
        self.index = _rows(os.path.join(res, f"step5_index_{split}.csv"))
        self.segments = _rows(os.path.join(res, f"step5_segments_{split}.csv"))
        self.abstain_rule = None
        self.by_verse = {}
        for k, r in enumerate(self.index):
            if r["primary_verse"] != "OFF":
                self.by_verse.setdefault(r["primary_verse"], []).append(k)
        self.pp = [float(r["p_primary"]) for r in self.index]

    def shloka_matches(self, text):
        return ["BG 2.23"] if "chindanti" in text.lower() else []

    def rank_verses(self, text, q=None):
        first = [(r, "verse number") for r in self.verse_numbers(text)]
        first += [(r, "shloka match") for r in self.shloka_matches(text) if r not in dict(first)]
        rest = [r for r in self.refs if r not in dict(first)]
        return first + [(r, "meaning profile") for r in rest]

    def abstains(self, text):
        return (not self.verse_numbers(text) and not self.shloka_matches(text)
                and any(w in text.lower() for w in ("hello", "tcp")))

    def retrieve(self, text):
        top = next((r for r, _ in self.rank_verses(text) if r in self.by_verse), None)
        ks = [0, 1] + [k for k in self.by_verse.get(top, []) if k > 1][:3]
        return ks, top

    def extractive(self, text, ks, n=3):
        return [(k, self.text[self.index[k]["unit_id"]].split(". ")[0]) for k in ks[:n]]

    def llm_summary(self, text, ks):
        raise RuntimeError("no API key in tests")


# ---- helpers -------------------------------------------------------------------------
def test_position_label_and_links():
    meta = {"v": {"duration_sec": 1000.0, "youtube_id": "abc"}, "w": {"duration_sec": None,
                                                                    "youtube_id": "xyz"}}
    assert ui.position_label(0.5, 1.0, "v", meta) == "~08:20 – 16:40 (approximate)"
    assert ui.position_label(0.59, 1.0, "w", meta) == "~59%–100% of the lecture (approximate)"
    assert ui.youtube_url("v", 0.5, meta) == "https://www.youtube.com/watch?v=abc&t=500s"
    assert ui.youtube_url("w", 0.5, meta) == "https://www.youtube.com/watch?v=xyz"   # no time guessed
    assert ui.youtube_url("nope", 0.5, meta) == ""
    assert ui.mmss(3725) == "1:02:05"


def test_meta_loading_ignores_blanks():
    meta = ui.load_lecture_meta()
    assert meta["kn_sumith_day01"]["speaker"] == "Sumith"
    assert meta["kn_sumith_day01"]["youtube_id"] == "SwBF8M5tB2k"
    assert meta["kn_sumith_day01"]["duration_sec"] is None
    assert meta["video_01"]["speaker"] == ""          # blank row never invents a name


def test_grouping_neighbours_and_browse():
    g = StubGitaGround()
    lookup = ui.LectureLookup(g.index)
    cov = g.coverage("BG 2.47")
    groups = ui.group_coverage(cov)
    assert [t for t, _ in groups] == [t for t in ui.TRADITION_ORDER if any(t == t2 for t2, _ in groups)]
    for _, speakers in groups:
        for _, grp in speakers:
            eff = [float(s["effective_dwell"]) for s in grp["segments"]]
            assert eff == sorted(eff, reverse=True)       # deepest first
    k = g.by_verse["BG 2.47"][0]
    nxt = lookup.neighbour(k, 1)
    if nxt is not None:
        assert g.index[nxt]["video_file"] == g.index[k]["video_file"]
        assert int(g.index[nxt]["unit_index"]) == int(g.index[k]["unit_index"]) + 1
    seg = next(s for s in g.segments if s["primary_verse"] == "BG 2.47")
    ks = lookup.segment_ks(seg)
    assert len(ks) == int(seg["dwell"])                    # first_unit..last_unit, in order
    assert [int(g.index[x]["unit_index"]) for x in ks] == sorted(int(g.index[x]["unit_index"]) for x in ks)
    tt = ui.theme_table(g.themes)
    assert len(tt) == 9 and sum(len(v[1]) for v in tt.values()) == 72
    assert ui.bare_number("BG 2.30") == "2.30"
    # display filters only hide
    only_adv = ui.filter_coverage(cov, ["Advaita"], None, None)
    assert all(t == "Advaita" for _, t in only_adv)
    kn = ui.filter_passages(list(range(len(g.index))), g.index, None, ["kn"])
    assert all(g.index[x]["language"] == "kn" for x in kn)


# ---- the app, headless ---------------------------------------------------------------
def _app():
    from streamlit.testing.v1 import AppTest
    mod = types.ModuleType("gitaground")
    mod.GitaGround, mod.ABSTAIN = StubGitaGround, real.ABSTAIN
    sys.modules["gitaground"] = mod
    at = AppTest.from_file(os.path.join(HERE, "gitaground_app.py"), default_timeout=120)
    return at.run()


def _submit(at, text):
    at.text_area(key="query").set_value(text)
    next(b for b in at.button if b.label == "Submit").click()
    return at.run()


def _headers(at):
    return [h.value for h in at.header]


def test_app_flows():
    at = _app()
    assert not at.exception

    # a verse number: Level 1 shows only that verse, Levels 2-3 follow
    at = _submit(at, "2.47")
    assert not at.exception
    assert _headers(at)[0] == "1 · Matched verses"
    assert any("BG 2.47" in e.label for e in at.expander)
    assert sum(1 for e in at.expander if e.label.startswith("BG ")) == 1
    assert "2 · Who explains BG 2.47" in _headers(at) and "3 · Evidence" in _headers(at)
    text = " ".join(str(m.value) for m in at.markdown)
    assert "confidence" not in text.lower() and "cosine" not in text.lower()
    assert any(e.label.startswith("[1]") for e in at.expander)           # passages listed

    # off-topic: the note and nothing else
    at = _submit(at, "hello")
    assert [e.value for e in at.error] == [real.ABSTAIN]
    assert _headers(at) == [] and not [e for e in at.expander if e.label.startswith("[")]

    # a verse no indexed lecture covers: warning, Level 1 only
    g = StubGitaGround()
    missing = next((r for r in g.refs if r not in g.by_verse), None)
    if missing:
        at = _submit(at, missing.replace("BG ", ""))
        assert at.warning and _headers(at) == ["1 · Matched verses"]

    # shloka match, then a related-verse click re-runs on that verse
    at = _submit(at, "nainam chindanti shastrani nainam dahati pavakah")
    assert any(e.label.startswith("BG 2.23") for e in at.expander)
    rel = [b for b in at.button if b.key and b.key.startswith("rel_")]
    assert rel, "related verse buttons expected for BG 2.23"
    rel[0].click()
    at.run()
    assert not at.exception
    assert at.text_area(key="query").value.replace(".", "").isdigit()

    # AI summary with no key: an error box, never a crash
    at = _submit(at, "2.47")
    next(r for r in at.radio if r.label == "Summary").set_value("AI summary (GPT-4o-mini)").run()
    assert not at.exception and any("unavailable" in e.value for e in at.error)

    # display filter hides, never errors
    at.sidebar.multiselect[0].set_value(["Advaita"]).run()
    assert not at.exception
    assert all("Dvaita" != s.value for s in at.subheader)


def test_app_browse_and_about():
    at = _app()
    at.sidebar.radio(key="view").set_value("Browse by theme").run()
    assert not at.exception
    labels = at.selectbox[0].options
    assert len(labels) == 9
    g = StubGitaGround()
    tt = ui.theme_table(g.themes)
    for tid in tt:                                         # every theme and verse renders
        at.selectbox[0].select(tid).run()
        assert not at.exception
        for ref in tt[tid][1]:
            at.selectbox[1].select(ref).run()
            assert not at.exception, (tid, ref)
    at.sidebar.radio(key="view").set_value("About").run()
    assert not at.exception and any("Known limitations" in s.value for s in at.subheader)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
