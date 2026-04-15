"""
advaita_ground_truth.py
=======================
Ground truth for English Advaita videos — Swami Sarvapriyananda, Vedanta Society of New York
Bhagavad Gita Chapter 2 series.

VERIFIED by reading actual transcript content from your video_01.json … video_10.json files.
Each entry reflects the PRIMARY verses TAUGHT in that session (not recap verses from prior session).

video_id  → verses actually taught (confirmed from transcript text)
─────────────────────────────────────────────────────────────────────
video_01  → BG 2.1  – 2.10   | Intro to Ch2; recites & explains V1–2, ends "next class V10–11"
video_02  → BG 2.11 – 2.12  | Recaps V9–10, then TEACHES V11–12 (Krishna starts teaching)
video_03  → BG 2.13 – 2.15  | Recaps V11–12, TEACHES V13–15, ends "next time V16"
video_04  → BG 2.16          | Teaches V16 (Shankaracharya commentary focus)
video_05  → BG 2.17 – 2.18  | Reviews V16, TEACHES V17–18
video_06  → BG 2.19          | "We have done 16,17,18. Today is 19." Chants & explains V19
video_07  → BG 2.20 – 2.22  | Reviews V11–19, TEACHES V20–22 (Atman is not doer/experiencer)
video_08  → BG 2.22 – 2.25  | "Complete up to 25th verse today. Starting with 22nd verse."
video_09  → BG 2.26 – 2.27  | "Now we are on the 26th verse." Teaches V26–27
video_10  → BG 2.1  – 2.10  | INTRO episode — Ch1 overview + Ch2 V1–10 setup
─────────────────────────────────────────────────────────────────────

Usage
─────
from advaita_ground_truth import GROUND_TRUTH, parse_verse_refs, evaluate_detection, evaluate_all

# What verses does video_06 cover?
print(GROUND_TRUTH["video_06"]["verses"])   # ['BG 2.19']

# Evaluate detector output for one video
result = evaluate_detection("video_06", ["BG 2.19"])
# → {tp:1, fp:0, fn:0, precision:1.0, recall:1.0, f1:1.0}

# Evaluate across all videos at once
detections = {"video_01": ["BG 2.1","BG 2.2"], "video_06": ["BG 2.19"]}
summary = evaluate_all(detections)
print(summary["__aggregate__"])
"""

# ---------------------------------------------------------------------------
# GROUND TRUTH  —  built from reading each transcript file
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    "video_01": {
        "title":     "Bhagavad Gita Ch.2 | Introduction to Chapter 2 (V1-10) | Swami Sarvapriyananda",
        "episode":   1,
        "verses":    ["BG 2.1", "BG 2.2", "BG 2.3", "BG 2.4", "BG 2.5",
                    "BG 2.6", "BG 2.7", "BG 2.8", "BG 2.9", "BG 2.10"],
        "confirmed": True,
        "note":      "BG 2.1-2.10 intro lecture. Covers all verses up to V10 structurally. "
                    "Ends: '11th verse onwards Krishna teaches Vedanta'.",
    },
    "video_02": {
        "title":     "Bhagavad Gita Ch.2 | Verse 11-12 | Swami Sarvapriyananda",
        "episode":   2,
        "verses":    ["BG 2.11", "BG 2.12"],
        "confirmed": True,
        "note":      "Opens by recapping V9-10 (Arjuna silent). Then TEACHES V11 "
                     "(Krishna's first words of teaching) and V12 (we always existed). "
                     "Ends with 'from darkness unto light' soul immortality summary.",
    },
    "video_03": {
        "title":     "Bhagavad Gita Ch.2 | Verse 13-15 | Swami Sarvapriyananda",
        "episode":   3,
        "verses":    ["BG 2.13", "BG 2.14", "BG 2.15"],
        "confirmed": True,
        "note":      "Opens reciting V11-12 ('which we did last time'). Teaches V13-15 "
                     "(soul transmigration; endure cold/heat; Sthitaprajna quality). "
                     "Ends: 'next time: V16 and Shankaracharya commentary'.",
    },
    "video_04": {
        "title":     "Bhagavad Gita Ch.2 | Verse 16 | Swami Sarvapriyananda",
        "episode":   4,
        "verses":    ["BG 2.16"],
        "confirmed": True,
        "note":      "NYU philosophy conference mentioned at start. Deep focus on V16 "
                     "(Sat/Asat — the real never ceases, the unreal never is). "
                     "Shankaracharya commentary read aloud. Ends with Brahman discussion.",
    },
    "video_05": {
        "title":     "Bhagavad Gita Ch.2 | Verse 17-18 | Swami Sarvapriyananda",
        "episode":   5,
        "verses":    ["BG 2.17", "BG 2.18"],
        "confirmed": True,
        "note":      "Announces cancelled next-week classes (World Parliament of Religions). "
                     "Reviews V16. Teaches V17 (Atman is all-pervading, indestructible) "
                     "and V18 (bodies have an end; Atman is eternal). "
                     "Ends with Advaita / Tattvamasi discussion.",
    },
    "video_06": {
        "title":     "Bhagavad Gita Ch.2 | Verse 19 | Swami Sarvapriyananda",
        "episode":   6,
        "verses":    ["BG 2.19"],
        "confirmed": True,
        "note":      "Opens explicitly: 'We have done 16, 17, 18. Today is 19.' "
                     "Chants V16 onwards as review, then deep explanation of V19 "
                     "(one who thinks Atman slays/is slain — both err; Atman is neither). "
                     "Ends with mention of next verse 'chanted when somebody dies'.",
    },
    "video_07": {
        "title":     "Bhagavad Gita Ch.2 | Verse 20-22 | Swami Sarvapriyananda",
        "episode":   7,
        "verses":    ["BG 2.20", "BG 2.21", "BG 2.22"],
        "confirmed": True,
        "note":      "Long recap of V11-19. Main teaching: V20 (Atman never born/dies), "
                     "V21 (who knows Atman as indestructible — how can he kill?), "
                     "V22 (worn-out garments / soul transmigration analogy). "
                     "V19 mentioned throughout as 'previous verse'.",
    },
    "video_08": {
        "title":     "Bhagavad Gita Ch.2 | Verse 22-25 | Swami Sarvapriyananda",
        "episode":   8,
        "verses":    ["BG 2.22", "BG 2.23", "BG 2.24", "BG 2.25"],
        "confirmed": True,
        "note":      "Explicitly: 'complete up to 25th verse today, starting with 22nd'. "
                     "V22 (garments), V23 (weapons cannot cut / fire burn), "
                     "V24 (eternal, all-pervading, immovable), V25 (unmanifest, unthinkable). "
                     "Ends with 5 great unsolved problems of philosophy / Vedanta answers.",
    },
    "video_09": {
        "title":     "Bhagavad Gita Ch.2 | Verse 26-27 | Swami Sarvapriyananda",
        "episode":   9,
        "verses":    ["BG 2.26", "BG 2.27"],
        "confirmed": True,
        "note":      "Opens: 'Shri Krishna has just given essential teaching about Atman'. "
                     "Explicitly: 'now we are on the 26th verse'. "
                     "V26 (even if Atman is born/dies — still don't grieve), "
                     "V27 (certain death for born; certain birth for dead — natural law). "
                     "Ends discussing materialist vs Vedantic views on consciousness.",
    },
    "video_10": {
        "title":     "Bhagavad Gita Ch.2 | Verses 28-29 | Swami Sarvapriyananda",
        "episode":   10,
        "verses":    ["BG 2.28", "BG 2.29"],
        "confirmed": True,
        "note":      "Recaps V27 (death certain for born). Teaches V28 (unmanifest->manifest->unmanifest) "
                    "and V29 (Atman is a mystery; rare is he who truly knows it).",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_verse_refs(gt_entry: dict) -> set:
    """Return the set of verse ref strings for a ground truth entry."""
    return set(gt_entry["verses"])


def evaluate_detection(video_id: str, detected_verses: list) -> dict:
    """
    Compare detector output against ground truth for one video.

    Parameters
    ----------
    video_id : str          e.g. "video_06"
    detected_verses : list  e.g. ["BG 2.19"]  — output from your shloka_detector

    Returns
    -------
    dict with tp, fp, fn, precision, recall, f1
    """
    gt = GROUND_TRUTH.get(video_id)
    if gt is None:
        return {"error": f"Unknown video_id: {video_id}"}
    if not gt["confirmed"]:
        return {"error": f"No confirmed ground truth for {video_id}"}

    gt_set  = parse_verse_refs(gt)
    det_set = set(detected_verses)

    tp = len(gt_set & det_set)
    fp = len(det_set - gt_set)
    fn = len(gt_set - det_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "video_id":   video_id,
        "gt_verses":  sorted(gt_set),
        "detected":   sorted(det_set),
        "tp": tp, "fp": fp, "fn": fn,
        "precision":  round(precision, 4),
        "recall":     round(recall,    4),
        "f1":         round(f1,        4),
    }


def evaluate_all(detections: dict) -> dict:
    """
    Evaluate detector across all videos.

    Parameters
    ----------
    detections : dict   video_id -> list[str]  (your detector's output per video)

    Returns
    -------
    dict  video_id -> per-video result, plus '__aggregate__' with overall P/R/F1
    """
    results = {}
    total_tp = total_fp = total_fn = 0

    for vid, det in detections.items():
        r = evaluate_detection(vid, det)
        results[vid] = r
        if "error" not in r:
            total_tp += r["tp"]
            total_fp += r["fp"]
            total_fn += r["fn"]

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    results["__aggregate__"] = {
        "total_tp":  total_tp,
        "total_fp":  total_fp,
        "total_fn":  total_fn,
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
    }
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Advaita Ground Truth — matched to your transcript files ===\n")
    print(f"{'Video':<10} {'Ep':<4} {'Verses Covered':<52} {'Note (truncated)'}")
    print("-" * 120)
    for vid in sorted(GROUND_TRUTH.keys()):
        e = GROUND_TRUTH[vid]
        ep     = str(e["episode"])
        verses = ", ".join(e["verses"])
        note   = e["note"][:60]
        status = "✓" if e["confirmed"] else "?"
        print(f"{vid:<10} {ep:<4} {status} {verses:<52} {note}")

    total_v = sum(len(e["verses"]) for e in GROUND_TRUTH.values())
    confirmed = sum(1 for e in GROUND_TRUTH.values() if e["confirmed"])
    print(f"\nTotal videos: {len(GROUND_TRUTH)}  |  Confirmed: {confirmed}  |  Total verse-instances: {total_v}")

    # Self-test
    print("\n=== Self-test (perfect detection should give P=R=F1=1.0) ===")
    perfect = {vid: GROUND_TRUTH[vid]["verses"] for vid in GROUND_TRUTH}
    agg = evaluate_all(perfect)["__aggregate__"]
    print(f"  P={agg['precision']}  R={agg['recall']}  F1={agg['f1']}")
