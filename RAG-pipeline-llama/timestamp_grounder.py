"""
Timestamp Grounding Module
============================
Links Q&A answers back to specific moments in the original discourse videos.

When the system answers a question citing BG 2.13, this module provides:
  "Listen to the original explanation: Video 5, 12:34–18:02"
  with a clickable YouTube link that starts at the right timestamp.

Approach:
  - Store approximate timestamps in chunk metadata (from annotation or Whisper)
  - When a chunk is retrieved, look up its timestamp range
  - Generate a YouTube deeplink with ?t=XXs parameter
  - Only show timestamps when confidence is high and source is relevant

Usage:
    from timestamp_grounder import TimestampGrounder
    grounder = TimestampGrounder(video_index)
    grounded_answer = grounder.add_timestamps(answer, retrieved_chunks)
"""
import re
import json


class VideoMetadata:
    """Stores metadata for a single video including timestamps."""
    def __init__(self, video_id, title, youtube_url=None, duration_sec=None):
        self.video_id = video_id
        self.title = title
        self.youtube_url = youtube_url  # full URL or just video ID
        self.duration_sec = duration_sec
        self.verse_timestamps = {}  # verse_ref -> (start_sec, end_sec)

    def add_verse_timestamp(self, verse_ref, start_sec, end_sec):
        self.verse_timestamps[verse_ref] = (start_sec, end_sec)

    def get_youtube_link(self, start_sec):
        """Generate YouTube deep link with timestamp."""
        if not self.youtube_url:
            return None
        base = self.youtube_url.split("?")[0]  # strip existing params
        return f"{base}?t={int(start_sec)}s"

    def format_timestamp(self, seconds):
        """Convert seconds to MM:SS format."""
        m, s = divmod(int(seconds), 60)
        return f"{m}:{s:02d}"


class TimestampGrounder:
    """Adds source video timestamps to Q&A answers."""

    def __init__(self, video_metadata_path=None):
        """
        Args:
            video_metadata_path: path to JSON file with video metadata + timestamps
        """
        self.videos = {}  # video_file -> VideoMetadata
        if video_metadata_path:
            self.load_metadata(video_metadata_path)

    def load_metadata(self, path):
        """Load video metadata from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            vid = VideoMetadata(
                video_id=entry["video_file"],
                title=entry.get("title", entry["video_file"]),
                youtube_url=entry.get("youtube_url"),
                duration_sec=entry.get("duration_sec"),
            )
            for vt in entry.get("verse_timestamps", []):
                vid.add_verse_timestamp(
                    vt["verse_ref"],
                    vt["start_sec"],
                    vt["end_sec"]
                )
            self.videos[entry["video_file"]] = vid

    def register_video(self, video_file, youtube_url=None, title=None):
        """Register a video manually."""
        self.videos[video_file] = VideoMetadata(
            video_id=video_file,
            title=title or video_file,
            youtube_url=youtube_url
        )
        return self.videos[video_file]

    def find_timestamp(self, video_file, verse_ref):
        """Look up the timestamp for a verse in a video."""
        vid = self.videos.get(video_file)
        if not vid:
            return None

        ts = vid.verse_timestamps.get(verse_ref)
        if not ts:
            return None

        start_sec, end_sec = ts
        return {
            "video_file": video_file,
            "video_title": vid.title,
            "verse_ref": verse_ref,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_formatted": vid.format_timestamp(start_sec),
            "end_formatted": vid.format_timestamp(end_sec),
            "youtube_link": vid.get_youtube_link(start_sec),
            "duration_sec": end_sec - start_sec,
        }

    def ground_retrieved_chunks(self, retrieved_chunks):
        """
        Given a list of retrieved chunks (from the retriever),
        find timestamps for each and return grounding info.
        Only returns timestamps where we have confident data.
        """
        groundings = []

        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            video_file = meta.get("video_file", "")
            verse_ref = meta.get("verse_ref", "")

            if not video_file or not verse_ref:
                continue

            ts = self.find_timestamp(video_file, verse_ref)
            if ts:
                ts["chunk_id"] = chunk.get("chunk_id", "")
                ts["retrieval_similarity"] = chunk.get("similarity", 0)
                groundings.append(ts)

        return groundings

    def format_timestamp_citations(self, groundings, max_citations=3):
        """
        Format timestamp groundings as readable citations for the answer.
        Returns a string to append to the LLM's answer.
        """
        if not groundings:
            return ""

        # Sort by retrieval similarity (most relevant first)
        groundings = sorted(groundings, key=lambda g: g.get("retrieval_similarity", 0),
                          reverse=True)[:max_citations]

        lines = ["\n📺 **Listen to the original discourse:**"]
        for g in groundings:
            video_title = g["video_title"]
            start = g["start_formatted"]
            end = g["end_formatted"]
            verse = g["verse_ref"]

            if g.get("youtube_link"):
                lines.append(
                    f"  → [{verse}] {video_title} ({start}–{end}) — "
                    f"[▶ Watch]({g['youtube_link']})"
                )
            else:
                lines.append(
                    f"  → [{verse}] {video_title} ({start}–{end})"
                )

        return "\n".join(lines)

    def enhance_answer(self, answer, retrieved_chunks):
        """
        Main entry point: take an LLM answer and retrieved chunks,
        add timestamp citations if available.
        """
        groundings = self.ground_retrieved_chunks(retrieved_chunks)

        if not groundings:
            return answer  # no timestamps available, return as-is

        citations = self.format_timestamp_citations(groundings)
        return answer + "\n" + citations


# --- Sample video metadata (update with real YouTube URLs and timestamps) ---
SAMPLE_VIDEO_METADATA = [
    {
        "video_file": "gita-1st_video",
        "title": "Bhagavad Gita Ch.2 – Part 1 (BG 2.1–2.3)",
        "youtube_url": "",  # Fill with actual YouTube URL
        "duration_sec": 1500,
        "verse_timestamps": [
            {"verse_ref": "BG 2.1", "start_sec": 180, "end_sec": 540},
            {"verse_ref": "BG 2.2", "start_sec": 540, "end_sec": 900},
            {"verse_ref": "BG 2.3", "start_sec": 900, "end_sec": 1350},
        ]
    },
    {
        "video_file": "gita-2nd_video",
        "title": "Bhagavad Gita Ch.2 – Part 2 (BG 2.4–2.6)",
        "youtube_url": "",
        "duration_sec": 1500,
        "verse_timestamps": [
            {"verse_ref": "BG 2.4", "start_sec": 120, "end_sec": 500},
            {"verse_ref": "BG 2.5", "start_sec": 500, "end_sec": 880},
            {"verse_ref": "BG 2.6", "start_sec": 880, "end_sec": 1350},
        ]
    },
    {
        "video_file": "gita-3rd_video",
        "title": "Bhagavad Gita Ch.2 – Part 3 (BG 2.7–2.9)",
        "youtube_url": "",
        "duration_sec": 1500,
        "verse_timestamps": [
            {"verse_ref": "BG 2.7", "start_sec": 150, "end_sec": 550},
            {"verse_ref": "BG 2.8", "start_sec": 550, "end_sec": 900},
            {"verse_ref": "BG 2.9", "start_sec": 900, "end_sec": 1350},
        ]
    },
    {
        "video_file": "gita-4th_video",
        "title": "Bhagavad Gita Ch.2 – Part 4 (BG 2.10–2.11)",
        "youtube_url": "",
        "duration_sec": 1500,
        "verse_timestamps": [
            {"verse_ref": "BG 2.10", "start_sec": 120, "end_sec": 650},
            {"verse_ref": "BG 2.11", "start_sec": 650, "end_sec": 1350},
        ]
    },
    {
        "video_file": "gita-5th_video",
        "title": "Bhagavad Gita Ch.2 – Part 5 (BG 2.12–2.13)",
        "youtube_url": "",
        "duration_sec": 1500,
        "verse_timestamps": [
            {"verse_ref": "BG 2.12", "start_sec": 150, "end_sec": 700},
            {"verse_ref": "BG 2.13", "start_sec": 700, "end_sec": 1350},
        ]
    },
    {
        "video_file": "gita-6th_video",
        "title": "Bhagavad Gita Ch.2 – Part 6 (BG 2.14)",
        "youtube_url": "",
        "duration_sec": 1500,
        "verse_timestamps": [
            {"verse_ref": "BG 2.14", "start_sec": 120, "end_sec": 1300},
        ]
    },
]


def create_sample_metadata(output_path="data/video_metadata.json"):
    """Write sample metadata file for testing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_VIDEO_METADATA, f, ensure_ascii=False, indent=2)
    print(f"Created sample metadata: {output_path}")
    print("  → Update youtube_url fields with actual URLs")
    print("  → Update verse_timestamps with actual times from videos")


if __name__ == "__main__":
    import os
    create_sample_metadata("data/video_metadata.json")

    # Demo
    grounder = TimestampGrounder("data/video_metadata.json")

    # Simulate retrieved chunks
    fake_chunks = [
        {
            "chunk_id": "chunk_021",
            "similarity": 0.85,
            "metadata": {"video_file": "gita-5th_video", "verse_ref": "BG 2.12"}
        },
        {
            "chunk_id": "chunk_022",
            "similarity": 0.72,
            "metadata": {"video_file": "gita-5th_video", "verse_ref": "BG 2.13"}
        },
    ]

    fake_answer = "Krishna declares that the soul is eternal and never ceases to exist."

    enhanced = grounder.enhance_answer(fake_answer, fake_chunks)
    print("\nEnhanced answer:")
    print(enhanced)
