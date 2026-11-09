"""Find candidate two-dog videos in the Pawgaze and K9-Bench question files.

Both benchmarks ship only YouTube IDs/URLs plus multiple-choice questions, so
this is a keyword heuristic over the questions; review the results manually.

Each video gets a tier, best first:

- ``dog_dog_strong``: 2+ questions describe dogs interacting with each other.
- ``dog_dog``: 1 such question.
- ``multi_dog``: several dogs are present, but no question describes them interacting.
- ``human_led``: several dogs, but most of those questions centre on people
  (commands, handlers, owners), so the dogs likely respond to a human, not each other.
- ``none``: no multi-dog question (only listed with ``include_all``).
"""

from __future__ import annotations

import re

import pandas as pd

_DOGS = r"(?:dogs?|pupp(?:y|ies)|canines?)"
_DOGS_PLURAL = r"(?:dogs|puppies|canines)"

MULTI_DOG_PATTERN = (
    r"\b(?:other dogs?|another dog|two dogs|three dogs|both dogs|second dog|first dog"
    r"|multiple dogs|each other|the dogs)\b"
)
DYADIC_PATTERN = "|".join([
    r"\beach other\b",
    r"\bone another\b",
    rf"\b(?:with|toward|towards|at|from|to|and) (?:the |a |an )?(?:other|another|second|first|older|younger|new) {_DOGS}\b",
    rf"\bbetween (?:the )?(?:two |both |three )?{_DOGS_PLURAL}\b",
    rf"\b(?:two|both|three) {_DOGS_PLURAL} (?:play|chase|wrestl|tug|interact|sniff|greet|fight|compet)",
    rf"\b{_DOGS}(?:'|’)? (?:play(?:ful)? (?:interaction|session|fight)|wrestling|chasing|interaction with)\b",
    rf"\bplay(?:s|ing|ed)? (?:together|with (?:the |a |an )?(?:other|another|second) {_DOGS})\b",
    r"\b(?:dog|canine)-(?:to-)?(?:dog|canine)\b",
])
# Dogs that are not physically present: on screens, in mirrors, toys.
NOT_PRESENT_PATTERN = (
    r"\b(?:videos? of|on (?:a |the )?(?:screen|tv|television|laptop|phone)|reflect\w*|mirror\w*"
    r"|stuffed|toy dogs?)\b"
)
HUMAN_PATTERN = (
    r"\b(?:human|handler|owner|man|woman|person|people|trainer|girl|boy|child|kid|family"
    r"|command|cue|signal)s?\b"
)

TIERS = ["dog_dog_strong", "dog_dog", "multi_dog", "human_led", "none"]
_YOUTUBE_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/)([\w-]{11})")


def youtube_id(value: str) -> str:
    """Extract an 11-character YouTube ID from a URL; return bare IDs unchanged."""
    match = _YOUTUBE_ID.search(value)
    return match.group(1) if match else value


def normalize_questions(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Map a Pawgaze (``scene_name``) or K9-Bench (``video_url``) table to common columns."""
    if "scene_name" in df:
        ids = df["scene_name"].astype(str)
    elif "video_url" in df:
        ids = df["video_url"].astype(str).map(youtube_id)
    else:
        raise ValueError(f"{dataset}: expected a 'scene_name' or 'video_url' column.")
    return pd.DataFrame({"video_id": ids, "dataset": dataset, "question": df["question"].astype(str)})


def classify_questions(questions: pd.DataFrame) -> pd.DataFrame:
    """Add boolean ``multi_dog``, ``dyadic`` and ``human`` flags per question."""
    text = questions.question.str.lower()
    present = ~text.str.contains(NOT_PRESENT_PATTERN, regex=True)
    return questions.assign(
        multi_dog=text.str.contains(MULTI_DOG_PATTERN, regex=True) & present,
        dyadic=text.str.contains(DYADIC_PATTERN, regex=True) & present,
        human=text.str.contains(HUMAN_PATTERN, regex=True),
    )


def apply_footage_filter(
    videos: pd.DataFrame,
    metadata: pd.DataFrame,
    min_duration_s: float = 120.0,
    landscape_only: bool = True,
) -> pd.DataFrame:
    """Merge YouTube metadata and flag footage unlikely to give long uncut tracks.

    Adds ``duration_s``, ``width``, ``height``, ``channel_id`` and ``footage_issue``:
    the first problem found (``no_metadata``, ``unavailable``, ``short``,
    ``too_short``, ``portrait``), or NaN if the video passes. In the pilot,
    YouTube Shorts cut 10-24 times a minute versus 1.5-6 for longer videos, so
    length and format stand in for cut rate, which needs the video itself.
    """
    columns = ["video_id", "available", "duration_s", "width", "height", "is_short", "channel_id"]
    merged = videos.merge(metadata[columns], on="video_id", how="left")

    issue = pd.Series(pd.NA, index=merged.index, dtype="object")
    checks = [
        ("portrait", landscape_only & (merged.height > merged.width)),
        ("too_short", merged.duration_s < min_duration_s),
        ("short", merged.is_short.eq(True)),
        ("unavailable", merged.available.eq(False)),
        ("no_metadata", merged.available.isna()),
    ]
    for name, mask in checks:  # later checks take priority
        issue[mask.fillna(False).astype(bool)] = name
    merged["footage_issue"] = issue
    return merged.drop(columns=["available", "is_short"])


def candidate_videos(questions: pd.DataFrame, include_all: bool = False) -> pd.DataFrame:
    """One row per video with question counts and a tier, best candidates first."""
    questions = classify_questions(questions)
    questions["human_multi_dog"] = questions.human & questions.multi_dog
    keys = ["video_id", "dataset"]
    videos = questions.groupby(keys).agg(
        n_questions=("question", "size"),
        n_dyadic_questions=("dyadic", "sum"),
        n_multi_dog_questions=("multi_dog", "sum"),
        n_human_led_questions=("human_multi_dog", "sum"),
    )

    tier = pd.Series("none", index=videos.index)
    multi = videos.n_multi_dog_questions > 0
    human_led = videos.n_human_led_questions > videos.n_multi_dog_questions / 2
    tier[multi & human_led] = "human_led"
    tier[multi & ~human_led] = "multi_dog"
    tier[videos.n_dyadic_questions >= 1] = "dog_dog"
    tier[videos.n_dyadic_questions >= 2] = "dog_dog_strong"
    videos["tier"] = pd.Categorical(tier, categories=TIERS, ordered=True)

    # Prefer a dyadic question as the example, then any multi-dog question.
    flagged = questions[questions.dyadic | questions.multi_dog].sort_values("dyadic", ascending=False)
    videos = videos.join(flagged.groupby(keys).question.first().rename("example_question")).reset_index()

    if not include_all:
        videos = videos[videos.tier != "none"]
    videos.insert(1, "youtube_url", "https://www.youtube.com/watch?v=" + videos.video_id)
    return videos.sort_values(
        ["tier", "n_dyadic_questions", "n_multi_dog_questions"], ascending=[True, False, False]
    ).reset_index(drop=True)
