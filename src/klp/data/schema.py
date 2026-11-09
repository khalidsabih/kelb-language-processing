"""Canonical schemas for KLP.

Input is per-frame dog tracks (one row per dog per frame), extracted from video
by an external detection/tracking/pose step. ``klp.data.tracks`` turns them into
interaction windows: one row per sender-receiver pair and anchor time.
"""

# Positions may use any unit that is consistent within a video (e.g. pixels);
# heading is the dog's body orientation in radians.
REQUIRED_TRACK_COLUMNS = ["video_id", "track_id", "time_s", "x", "y", "heading"]

# Optional per-frame measurements and the sender window feature each one feeds.
# Missing columns become NaN, which removes that sender modality from experiments.
TRACK_MEASUREMENT_FEATURES = {
    "audio_energy": "sender_audio_energy",
    "audio_pitch": "sender_audio_pitch",
    "pose_compactness": "sender_pose_compactness",
    "head_height": "sender_head_height",
}

ID_COLUMNS = [
    "sample_id",
    "video_id",
    "channel_id",
    "sender_id",
    "receiver_id",
    "source",
]

# Kinds of sender event detected by klp.data.events (keys of its DEFAULT_KINDS).
EVENT_KINDS = ["move_onset", "stop", "approach", "withdraw", "turn_to", "turn_away"]

# Signals produced by the sender. Only these count toward the sender increment.
SENDER_FEATURES = {
    "audio": ["sender_audio_energy", "sender_audio_pitch"],
    "pose": ["sender_pose_compactness", "sender_head_height"],
    "motion": ["sender_speed", "sender_acceleration"],
    "orientation": ["sender_facing_receiver"],
    # 1 if the sender had that event toward this receiver in the past window (klp.data.events);
    # NaN until klp.data.events.add_event_features fills them.
    "events": [f"sender_event_{kind}" for kind in EVENT_KINDS],
}

RECEIVER_PAST_FEATURES = [
    "receiver_speed_past",
    "receiver_orientation_past",
    "receiver_distance_past",
]

# Dyadic state is shared by both dogs, and context describes the scene; neither
# is sender information, so both belong in the baseline.
DYADIC_FEATURES = ["pair_distance"]
CONTEXT_FEATURES = ["visibility_score"]

TARGET_COLUMNS = [
    "delta_distance",
    "delta_orientation",
    "delta_velocity",
]

ALL_SENDER_FEATURES = [
    feature
    for feature_list in SENDER_FEATURES.values()
    for feature in feature_list
]
