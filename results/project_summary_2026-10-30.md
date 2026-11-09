# KLP project summary (2026-10-30)

**Question:** does sender-dog information improve prediction of a receiver dog's future behavior beyond the receiver's own history?

**Answer so far:** no measurable sender effect in the regression over all windows. Specific sender events (cycle 5) do show receiver responses, but only in an exploratory analysis that needs confirmation. Both results rest on little data: the regression keeps only 32 videos, 8 channels and about 2,300 windows after filtering, and each event cell has 14–75 events. The data, not the method, is the bottleneck.

## 1. Analysis code (`src/klp`, 38 tests passing)

- Per-frame dog tracks become sender→receiver windows: 1 s of history, then the receiver's change in distance, orientation and speed.
- Windows never cross camera cuts or tracking gaps; distances are in body sizes; splits keep each YouTube channel on one side.
- Compares a baseline (receiver's own history plus pair distance) with baseline plus sender features, and drops one sender modality at a time.

## 2. Data and tracking

- From Pawgaze questions: 296 candidate videos, 125 usable, from only 18 channels.
- 59 videos tracked with YOLO11s + ByteTrack.
- K9-Bench access is pending (Hugging Face request and login).

## 3. Dog pose

- Trained YOLO11s-pose on Stanford Dog-Pose (pose mAP50 0.83).
- Added to extraction: runs on a crop of each tracked dog; keypoints for over 90% of dogs.
- Body heading about 90% correct on 61 hand-labelled side-on dogs; head direction is unreliable.

## 4. Cycle 1

- Sender effect about zero. The default interaction model overfit, so its numbers came out negative.
- Details: `pilot_experiments_2026-10-30.md`.

## 5. Cycle 2

1. **Distance target:** now uses the same along-the-line measure as the past feature, so a sideways step no longer counts as moving away.
2. **Nulls:** sender features shuffled within the video, and time-shifted within the pair. They showed that cycle 1's small gain (+0.009 R²) was video-level: shuffled sender data scored the same (+0.008, p = 0.43).
3. **Interaction filter:** pairs tracked continuously for at least 5 s and at most 3 body sizes apart.
4. **Camera motion:** new `scripts/pilot/stabilize_tracks.py`. Motion shared by all dogs fell from 43% to 33% overall, and from 75% to 25% on a handheld pack walk.
5. **Response-lag sweep (0–3 s):** no sender gain at any delay; increments about −0.01, p = 1.0 against the shuffled null.
6. **Sender move onsets:** 398 detected, but only 28 occur within the filtered pairs. Receiver responses after them don't differ from control times; every interval includes zero.

Details: `pilot_experiments_2026-10-30_cycle2.md` (items 1–4) and `pilot_experiments_2026-10-30_cycle2_lag_events.md` (items 5–6).

## Since then

- The plain linear ridge is now the default model (`model: linear` in `configs/v0.yaml`).
- **Cycle 3, audio (2026-10-31):** soundtracks scored with AST (`scripts/pilot/extract_audio.py`). It's unusable on this footage:
  - most soundtracks are music or narration, and only 1 video has its own sound;
  - in the clean subset, a dog sound precedes only 18 windows (10 moments);
  - the mouth-opening cue for attributing a sound to one dog doesn't respond to barks.

  No audio experiment was run. Details: `pilot_audio_2026-10-31.md`.
- **Cycle 5, communication events (2026-10-31):** first positive, exploratory result.
  - Six kinds of sender event were detected per pair: move onset, stop, approach, withdraw, turn toward, turn away (1,180 events in total).
  - Receiver responses were compared with matched control moments.
  - 8 of 54 tests pass Holm correction and hold when any one channel is dropped and when frames without camera stabilization are removed. They point to following and yielding: after the sender withdraws, the receiver speeds up and moves toward where it was; after the sender stops or approaches, the receiver slows; after the sender turns toward it, the receiver moves away.
  - Each cell has only 14–75 events from 4–9 channels.
  - A common cause that reaches the two dogs in turn (for example a person) is not ruled out. A follow-up found that receivers with the sender behind them respond as strongly as receivers facing it (difference +0.005 SD, p = 0.56). That is what a shared cue predicts, so the check gives no support for the communication reading. The test detects only large differences (about 0.8 SD).
  - Adding event indicators to the regression as a sender modality gives no gain on held-out channels; on the clean set, events alone change R² by −0.011 against −0.010 for shuffled indicators, p = 0.71. That's expected: with events in about 6% of windows, even real effects would add at most about 0.005 R².
  - Needs confirmation on new data. Details: `pilot_communication_events_2026-10-31.md`.

## Not done

- Items 8–9: a better pose representation, nonlinear temporal models.
- Decisions pending:
  - better footage, ideally own recordings of continuous two-dog interactions;
  - whether to use YouTube cookies for the 45 bot-blocked videos;
  - K9-Bench access.
