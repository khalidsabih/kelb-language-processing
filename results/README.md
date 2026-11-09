# KLP v0 — Pilot Results

This folder contains the exploratory pilot analyses for KLP v0.

> **Interpretation boundary:** these results describe predictive and event-response structure in observational dog-interaction footage. They do **not** establish causation, communication, semantic meaning, intention, or "dog language."

## Analysis subsets

The pilot uses different subsets for different questions. They should not be treated as one common sample:

- **59 videos** were processed for tracking/audio feasibility work.
- **51 videos** contributed behavioral events before event-specific matching/exclusions.
- The clean broad-prediction analysis retained **2,322 windows from 32 videos and 8 YouTube channels**.

## Results at a glance

| Question | Result | Interpretation |
|---|---|---|
| Does Dog A improve broad prediction of Dog B? | **No reliable sender advantage** | Visual sender features did not improve held-out-channel prediction beyond Dog B's own history plus dyadic/context features. |
| Does the broad result depend on one model, target, or short lag? | **No** | The null broad result persisted across linear/interaction ridge checks, all three targets, and 0–3 s response lags. |
| Are some sender behavioral events followed by structured changes in Dog B? | **Yes, exploratory** | 8 of 54 matched event-response tests survived Holm correction. The directions were discovered post hoc and cells are small. |
| Do event indicators improve global prediction on unseen channels? | **No** | Sparse event indicators were no better than shuffled indicators in clean, relaxed, or event-only regression tests. |
| Do the event effects support a simple visual-signal account? | **No clear support** | Receivers oriented toward the sender responded about as strongly as receivers oriented away; the test only had power for large facing effects. |
| Can the pilot support sender-specific audio analysis? | **No** | Shared microphones, music/narration, sparse dog vocalizations, and failed vocalizer attribution prevent a sender-audio test. |
| Main next step | **Better identifiable multimodal data** | Replicate preregistered event hypotheses using synchronized video, attributable audio, known identities, and explicit measurement of shared causes. |

## 1. Broad sender prediction

Clean cycle-2 analysis:

- **2,322 windows**
- **32 videos**
- **8 YouTube channels**
- **10 held-out-channel splits**
- 1 s past -> 1 s future, 0.5 s stride
- camera-stabilized tracks with close/continuous pair filtering

Linear ridge:

| Metric | Value |
|---|---:|
| Receiver-only baseline R² | 0.0516 ± 0.0333 |
| Sender + receiver R² | 0.0438 ± 0.0340 |
| Sender increment R² | **-0.0078 ± 0.0072** |
| Shuffled-sender null mean | +0.0022 |
| p vs. shuffled sender | 1.00 |

The three target-specific sender increments were also non-positive:

| Target | Sender increment R² |
|---|---:|
| Distance | -0.0121 ± 0.0128 |
| Orientation | -0.0011 ± 0.0032 |
| Velocity | -0.0103 ± 0.0231 |

An interaction-ridge check also did not show a reliable sender advantage (increment **-0.0421 ± 0.0524**, shuffle-null p = 0.33).

### Response-lag robustness

The receiver target window was shifted later by 0, 0.5, 1, 2, and 3 s. Linear-ridge sender increments were:

`-0.0078, -0.0095, -0.0096, -0.0109, -0.0334`

At every lag, **p = 1.0 against the shuffled-sender null**.

The circular time-shift null in the clean dataset retained only 336 eligible windows, and all models there had negative R²; it is therefore not treated as interpretable evidence.

**Conclusion:** no measurable global sender advantage in the current clean pilot, and no hidden broad effect appeared in the tested short response lags.

Detailed reports:
- [pilot_experiments_2026-10-30_cycle2.md](pilot_experiments_2026-10-30_cycle2.md)
- [pilot_experiments_2026-10-30_cycle2_lag_events.md](pilot_experiments_2026-10-30_cycle2_lag_events.md)

## 2. Why the earlier apparent gain was not retained

An earlier, less controlled linear analysis showed a small sender increment of about **+0.0086 R²**. The shuffled-sender null on those same windows was **+0.0076** (p = 0.43).

After camera stabilization, corrected targets, close/continuous pair filtering, and the final null comparison, the apparent gain disappeared.

**Interpretation:** the early positive-looking result was consistent with video/source-level structure or shared context rather than moment-to-moment sender information. It is kept in the repository as a historical analysis, not as the canonical KLP v0 conclusion.

## 3. Event-centered analysis

The event pipeline detected **1,180 behavioral events** across six observable event types:

- move onset
- stop
- approach
- withdraw
- turn toward
- turn away

Receiver responses were measured at 0, 1, and 2 s lags and compared with matched control moments from the same video. Placebo p-values were Holm-adjusted across 54 tests.

Eight cells survived Holm correction:

| Lag | Sender event | Receiver response | Estimate | d | Events / channels | Holm p |
|---:|---|---|---:|---:|---:|---:|
| 0 s | move onset | moves toward sender position | -0.074 | -0.36 | 75 / 9 | 0.027 |
| 0 s | stop | slows | -0.139 | -0.70 | 60 / 8 | 0.027 |
| 1 s | turn toward | moves away | +0.065 | +0.47 | 44 / 8 | 0.047 |
| 1 s | approach | slows | -0.117 | -0.59 | 23 / 7 | 0.027 |
| 1 s | withdraw | speeds up | +0.088 | +0.43 | 25 / 5 | 0.027 |
| 2 s | approach | moves toward sender position | -0.167 | -1.16 | 14 / 4 | 0.027 |
| 2 s | withdraw | moves toward sender position | -0.211 | -1.38 | 20 / 5 | 0.027 |
| 2 s | turn toward | speeds up | +0.105 | +0.51 | 32 / 7 | 0.027 |

All eight retained the same sign in the camera-only check and were stable to leaving out any one channel. No single channel contributed more than 43% of a surviving cell.

These effects are **exploratory associations**, not demonstrated communication. Effect directions were not preregistered, cells contain only 14–75 events from 4–9 channels, and an outside cause can still reach the two dogs in sequence.

Detailed report: [pilot_communication_events_2026-10-31.md](pilot_communication_events_2026-10-31.md)

## 4. Event indicators do not generalize as global predictive features

The six event kinds were also added to the sender regression as sparse indicators.

| Window set | Event-indicator increment R² | Shuffle null | p |
|---|---:|---:|---:|
| Clean | -0.011 ± 0.007 | -0.010 | 0.71 |
| Relaxed | -0.004 ± 0.003 | -0.004 | 0.62 |
| Event windows only | -0.026 ± 0.036 | -0.009 | 0.86 |

**Conclusion:** event indicators did not improve prediction on held-out channels. This does not contradict the matched analysis: only about 6% of windows contain an event, and the exploratory effects are specific to event kind, lag, and response target.

## 5. Falsification / receiver-orientation test

A follow-up asked whether the eight event effects were stronger when the receiver's **pose body axis was oriented toward the sender**.

| Split | Toward sender | Comparison group | Difference | p |
|---|---:|---:|---:|---:|
| facing vs. not-facing | +0.70 | +0.69 | +0.005 | 0.56 |
| facing vs. sender behind | +0.70 | +0.73 | -0.03 | 0.61 |

This gives **no support for the simplest visual-signal account** and is compatible with shared external causes remaining plausible.

Important limitation: the permutation test could reliably detect only large facing effects (roughly 0.8–0.9 SD). Smaller visual dependence is not ruled out. Dogs may also hear movement, have broad visual fields, and body-heading estimates are imperfect.

## 6. Audio feasibility

The audio pipeline processed the 59 pilot soundtracks with AST (AudioSet) using 1 s windows every 0.5 s, producing **32,875 windows** plus 768-d embeddings.

However:

- only **0.5%** of all windows had dog-vocal probability > 0.5;
- music occupied at least half the windows in **30/59 videos**;
- only **1 video** had a sufficiently clean "own sound" soundtrack under the pilot thresholds;
- in the clean broad-prediction subset, dog-vocal > 0.5 occurred in the sender's 1 s history in only **18 windows / 10 moments / 4 videos / 4 channels**;
- one microphone records every dog in the scene;
- the tested nose-chin mouth-opening cue did not reliably identify the vocalizing dog.

**Conclusion:** no sender-specific audio experiment was run. Detecting a dog sound is not the same as identifying its sender.

Detailed report: [pilot_audio_2026-10-31.md](pilot_audio_2026-10-31.md)

## What KLP v0 can and cannot claim

### Supported by this pilot

- dog interactions can be represented as synchronized behavioral time series;
- broad visual sender features do not improve held-out-channel prediction in the current clean sample;
- this null broad result persists across the tested models, targets, and short lags;
- some sender behavioral events are followed by structured receiver changes under matched observational analysis;
- those localized event associations do not yet generalize as a global predictive gain;
- the receiver-orientation check does not support the simplest visual-signal explanation;
- current web-video audio is source-attribution limited.

### Not established

- that Dog A caused Dog B's response;
- that the event associations are communicative signals;
- semantic meaning;
- intention;
- "dog language" or translation.

## Why the negative and falsification results matter

The pilot is designed to make positive claims difficult. Camera stabilization, held-out sources, sender shuffling, lag tests, matched controls, robustness checks, and falsification tests are used to distinguish interesting structure from artifacts and alternative explanations.

The current outcome narrows the scientific problem: the next bottleneck is not simply model capacity. It is **identifiability and better controlled data**.

The next study should treat the eight event-response directions as hypotheses to confirm on independent footage while directly measuring shared causes and improving source attribution.
