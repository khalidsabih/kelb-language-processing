# KLP v0 — Pilot Results

This folder contains the exploratory pilot analyses for KLP v0.

> **Interpretation boundary:** these results describe predictive and event-response structure in observational dog-interaction footage. They do **not** establish causation, communication, semantic meaning, or intention.

## Results at a glance

| Question | Result | Interpretation |
|---|---|---|
| Does Dog A improve broad prediction of Dog B? | **No reliable sender advantage** in the clean linear-ridge pilot | The current visual sender features did not improve held-out-channel prediction beyond Dog B's own history and dyadic/context features. |
| Are some sender events followed by structured changes in Dog B? | **Yes, exploratory** | 8 of 54 event-response tests survived Holm correction, but the directions were discovered post hoc and cells are small. |
| Do the event effects support a simple visual-signal account? | **No clear support** | Receivers facing the sender responded about as strongly as receivers not facing the sender. |
| Can the pilot support sender-specific audio analysis? | **No** | The web-video soundtracks are dominated by shared microphones, music, narration, and sparse dog vocalizations. |
| Main next step | **Better controlled multimodal data** | Synchronized video, attributable audio, known identities, and explicit measurement of shared causes. |

## 1. Broad sender prediction

Clean cycle-2 analysis:

- **2,322 windows**
- **32 videos**
- **8 YouTube channels**
- **10 held-out-channel splits**
- 1 s past -> 1 s future, 0.5 s stride
- stabilized tracks and close/continuous pair filtering

Linear ridge:

| Metric | Value |
|---|---:|
| Receiver-only baseline R² | 0.052 ± 0.033 |
| Sender + receiver R² | 0.044 ± 0.034 |
| Sender increment R² | **-0.008 ± 0.007** |
| Shuffled-sender null mean | +0.002 |
| p vs. shuffled sender | 1.00 |

**Conclusion:** no measurable global sender advantage in the current clean pilot.

Detailed report: [pilot_experiments_2026-10-30_cycle2.md](pilot_experiments_2026-10-30_cycle2.md)

## 2. Event-centered analysis

The event pipeline detected **1,180 sender events** across six observable event types:

- move onset
- stop
- approach
- withdraw
- turn toward
- turn away

Receiver responses were measured at 0, 1, and 2 s lags and compared with matched control moments from the same video.

Eight of 54 tests survived Holm correction. Examples include:

- sender stop -> receiver slows
- sender approach -> receiver slows
- sender withdraw -> receiver speeds up
- sender turn toward -> later receiver distance/speed changes

These effects are **exploratory associations**, not demonstrated communication.

Detailed report: [pilot_communication_events_2026-10-31.md](pilot_communication_events_2026-10-31.md)

## 3. Falsification / facing test

A follow-up asked whether the event responses were stronger when the receiver was facing the sender.

- facing mean standardized response: **+0.70**
- not-facing mean standardized response: **+0.69**
- difference: **+0.005**
- p = **0.56**

The stronger 'sender behind' comparison was also null.

**Interpretation:** this gives no support for the simplest visual-signal account and is consistent with shared external causes remaining plausible.

See the facing-test section in [pilot_communication_events_2026-10-31.md](pilot_communication_events_2026-10-31.md).

## 4. Audio feasibility

The pilot audio pipeline used AST (AudioSet) on 1 s windows every 0.5 s and retained dog-vocal/music/speech scores, loudness, and 768-d embeddings.

However:

- only **0.5%** of all audio windows had dog-vocal probability > 0.5;
- many videos were dominated by music or narration;
- one shared microphone records every dog in the scene;
- the tested mouth-opening cue did not reliably identify the vocalizing dog.

**Conclusion:** the current web-video audio cannot support sender-specific audio inference.

Detailed report: [pilot_audio_2026-10-31.md](pilot_audio_2026-10-31.md)

## What KLP v0 can and cannot claim

### Supported by this pilot

- dog interactions can be represented as synchronized behavioral time series;
- broad sender features do not currently improve held-out-channel prediction;
- some sender behavioral events are followed by structured receiver changes under matched observational analysis;
- current audio is attribution-limited.

### Not established

- that Dog A caused Dog B's response;
- that the event associations are communicative signals;
- semantic meaning;
- intention;
- "dog language" or translation.

## Why the negative results matter

The pilot is designed to make positive claims difficult. Camera stabilization, held-out sources, sender shuffling, temporal shifts, matched controls, and falsification tests are used to separate interesting structure from obvious artifacts.

The current outcome is therefore useful even without a positive global prediction result: it narrows the hypothesis, identifies event-centered structure worth confirming, and clarifies what the next dataset must measure.