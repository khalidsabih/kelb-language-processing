# Pilot communication events (cycle 5), 2026-10-31

**Question:** after a sender dog abruptly changes what it is doing relative to a receiver, does the receiver respond differently than at comparable moments without such a change?

**Answer:** 8 of 54 tests survive Holm correction. All 8 are stable when any one channel is dropped and when frames without camera stabilization are removed. Together they form a consistent picture of following and yielding:
- The receiver speeds up and moves toward where the sender was after the sender withdraws.
- It slows down after the sender stops or approaches.
- It moves away after the sender turns to face it.

This is the project's first positive result. It is still exploratory:
- The effect directions were not fixed in advance.
- Each cell has 14–75 events from 4–9 channels.
- The design cannot rule out a common cause that reaches the two dogs in turn, for example a person calling them.

These cells need confirmation on new footage.

**Follow-up (see below):** receivers with the sender behind them respond as strongly as receivers facing it. A shared outside cue predicts exactly that, so this check gives no support for reading these effects as communication.

## Method

`scripts/run_communication_events.py`, with settings under `communication_events` in `configs/v0.yaml`.

- **Events** (`klp.data.events`): for each ordered pair of dogs tracked together, a change in one sender signal between the mean over the 0.5 s before and the mean over the next 0.3 s (speeds in body sizes/s). Events of the same kind and pair closer than 2 s are merged.

  | kind | signal | before | after |
  |---|---|---|---|
  | move_onset | sender speed | ≤ 0.3 | ≥ 1.0 |
  | stop | sender speed | ≥ 1.0 | ≤ 0.3 |
  | approach | sender velocity toward receiver | ≤ 0.3 | ≥ 1.0 |
  | withdraw | sender velocity toward receiver | ≥ −0.3 | ≤ −1.0 |
  | turn_to | cosine, sender's pose body heading vs direction to receiver | ≤ 0.5 | ≥ 0.87 |
  | turn_away | same cosine | ≥ 0.87 | ≤ 0.5 |

- **Exclusions:**
  - Turn events use only headings from pose; a heading taken from the motion direction would turn whenever the dog walks.
  - Events where the receiver had the same kind of event in the preceding 0.8 s are dropped, since a simultaneous change points to a shared cause.
- **Pairs:** at most 5 body sizes apart, with no minimum tracked run. This is looser than the regression filter (3 body sizes, 5 s run), because events are rare. Tracks are camera-stabilized.
- **Response:** the receiver's targets in the 1 s window starting 0, 1 or 2 s after the event (lag):
  - `delta_distance`: movement away from the sender's position at the event, in body sizes;
  - `delta_orientation`: change in facing the sender, as a cosine;
  - `delta_velocity`: future minus past speed, in body sizes/s.
- **Matched controls:**
  - For each event, the 5 nearest moments in the same video by the receiver's own past speed, orientation and distance change, the pair distance and visibility.
  - Controls are at least 3 s from any event of that kind in the pair.
  - Differences are bias-corrected with a linear model of the targets on the matched features, fitted on all controls, because some cells had mean imbalance up to 0.64 SD (approach at 2 s).
- **Inference:**
  - Placebo p: the same matching repeated from random control moments of each event's pair, 2,000 times. The p-value is two-sided around the placebo mean.
  - Holm adjustment across all 6 kinds × 3 lags × 3 targets = 54 tests. With 2,000 placebos the smallest possible Holm p is 54/2001 = 0.027.
  - 95% intervals resample whole channels. On simulated null data with 6 channels they covered 87% rather than 95%, so the placebo p is the primary test.
  - On the same null data, 7.5% of placebo p-values fell below 0.05 (120 tests).

## Events detected

1,180 events in all co-tracked pairs of 51 videos:

| kind | events | videos | channels |
|---|---|---|---|
| move_onset | 279 | 41 | 10 |
| turn_to | 224 | 42 | 10 |
| turn_away | 212 | 31 | 10 |
| approach | 180 | 34 | 10 |
| stop | 149 | 37 | 9 |
| withdraw | 136 | 32 | 11 |

After the pair filter, the matching requirements and the exclusions, between 14 and 83 events per kind and lag remain. Longer lags need longer tracked runs, so fewer events remain at 2 s.

## Results that survive Holm correction

Unit is body sizes for distance and body sizes/s for velocity. d is the difference divided by the control SD.

| lag | sender event | receiver response | estimate | d | events / channels | Holm p | channels with same sign | leave-one-channel-out range | camera-ok frames only |
|---|---|---|---|---|---|---|---|---|---|
| 0 s | move_onset | moves toward sender's position | −0.074 | −0.36 | 75 / 9 | 0.027 | 8 / 9 | −0.088 to −0.062 | −0.109 |
| 0 s | stop | slows | −0.139 | −0.70 | 60 / 8 | 0.027 | 6 / 8 | −0.183 to −0.067 | −0.122 |
| 1 s | turn_to | moves away | +0.065 | +0.47 | 44 / 8 | 0.047 | 6 / 8 | +0.028 to +0.102 | +0.068 |
| 1 s | approach | slows | −0.117 | −0.59 | 23 / 7 | 0.027 | 7 / 7 | −0.151 to −0.096 | −0.109 |
| 1 s | withdraw | speeds up | +0.088 | +0.43 | 25 / 5 | 0.027 | 5 / 5 | +0.078 to +0.111 | +0.106 |
| 2 s | approach | moves toward sender's position | −0.167 | −1.16 | 14 / 4 | 0.027 | 4 / 4 | −0.247 to −0.032 | −0.161 |
| 2 s | withdraw | moves toward sender's position | −0.211 | −1.38 | 20 / 5 | 0.027 | 4 / 5 | −0.294 to −0.118 | −0.250 |
| 2 s | turn_to | speeds up | +0.105 | +0.51 | 32 / 7 | 0.027 | 6 / 7 | +0.078 to +0.118 | +0.115 |

- In the camera-ok run, all estimates have the same sign and raw placebo p ≤ 0.001.
- The leave-one-channel-out and camera checks come from `scripts/check_event_robustness.py`.
- No single channel holds more than 43% of a cell's events.
- The weakest cell is approach at 2 s: only 4 channels, the worst pre-correction balance (0.64 SD), and the estimate falls to −0.032 without one channel.

Near misses (Holm p 0.09) are withdraw → receiver turns away at 0 s, approach → receiver turns away at 2 s, and approach → receiver speeds up at 0 s. Orientation responses are noisier, partly because body heading is wrong in about 10% of frames. The full table is in `data/external/pilot/communication_events.csv` and `.log`. They are kept local because they list video IDs.

## Interpretation

- **Withdraw:** when the sender moves away, the receiver speeds up after 1 s and moves toward where the sender was after 2 s. This looks like following.
- **Move onset:** when a still sender starts moving, the receiver moves toward the sender's starting position within the next second. This fits joining or following.
- **Approach:** when the sender approaches, the receiver slows after 1 s. At 2 s it moves toward the sender's position at the event, which by then lies between the two dogs. This reads as holding ground or meeting rather than retreating.
- **Turn toward:** when the sender turns to face the receiver, the receiver moves away after 1 s and speeds up after 2 s. This fits yielding to, or being chased off by, a dog that orients toward it.
- **Stop:** when the sender stops, the receiver slows in the same second.

## Caveats

- **Common cause.** Matching on the receiver's state removes differences in what the receiver was already doing. It does not remove a stimulus that reaches the sender first and the receiver later, such as a person, a ball or a sound. Dropping events with a simultaneous receiver change (0.8 s) only removes the fastest shared reactions. This applies most to stop and move_onset at lag 0.
- **Active moments.** Events occur in active stretches of interaction. Controls come from the same video and match the receiver's state, but they may still be calmer moments overall.
- **Small and exploratory.** No direction was predicted before the run. There are 14–75 events per cell and 4–9 channels, and 54 tests, of which 8 pass. The intervals are too narrow with this few channels, as noted above.
- **Tracking errors.** An identity switch or a jump in detection creates a false sender event and false receiver motion at the same moment. Stabilization failures were checked and are not the explanation. Identity switches were not checked separately.
- **Relation to earlier results.** The regression over all windows (cycles 1–2) found no sender increment. That is consistent with these results: events are about 1% of windows, so an effect of d ≈ 0.5 confined to them does not move R² over all windows. The cycle-2 onset analysis found nothing because it had only 28 onsets within the stricter pair filter.

## Follow-up: does the receiver need to see the sender?

`scripts/check_event_facing.py` asks whether the responses depend on whether the receiver could see the sender. If they come from a signal the receiver sees, they should be stronger when it faces the sender. A shared outside cue (a person, a noise) predicts no difference.

**Method:**
- For the 8 significant cells, each event's matched-control difference is split by the receiver's facing at the event frame. Facing is the cosine between its pose body heading and the direction to the sender.
- Events without a pose heading are left out; that is 79 of 293 cell rows.
- Responses are scaled by the control SD and signed so that + is the cell's overall direction.
- **Prediction, fixed before the run:** facing receivers respond more. Tested one-sided, pooled over the cells, by permuting facing labels across distinct events of the same kind (10,000 permutations).

| split | facing events | not-facing events | mean response, facing | mean response, not facing | difference | p |
|---|---|---|---|---|---|---|
| facing: cos > 0.5 (within 60°); not facing: cos < 0 (beyond 90°) | 86 | 61 | +0.70 | +0.69 | +0.005 | 0.56 |
| not facing: cos < −0.5 (sender behind, beyond 120°) | 86 | 47 | +0.70 | +0.73 | −0.03 | 0.61 |

- Per cell, the pattern is mixed: not-facing receivers respond more in 4 of the 8 cells (first split), but per-cell groups have only 3–25 events.
- **Power:** the permutation SD of the difference is 0.33 (0.36 with the behind cut). The test therefore detects only differences of about 0.8–0.9 SD with 80% power, about the size of the effects themselves. This rules out the clearest signal account, where only receivers that see the sender respond. That account predicts a difference near the full effect (about 0.7), and observing +0.005 if the true difference were 0.7 has probability about 0.02. A partial dependence on sight is not ruled out.
- **Interpretation:** receivers with the sender behind them respond as much as those facing it. This is what a shared outside cue predicts, and it weakens the reading of these effects as signals. Other explanations remain:
  - dogs hear each other's movement;
  - with a visual field of about 250°, a dog sees much of what is behind its body axis;
  - body heading is wrong in roughly 10% of frames, which blurs the split.

  Even so, the follow-up gives no support for the communication reading.

## Follow-up: event indicators as regression features

`scripts/run_event_regression.py` adds a sender modality `events`.
- It holds one indicator per event kind: 1 if the sender had that event toward the receiver in the 1 s past window (`klp.data.events.add_event_features`).
- It is tested in the cycle-2 regression setup: linear ridge, 10 splits that hold out whole channels, and a shuffled-sender null with 20 draws.

| window set | windows (with an event) | videos / channels | baseline R² | events alone: increment | shuffle null | p | events on top of pose, motion, orientation |
|---|---|---|---|---|---|---|---|
| clean (regression filter, cap 200/video) | 2,322 (141) | 32 / 8 | 0.052 | −0.011 ± 0.007 | −0.010 | 0.71 | −0.010 ± 0.008 |
| relaxed (≤ 5 body sizes, cap 200/video) | 3,964 (263) | 51 / 11 | 0.080 | −0.004 ± 0.003 | −0.004 | 0.62 | −0.004 ± 0.004 |
| event windows only (relaxed) | 375 (375) | 34 / 10 | −0.019 | −0.026 ± 0.036 | −0.009 | 0.86 | −0.027 ± 0.036 |

- **Result:** the event indicators don't improve prediction on held-out channels in any set. They lower R² slightly and are no better than shuffled indicators. Within event windows, knowing the event kind doesn't help either.
- **This was expected and doesn't contradict the matched analysis:**
  - About 6% of windows contain an event. Even if every one of them had a d = 0.5 effect on one target, the gain in R² averaged over the three targets would be at most about 0.005.
  - The actual effects are specific to kind, lag and target, so the realistic gain is about 0.001–0.002.
  - That is below the split-to-split SD (0.003–0.008) and below the cost of fitting 6 sparse indicators across channels (−0.004 to −0.010).
- **What the two analyses measure:**
  - The matched-event analysis compares each event with moments in the same video.
  - The regression asks for a gain in prediction that carries over to unseen channels.
  - At pilot scale the regression can't detect effects of this size. Testing events through prediction needs many more events or a model that predicts only around events.

## Next steps

- **Confirm on new data:** treat the 8 cells as preregistered hypotheses, with their directions, on K9-Bench or new recordings, using the same code and config. Include the facing split, with enough events per group to detect a 0.3 SD difference. That needs roughly 7 times as many events.
- **Separate shared cues directly:** use footage where people and off-screen events can be annotated, or recordings without people present. Both depend on better data (cycle 4).
- **Event indicators in the regression:** done; no gain, and none was detectable at this scale (see above).
