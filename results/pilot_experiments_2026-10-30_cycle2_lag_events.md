# Pilot experiments, cycle 2: response-lag sweep and sender move onsets (2026-10-30)

Same pipeline as `pilot_experiments_2026-10-30_cycle2.md`: stabilised pose tracks, corrected `delta_distance`, pairs tracked ≥ 5 s and ≤ 3 body sizes apart, cap 200 windows per video, 10 channel splits, 20 shuffle-null draws.

- **Lag sweep:** the receiver's 1 s target window starts `lag_s` after the sender's 1 s past window ends (0, 0.5, 1, 2, 3 s).
- **Move onsets:** sender mean speed < 0.3 body sizes/s over 0.5 s, then > 1.0 over the next 0.3 s; 2 s refractory. Onsets where the receiver also started in the preceding 0.8 s are excluded. A receiver-only linear baseline is fitted on control windows of other channels; its residuals after onsets are compared with residuals at control times (≥ 3 s from any onset of that sender) for the same pairs; 95% interval from resampling channels (2,000 draws).

## Summary

- No lag shows a sender increment above the shuffled-sender null (linear ridge: increments −0.008 to −0.033, p = 1.0 at every lag). R² falls at 3 s, where fewer windows remain (1,477).
- 398 onsets were detected (6% within 0.5 s of another dog's onset), but only 32 fall in filtered pair windows, 28 after excluding shared starts (12 videos, 6 channels). Receiver response after onsets does not differ from control times; every interval includes 0 and is wide relative to the effect sizes one would expect.
- The split experiment on the 28 onset windows produced meaningless R² (test sets of a few windows); the script now skips it below 300 windows or 4 channels.

## Full output (printed before the skip guard was added)

```
##### LINEAR
Model: linear ridge, alpha 1.0; filter: run >= 5.0 s, distance <= 3.0 body sizes; cap 200 per video

=== Response-lag sweep (target window starts lag_s after the anchor) ===
 lag_s  windows  videos  baseline_r2  increment_r2  increment_r2_std  inc_delta_distance  inc_delta_orientation  inc_delta_velocity  shuffle_null_r2  p_shuffle
   0.0     2322      32       0.0516       -0.0078            0.0072             -0.0121                -0.0011             -0.0103           0.0022        1.0
   0.5     2183      32       0.0407       -0.0095            0.0116             -0.0091                -0.0031             -0.0164           0.0050        1.0
   1.0     2045      32       0.0453       -0.0096            0.0112             -0.0073                -0.0045             -0.0169           0.0025        1.0
   2.0     1777      32       0.0612       -0.0109            0.0152             -0.0083                -0.0070             -0.0173           0.0042        1.0
   3.0     1477      32       0.0171       -0.0334            0.0504             -0.0800                 0.0005             -0.0207          -0.0158        1.0

=== Sender move onsets (still < 0.3 then > 1.0 body sizes/s) ===
Onsets: 398 from 335 dogs in 50 videos
Filtered windows anchored at a sender onset: 32, of which the receiver also just started: 4

Receiver response after sender onsets vs control times of the same pairs (28 onset windows, 1145 controls, 12 videos, 6 channels; baseline cross-fitted by channel; units: body sizes, cosine, body sizes/s):
                   event_minus_control  ci95_low  ci95_high  control_resid_sd
delta_distance                 -0.0228   -0.0957     0.0228            0.1357
delta_orientation               0.0393   -0.0763     0.2710            0.7331
delta_velocity                 -0.0149   -0.0916     0.1568            0.2151

=== Sender increment on onset windows only (28 windows, 6 channels) ===
baseline r2 -3.8882 ± 8.4252; increment -24.5946 ± 63.1087; shuffle null -14.7494, p = 0.905
##### INTERACTION
Model: interaction ridge, alpha 1.0; filter: run >= 5.0 s, distance <= 3.0 body sizes; cap 200 per video

=== Response-lag sweep (target window starts lag_s after the anchor) ===
 lag_s  windows  videos  baseline_r2  increment_r2  increment_r2_std  inc_delta_distance  inc_delta_orientation  inc_delta_velocity  shuffle_null_r2  p_shuffle
   0.0     2322      32       0.0173       -0.0421            0.0524             -0.0100                -0.0164             -0.1000          -0.0664     0.3333
   0.5     2183      32       0.0051       -0.1014            0.1041             -0.1123                -0.0292             -0.1626          -0.0686     0.8571
   1.0     2045      32       0.0135       -0.0966            0.0585             -0.0835                -0.0622             -0.1442          -0.0647     0.8571
   2.0     1777      32       0.0165       -0.1425            0.0824             -0.1655                -0.0611             -0.2008          -0.0873     1.0000
   3.0     1477      32      -0.1488       -0.3039            0.4488             -0.5256                -0.1072             -0.2790          -0.0891     1.0000

=== Sender move onsets (still < 0.3 then > 1.0 body sizes/s) ===
Onsets: 398 from 335 dogs in 50 videos
Filtered windows anchored at a sender onset: 32, of which the receiver also just started: 4

Receiver response after sender onsets vs control times of the same pairs (28 onset windows, 1145 controls, 12 videos, 6 channels; baseline cross-fitted by channel; units: body sizes, cosine, body sizes/s):
                   event_minus_control  ci95_low  ci95_high  control_resid_sd
delta_distance                 -0.0228   -0.0957     0.0228            0.1357
delta_orientation               0.0393   -0.0763     0.2710            0.7331
delta_velocity                 -0.0149   -0.0916     0.1568            0.2151

=== Sender increment on onset windows only (28 windows, 6 channels) ===
baseline r2 -229.6606 ± 631.2311; increment +210.4986 ± 600.2426; shuffle null +141.9000, p = 0.286

[exited with code 0]
```
