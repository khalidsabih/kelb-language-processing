# Pilot experiments, cycle 2: corrected targets, nulls, interaction filter, camera stabilisation (2026-10-30)

Changes since `pilot_experiments_2026-10-30.md`:

1. **Distance target.** `delta_distance` is now the receiver's displacement along the line away from the sender's anchor position (sideways movement does not count), matching `receiver_distance_past`. Before, it was the change in straight-line distance, which scores a close receiver stepping past the sender as retreating.
2. **Sender nulls.** The real sender increment is compared with sender features shuffled within each video (`shuffle`) and circularly shifted within each pair by at least 5 s (`time_shift`), 20 draws each, on the same splits and rows.
3. **Interaction filter.** Only pairs tracked together without a gap for at least 5 s and at most 3 body sizes apart at the anchor time.
4. **Camera stabilisation.** `scripts/pilot/stabilize_tracks.py` estimates camera motion from background optical flow (dog boxes masked) as a similarity transform per frame, chained within each shot. Estimated in 97–98% of rows. Share of image motion common to all dogs in a frame (frames with 3+ dogs): 43% before, 33% after (75% to 25% in A49dvrnvyOs, a handheld pack walk).

## Diagnostics behind the changes (pose tracks, before stabilisation)

- `delta_distance`: median 0.00 body sizes, quartiles −0.06 / 0.07, top 1% of windows = 37% of its variance.
- No tracker jumps: per-frame steps above 1 body size: 0.00%.
- The largest targets cluster in time across many pairs (e.g. A49dvrnvyOs at t ≈ 169–171 s, all dogs drifting left during a camera turn).
- Before stabilisation and filtering, the linear-ridge sender increment (+0.0086) matched the shuffled-sender null (+0.0076, p = 0.43), so it was video-level, not moment-to-moment.

## Summary

Tracks: `data/external/pilot/{,batch2/}pose/tracks_stabilized.parquet`. After filtering and the 200-per-video cap: 2,322 windows from 32 videos, 8 channels; per split ~1,860 train / ~460 test windows, ~40 test dogs.

| Model | Baseline R² | Sender increment R² | Shuffle null | p (shuffle) |
|---|---|---|---|---|
| Linear ridge | 0.052 ± 0.033 | −0.008 ± 0.007 | +0.002 | 1.00 |
| Interaction ridge | 0.017 ± 0.030 | −0.042 ± 0.052 | −0.066 | 0.33 |

The time-shift null has only 336 eligible windows (pairs long enough for a 5 s shift); every model there has negative R², so it is not interpretable yet.

## Full output: linear ridge

```
$ python3.13 scripts/run_experiments.py --tracks data/external/pilot/pose/tracks_stabilized.parquet data/external/pilot/batch2/pose/tracks_stabilized.parquet --config <v0.yaml with model: linear>
Tracks: 59 videos, 7236 dogs, 187838 rows
Windows: 9602
Model: linear ridge, alpha 1.0
Sender modalities used: ['pose', 'motion', 'orientation']
Windows with complete features: 7120 of 9602
Continuous (>= 5.0 s) and close (<= 3.0 body sizes): 4023 windows from 32 videos, 8 channels
After cap of 200 per video: 2322 windows from 32 videos, largest video share 9%

10 identity-disjoint splits by channel (mean ± std)
Rows: train=1862.6000 ± 193.1310, test=459.4000 ± 193.1310
Dropped (cross-partition) rows: 0.0000 ± 0.0000 of total
Test dogs per split: 40.3000 ± 23.5705

=== Sender incremental-value experiment ===
baseline r2:             0.0516 ± 0.0333
sender_plus_baseline r2: 0.0438 ± 0.0340
sender_increment_r2:     -0.0078 ± 0.0072
sender_increment_mae:    -0.0021 ± 0.0008
  increment r2 delta_distance: -0.0121 ± 0.0128
  increment r2 delta_orientation: -0.0011 ± 0.0032
  increment r2 delta_velocity: -0.0103 ± 0.0231

=== Leave-one-modality-out ablation ===
dropped_modality  delta_r2_vs_full_mean  delta_r2_vs_full_std  delta_mae_vs_full_mean  delta_mae_vs_full_std
            pose              -0.001038              0.003146               -0.000758               0.000659
     orientation              -0.003656              0.004587               -0.000754               0.000745
          motion              -0.004135              0.003327               -0.000646               0.000994

=== Sender nulls (20 draws each; same splits, same rows as the real data) ===
      kind  n_rows  real_increment_r2  null_increment_r2_mean  null_increment_r2_p95  real_minus_null_r2  p_value
   shuffle    2322          -0.007819                0.002224               0.005315           -0.010044 1.000000
time_shift     336          -0.075409               -0.105569              -0.082993            0.030160 0.047619
```

## Full output: interaction ridge (v0.yaml)

```
$ python3.13 scripts/run_experiments.py --tracks data/external/pilot/pose/tracks_stabilized.parquet data/external/pilot/batch2/pose/tracks_stabilized.parquet
Tracks: 59 videos, 7236 dogs, 187838 rows
Windows: 9602
Model: interaction ridge, alpha 1.0
Sender modalities used: ['pose', 'motion', 'orientation']
Windows with complete features: 7120 of 9602
Continuous (>= 5.0 s) and close (<= 3.0 body sizes): 4023 windows from 32 videos, 8 channels
After cap of 200 per video: 2322 windows from 32 videos, largest video share 9%

10 identity-disjoint splits by channel (mean ± std)
Rows: train=1862.6000 ± 193.1310, test=459.4000 ± 193.1310
Dropped (cross-partition) rows: 0.0000 ± 0.0000 of total
Test dogs per split: 40.3000 ± 23.5705

=== Sender incremental-value experiment ===
baseline r2:             0.0173 ± 0.0295
sender_plus_baseline r2: -0.0248 ± 0.0349
sender_increment_r2:     -0.0421 ± 0.0524
sender_increment_mae:    -0.0083 ± 0.0084
  increment r2 delta_distance: -0.0100 ± 0.1273
  increment r2 delta_orientation: -0.0164 ± 0.0435
  increment r2 delta_velocity: -0.1000 ± 0.0640

=== Leave-one-modality-out ablation ===
dropped_modality  delta_r2_vs_full_mean  delta_r2_vs_full_std  delta_mae_vs_full_mean  delta_mae_vs_full_std
          motion               0.008334              0.028017               -0.001640               0.002321
     orientation              -0.015661              0.013018               -0.002765               0.002538
            pose              -0.028525              0.033770               -0.004584               0.008830

=== Sender nulls (20 draws each; same splits, same rows as the real data) ===
      kind  n_rows  real_increment_r2  null_increment_r2_mean  null_increment_r2_p95  real_minus_null_r2  p_value
   shuffle    2322          -0.042138               -0.066433              -0.019347            0.024295 0.333333
time_shift     336          -2.590241               -2.544622              -1.812569           -0.045618 0.523810
```

## Intermediate run: corrected target + nulls, unstabilised, unfiltered, interaction ridge

```
Tracks: 59 videos, 7236 dogs, 187838 rows
Windows: 9602
Sender modalities used: ['pose', 'motion', 'orientation']
Windows with complete features: 7120 of 9602
After cap of 200 per video: 3996 windows from 51 videos, largest video share 5%

10 identity-disjoint splits by channel (mean ± std)
Rows: train=2739.6000 ± 570.3395, test=1256.4000 ± 570.3395
Dropped (cross-partition) rows: 0.0000 ± 0.0000 of total
Test dogs per split: 239.5000 ± 100.5421

=== Sender incremental-value experiment ===
baseline r2:             0.0485 ± 0.0376
sender_plus_baseline r2: 0.0022 ± 0.0391
sender_increment_r2:     -0.0463 ± 0.0344
sender_increment_mae:    -0.0085 ± 0.0037
  increment r2 delta_distance: -0.0910 ± 0.0795
  increment r2 delta_orientation: -0.0223 ± 0.0158
  increment r2 delta_velocity: -0.0256 ± 0.0457

=== Leave-one-modality-out ablation ===
dropped_modality  delta_r2_vs_full_mean  delta_r2_vs_full_std  delta_mae_vs_full_mean  delta_mae_vs_full_std
     orientation              -0.009817              0.016153               -0.002006               0.003339
          motion              -0.018521              0.030652               -0.001909               0.003336
            pose              -0.021988              0.014985               -0.006485               0.003666

=== Sender nulls (20 draws each; same splits, same rows as the real data) ===
      kind  n_rows  real_increment_r2  null_increment_r2_mean  null_increment_r2_p95  real_minus_null_r2  p_value
   shuffle    3996          -0.046325               -0.031494              -0.015631            -0.01483 0.904762
time_shift     443          -1.137451               -0.869521              -0.734374            -0.26793 1.000000
```

Same windows, linear ridge (printed by an inline script):

```
linear ridge: baseline 0.0575 ± 0.0274, increment +0.0086 ± 0.0116
  delta_distance: -0.0010 ± 0.0085
  delta_orientation: -0.0002 ± 0.0037
  delta_velocity: +0.0271 ± 0.0299
dropped_modality  delta_r2_vs_full_mean  delta_r2_vs_full_std  delta_mae_vs_full_mean  delta_mae_vs_full_std
          motion                 0.0099                0.0079                  0.0008                 0.0014
     orientation                 0.0000                0.0012                 -0.0003                 0.0005
            pose                -0.0015                0.0059                 -0.0009                 0.0015
      kind  n_rows  real_increment_r2  null_increment_r2_mean  null_increment_r2_p95  real_minus_null_r2  p_value
   shuffle    3996             0.0086                  0.0076                 0.0105              0.0010   0.4286
time_shift     443            -0.0664                 -0.0602                -0.0477             -0.0062   0.8095
```
