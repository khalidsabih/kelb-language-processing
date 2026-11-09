# Pilot experiments: pose vs motion-only tracks (2026-10-30)

Question: does sender-dog information improve prediction of the receiver dog's next 1 s beyond the receiver's own history?

## Setup

- Data: 59 YouTube videos from the Pawgaze candidate list (15 in pilot batch 1, 44 in batch 2), tracked with YOLO11s + ByteTrack at 10 fps.
- Pose tracks: re-extracted with the Dog-Pose-trained YOLO11s-pose model (60 epochs, `best.pt`, pose mAP50 0.83 on Dog-Pose val), run on a crop around each tracked dog, keypoint cutoff 0.3. 74% of headings come from the pose body axis, 26% from motion direction; 80% of windows have complete sender pose features.
- Config: `configs/v0.yaml` unchanged (past 1 s, future 1 s, stride 0.5 s, max 200 windows per video, min modality c
overage 0.5, channel split, test fraction 0.25, 10 splits, ridge alpha 1.0, interaction ridge).
- Sign conventions: `sender_increment_r2` = R²(sender + baseline) − R²(baseline). Ablation `delta_r2_vs_full` = R²(full) − R²(without that modality); negative means the model does better without the modality.

## Summary

| Run | Windows (after cap) | Baseline R² | Sender increment R² |
|---|---|---|---|
| Pose tracks, interaction ridge (default) | 3,996 from 51 videos | 0.051 ± 0.031 | −0.046 ± 0.032 |
| Motion-only tracks, interaction ridge (default) | 4,536 from 51 videos | 0.082 ± 0.035 | −0.030 ± 0.036 |
| Pose tracks, linear ridge alpha 1 | 3,996 from 51 videos | 0.054 | +0.008 ± 0.011 |

The interaction ridge (all pairwise products) overfits at ~2,700 training windows, so every added modality looks harmful. With a linear ridge, the sender increment is small and not reliably above zero; it comes from motion (+0.009), while pose (−0.001) and orientation (0.000) add nothing. Baselines differ between pose and motion-only runs because the pose run keeps only windows with complete pose features; within each run, baseline and sender models use the same windows.

## Full output: pose tracks

```
$ python3.13 scripts/run_experiments.py --tracks data/external/pilot/pose/tracks.parquet data/external/pilot/batch2/pose/tracks.parquet
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
baseline r2:             0.0508 ± 0.0314
sender_plus_baseline r2: 0.0052 ± 0.0371
sender_increment_r2:     -0.0456 ± 0.0320
sender_increment_mae:    -0.0081 ± 0.0034
  increment r2 delta_distance: -0.0887 ± 0.0755
  increment r2 delta_orientation: -0.0223 ± 0.0158
  increment r2 delta_velocity: -0.0256 ± 0.0457

=== Leave-one-modality-out ablation ===
dropped_modality  delta_r2_vs_full_mean  delta_r2_vs_full_std  delta_mae_vs_full_mean  delta_mae_vs_full_std
     orientation              -0.009406              0.015793               -0.001834               0.003082
          motion              -0.020085              0.027802               -0.002245               0.003672
            pose              -0.021478              0.014453               -0.006083               0.003115
```

## Full output: motion-only tracks

```
$ python3.13 scripts/run_experiments.py --tracks data/external/pilot/tracks.parquet data/external/pilot/batch2/tracks.parquet
Tracks: 59 videos, 7236 dogs, 187838 rows
Windows: 9602
Sender modalities used: ['motion', 'orientation']
Windows with complete features: 8944 of 9602
After cap of 200 per video: 4536 windows from 51 videos, largest video share 4%

10 identity-disjoint splits by channel (mean ± std)
Rows: train=3128.2000 ± 620.4468, test=1407.8000 ± 620.4468
Dropped (cross-partition) rows: 0.0000 ± 0.0000 of total
Test dogs per split: 246.0000 ± 105.9098

=== Sender incremental-value experiment ===
baseline r2:             0.0823 ± 0.0351
sender_plus_baseline r2: 0.0518 ± 0.0629
sender_increment_r2:     -0.0304 ± 0.0363
sender_increment_mae:    -0.0043 ± 0.0032
  increment r2 delta_distance: -0.0380 ± 0.0770
  increment r2 delta_orientation: -0.0114 ± 0.0122
  increment r2 delta_velocity: -0.0419 ± 0.0594

=== Leave-one-modality-out ablation ===
dropped_modality  delta_r2_vs_full_mean  delta_r2_vs_full_std  delta_mae_vs_full_mean  delta_mae_vs_full_std
     orientation              -0.003341              0.014734               -0.002021               0.002580
          motion              -0.029098              0.032863               -0.003407               0.002368
```

## Model check: linear vs interaction ridge (pose tracks, same windows and splits)

Same 3,996 windows, modalities pose + motion + orientation, 10 channel splits. Format: baseline R², sender increment R² (mean ± std), then leave-one-modality-out `delta_r2_vs_full` (mean ± std).

```
linear ridge a=1: baseline 0.054, increment +0.008 ± 0.011; pose -0.001±0.006, motion +0.009±0.009, orientation +0.000±0.001
linear ridge a=100: baseline 0.056, increment +0.008 ± 0.010; pose -0.001±0.005, motion +0.009±0.008, orientation +0.000±0.001
interaction ridge a=100: baseline 0.054, increment -0.037 ± 0.030; pose -0.017±0.013, motion -0.015±0.027, orientation -0.007±0.014
```

## Caveats

- Web footage: fragmented tracks (median 0.7–4 s), much human-led footage, channel concentration, no camera-motion compensation.
- Pose heading was checked only on side-on dogs (61 hand-labelled by one labeller, ~90% correct); it is ill-defined for dogs facing toward or away from the camera. Head direction is unreliable (57–70% correct) and is not used.
- Pose sender features are crude (head height, compactness). No audio.
