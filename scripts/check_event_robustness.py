"""Robustness of the Holm-significant cycle-5 event responses (cells hard-coded from the 2026-10-31 run).

    python3.13 scripts/check_event_robustness.py <tracks_stabilized files...>

Per cell: leave-one-channel-out range, number of channels agreeing in sign, largest
channel share, and the estimate after dropping frames where camera stabilization failed.
"""
import sys
import numpy as np, pandas as pd, yaml
from klp.data.events import detect_events
from klp.data.schema import TARGET_COLUMNS
from klp.data.tracks import build_windows, filter_interacting, load_tracks, prepare_tracks
from klp.experiments.events import Matcher, event_response, flag_event_windows
from klp.features.select import baseline_features

cfg = yaml.safe_load(open("configs/v0.yaml")); ev = cfg["communication_events"]
raw = pd.concat([pd.read_parquet(p) for p in sys.argv[1:]], ignore_index=True)
CELLS = [(0.0, "move_onset", "delta_distance"), (1.0, "turn_to", "delta_distance"), (2.0, "approach", "delta_distance"),
         (2.0, "withdraw", "delta_distance"), (0.0, "stop", "delta_velocity"), (1.0, "approach", "delta_velocity"),
         (1.0, "withdraw", "delta_velocity"), (2.0, "turn_to", "delta_velocity")]

def run(tracks, label, loo):
    tracks = prepare_tracks(tracks)
    events = detect_events(tracks, before_s=ev["before_s"], after_s=ev["after_s"], refractory_s=ev["refractory_s"])
    out = []
    for lag in sorted({c[0] for c in CELLS}):
        w = build_windows(tracks, cfg["past_s"], cfg["future_s"], stride_s=0.0, lag_s=lag)
        w = filter_interacting(w.dropna(subset=baseline_features() + TARGET_COLUMNS), ev["min_run_s"], ev["max_pair_distance"])
        for l, kind, target in CELLS:
            if l != lag: continue
            r = event_response(w, events, kind, ev["n_controls"], ev["control_gap_s"], ev["simultaneous_s"], n_placebo=2000, n_boot=200)
            row = {"set": label, "lag_s": lag, "kind": kind, "target": target, "events": r["events"],
                   "estimate": r.get(target), "p": r.get(f"{target}_p")}
            if loo and r["events"]:
                f = flag_event_windows(w, events, kind, ev["control_gap_s"], ev["simultaneous_s"]).reset_index(drop=True)
                d = Matcher(f, baseline_features(), ev["n_controls"]).differences(np.flatnonzero(f.is_event.to_numpy()))
                ch = d.groupby("channel_id")[target].mean()
                loo_est = [d.loc[d.channel_id != c, target].mean() for c in ch.index]
                row.update(channels=len(ch), same_sign=int((np.sign(ch) == np.sign(row["estimate"])).sum()),
                           loo_min=min(loo_est), loo_max=max(loo_est),
                           top_channel_share=float(d.channel_id.value_counts(normalize=True).iloc[0]))
            out.append(row)
    return out

rows = run(raw, "all frames", loo=True)
rows += run(raw[raw.camera_ok.fillna(False).astype(bool)], "camera_ok only", loo=False)
pd.set_option("display.width", 220)
print(pd.DataFrame(rows).round(3).to_string(index=False))
