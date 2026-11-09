"""Pilot: dog vocal activity and acoustic embeddings from video soundtracks.

    .venv-pilot/bin/python scripts/pilot/extract_audio.py --audio data/external/pilot/audio

Audio comes from the video's YouTube audio stream (downloaded separately, since the
pilot videos are video-only). Each 1 s window, every 0.5 s, is scored with AST
fine-tuned on AudioSet (MIT/ast-finetuned-audioset-10-10-0.4593):

- ``dog_vocal``: highest probability among dog vocal classes (bark, yip, howl,
  bow-wow, growling, whimper (dog)); ``bark`` separately.
- ``music``, ``speech``: probabilities of those classes, to flag soundtracks that are
  edited or narrated rather than the scene's own sound.
- ``rms``: loudness.

Writes ``audio_frames.parquet`` (one row per video and window centre ``time_s``) and
``embeddings/<video_id>.npy`` (AST pooled embedding per window, float16, rows in the
same order). A single microphone cannot tell which dog made a sound, so these are
scene-level measurements, not attributed to a dog.
"""

import argparse
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
RATE = 16000
DOG_VOCAL = ["Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)"]


def load_audio(path: Path) -> np.ndarray:
    container = av.open(str(path))
    resampler = av.AudioResampler(format="flt", layout="mono", rate=RATE)
    chunks = [f.to_ndarray() for frame in container.decode(container.streams.audio[0]) for f in resampler.resample(frame)]
    container.close()
    return np.concatenate(chunks, axis=1).ravel()


def score(x, extractor, model, device, win_s=1.0, hop_s=0.5, batch=32):
    win, hop = round(win_s * RATE), round(hop_s * RATE)
    starts = np.arange(0, max(1, len(x) - win + 1), hop)
    labels = {label: i for i, label in model.config.id2label.items()}
    probs, embeddings = [], []
    for k in range(0, len(starts), batch):
        clips = [x[s:s + win] for s in starts[k:k + batch]]
        inputs = extractor(clips, sampling_rate=RATE, return_tensors="pt", max_length=128)
        with torch.no_grad():
            out = model(inputs.input_values.to(device), output_hidden_states=True)
        probs.append(torch.sigmoid(out.logits).cpu().numpy())
        hidden = out.hidden_states[-1]
        # Mean of the CLS and distillation tokens, as AST pools (here before its final layer norm).
        embeddings.append(((hidden[:, 0] + hidden[:, 1]) / 2).cpu().numpy().astype(np.float16))
    probs = np.concatenate(probs)
    frames = pd.DataFrame({
        "time_s": (starts + win / 2) / RATE,
        "dog_vocal": probs[:, [labels[c] for c in DOG_VOCAL]].max(axis=1),
        "bark": probs[:, labels["Bark"]],
        "music": probs[:, labels["Music"]],
        "speech": probs[:, labels["Speech"]],
        "rms": [float(np.sqrt(np.mean(x[s:s + win] ** 2))) for s in starts],
    })
    return frames, np.concatenate(embeddings)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio", type=Path, required=True, help="Folder of <video_id>.m4a/.webm audio files")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    extractor = ASTFeatureExtractor.from_pretrained(MODEL)
    model = ASTForAudioClassification.from_pretrained(MODEL).to(args.device).eval()
    (args.audio / "embeddings").mkdir(exist_ok=True)
    files = sorted(p for p in args.audio.iterdir() if p.suffix in {".m4a", ".webm", ".opus"})
    tables = []
    for i, path in enumerate(files, 1):
        frames, emb = score(load_audio(path), extractor, model, args.device)
        np.save(args.audio / "embeddings" / f"{path.stem}.npy", emb)
        tables.append(frames.assign(video_id=path.stem))
        print(f"[{i}/{len(files)}] {path.stem}: {len(frames)} windows, dog vocal > 0.5 in {(frames.dog_vocal > 0.5).mean():.1%}, "
              f"music > 0.5 in {(frames.music > 0.5).mean():.0%}, speech > 0.5 in {(frames.speech > 0.5).mean():.0%}", flush=True)
    out = pd.concat(tables, ignore_index=True)
    out.to_parquet(args.audio / "audio_frames.parquet", index=False)
    print(f"Wrote {len(out)} windows for {len(files)} videos")


if __name__ == "__main__":
    main()
