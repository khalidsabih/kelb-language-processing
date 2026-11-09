# Pilot audio (cycle 3), 2026-10-31

**Conclusion:** the pilot YouTube soundtracks cannot support a sender-audio analysis. Dog sounds are rare, most soundtracks are music or narration, and the tested way of attributing a sound to one dog (mouth opening) does not respond to barks. No audio experiment was run.

## What was done

- The 59 pilot videos were downloaded video-only, so their audio streams were fetched separately with yt-dlp (`bestaudio[ext=m4a]`, `data/external/pilot/audio/`).
- `scripts/pilot/extract_audio.py` scored 1 s windows every 0.5 s with AST fine-tuned on AudioSet (`MIT/ast-finetuned-audioset-10-10-0.4593`). That gave 32,875 windows with these outputs:
  - dog vocal: the maximum over bark, yip, howl, bow-wow, growling and whimper (dog);
  - music, speech and loudness;
  - a 768-d embedding per window.

## Soundtrack content

| | all 59 videos | 32 videos in the filtered (clean) subset |
|---|---|---|
| music in at least half of the windows | 30 | 12 |
| own sound (music in under 20% and speech in under 30% of windows) | 1 | 1 (75 of 4,023 clean windows) |
| any window with dog vocal > 0.5 | 18 | 12 |
| dog vocal > 0.5 in at least 2% of windows | 7 | 5 |

- Across all windows, only 0.5% have dog vocal > 0.5, and almost none have bark > 0.5.
- The rest is mostly speech: narration, or owners talking.
- In the clean subset, a dog sound occurs in the sender's 1 s history in only a few windows:
  - dog vocal > 0.5: 18 windows (10 distinct moments) in 4 videos from 4 channels.
  - dog vocal > 0.2: 77 windows (47 distinct moments) in 9 videos from 5 channels.

## Attribution check: does the mouth open during dog sounds?

One microphone records every dog in the scene at once. A sound only counts as sender information if it can be attributed to the sender rather than to the receiver. The candidate cue was the nose–chin keypoint gap, standardized within each tracked dog (z-score):

| dog vocal probability | dog frames | videos | gap median (body sizes) | mean gap z |
|---|---|---|---|---|
| < 0.05 | 139,310 | 59 | 0.113 | +0.00 |
| 0.05–0.2 | 4,911 | 38 | 0.110 | +0.02 |
| 0.2–0.5 | 4,479 | 30 | 0.115 | −0.03 |
| > 0.5 | 635 | 18 | 0.130 | −0.08 |

- During detected dog sounds the gap does not widen within dogs: the mean z is −0.08.
- Per video, the mean z ranges from −0.58 to +1.20, with no consistent sign.
- In 139 multi-dog frames during dog sounds, the dog with the widest gap leads the next by more than 1 SD in only 35% of frames.
- Taken together, the cue cannot pick out the vocalizing dog. Likely reasons:
  - the chin keypoint is noisy (the pose model was trained on Stanford Dog-Pose stills);
  - barks are short compared with the 1 s audio window;
  - often the dog making the sound is off screen.

## Implication

- Scene-level audio as baseline context is also not worth testing here: only 1 video has its own sound without music or narration.
- The audio code and embeddings stay available for better footage. This needs:
  - recordings with the original soundtrack, ideally own or partner recordings;
  - for attribution, either one microphone per dog (collar) or close-up video in which mouths are visible.
- This feeds into cycle 4 (better data).
