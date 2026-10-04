# Smoking verifier: before / after on Mendeley Testing

- date: 2026-10-04 17:50
- detector: models/smoking/best.pt sha256 fded2de373f09f73
- classifier: models/smoking_cls/smoking_cls_v1_mac.pt sha256 4f3f2b4c4997f656
- threshold: 0.10 (chosen on Validation)
- images: 224 (112 smoking); Testing was not used for training or selection

| | detected (TP/positives) | false alarms (FP/negatives) |
|---|---|---|
| before: detector alone | 105/112 (94%) | 41/112 (37%) |
| after: detector + verifier | 99/112 (88%) | 9/112 (8%) |

Decision: **DO NOT ADOPT** (needs fewer false alarms and recall at most 5% lower)

Mendeley is one source of whole-person photos; QA clips are reported separately (tools/eval_clips.py) and decide whether this works on the camera.
