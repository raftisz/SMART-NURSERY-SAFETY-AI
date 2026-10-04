# Smoking verifier: threshold selection on Mendeley Validation

- date: 2026-10-04 17:50
- detector: models/smoking/best.pt sha256 fded2de373f09f73
- classifier: models/smoking_cls/smoking_cls_v1_mac.pt sha256 4f3f2b4c4997f656
- images: 180 (90 smoking)
- rule: fewest false alarms with recall at most 5% below the detector alone

| threshold | detected (TP/positives) | false alarms (FP/negatives) |
|---|---|---|
| detector alone | 87/90 (97%) | 36/90 (40%) |
| + verifier >= 0.05 | 85/90 (94%) | 7/90 (8%) |
| + verifier >= 0.10 | 84/90 (93%) | 5/90 (6%) |
| + verifier >= 0.15 | 81/90 (90%) | 5/90 (6%) |
| + verifier >= 0.20 | 78/90 (87%) | 5/90 (6%) |
| + verifier >= 0.25 | 76/90 (84%) | 5/90 (6%) |
| + verifier >= 0.30 | 76/90 (84%) | 5/90 (6%) |
| + verifier >= 0.35 | 76/90 (84%) | 4/90 (4%) |
| + verifier >= 0.40 | 76/90 (84%) | 4/90 (4%) |
| + verifier >= 0.45 | 74/90 (82%) | 3/90 (3%) |
| + verifier >= 0.50 | 73/90 (81%) | 2/90 (2%) |
| + verifier >= 0.55 | 72/90 (80%) | 2/90 (2%) |
| + verifier >= 0.60 | 71/90 (79%) | 2/90 (2%) |
| + verifier >= 0.65 | 68/90 (76%) | 1/90 (1%) |
| + verifier >= 0.70 | 65/90 (72%) | 0/90 (0%) |
| + verifier >= 0.75 | 61/90 (68%) | 0/90 (0%) |
| + verifier >= 0.80 | 58/90 (64%) | 0/90 (0%) |
| + verifier >= 0.85 | 56/90 (62%) | 0/90 (0%) |
| + verifier >= 0.90 | 54/90 (60%) | 0/90 (0%) |
| + verifier >= 0.95 | 45/90 (50%) | 0/90 (0%) |

**Chosen threshold: 0.10** (then measure on Testing with `report`)
