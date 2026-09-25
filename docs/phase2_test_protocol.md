# Phase 2 Test Protocol

Purpose: measure whether **pretrained** YOLO11n detects `scissors` and `knife`
well enough to skip Phase 3 (dataset collection) and Phase 4 (fine-tuning).

Nothing in this document should be filled in by Claude. Every row of
`phase2_test_results.csv` comes from a run you perform yourself.

## Before you start

- **Real physical objects only.** Every trial must use an actual object in the
  room. Pointing the camera at a photo of scissors or a knife — on a phone, a
  laptop screen, or a printout — is **not a valid trial**. Screen images have
  wrong lighting, no depth, and screen texture, so they say nothing about real
  performance. Such a run is void and must not be entered in the CSV.
- Confidence threshold fixed at **0.25** for the whole matrix (the gate is
  evaluated on the *reported confidence*, not on the threshold). Only change it
  with `+`/`-` if you are doing an explicit threshold sweep, and note it.
- Use a tape measure or floor markers for the distance values. Estimated
  distances make the gate meaningless.
- One `s` press = one trial. It writes the frame plus one CSV row per detection.
- After each `s`, fill in the blank columns: `scenario`, `distance_m`,
  `lighting`, `background`, `notes`.
- If nothing is detected, still press `s`. The program writes a row with
  `detected(y/n)=n` — a miss is data.

- **Fix `--width`, `--height` and `--imgsz` once, and do not change them** for
  any part of the matrix, including the speed test. Input size trades accuracy
  against speed, so mixing sizes makes the accuracy and speed results describe
  two different systems. Write the values you chose into the outcome record.

Run it with:

```bash
source venv/bin/activate
python detect.py --source 0 --imgsz 640 --width 1280 --height 720
```

Use the same three values for every test below. If the camera refuses the
requested resolution, the program prints a `[WARN]` and records the size actually
used — carry that real value into the outcome record.

## Test matrix

### A. Scissors — 10 trials required at ≤ 2 m

| # | Object | Scenario | Distance | Lighting | Background |
|---|--------|----------|----------|----------|------------|
| A1 | scissors | flat on table, blades open | 0.5 m | bright | plain |
| A2 | scissors | flat on table, blades closed | 0.5 m | bright | plain |
| A3 | scissors | flat on table, blades open | 1.0 m | bright | plain |
| A4 | scissors | held in hand, blades visible | 1.0 m | bright | plain |
| A5 | scissors | held in hand, partly occluded by fingers | 1.0 m | bright | plain |
| A6 | scissors | flat on table | 2.0 m | bright | plain |
| A7 | scissors | held in hand | 2.0 m | bright | plain |
| A8 | scissors | flat on table | 1.0 m | dim | plain |
| A9 | scissors | flat on table | 1.0 m | bright | cluttered |
| A10 | scissors | angled 45° to camera | 1.0 m | bright | plain |

### B. Knife — 10 trials at ≤ 2 m, **only if a real knife is available**

> **If you cannot obtain a real knife, skip this section entirely.** Do not
> substitute a photograph on a phone or monitor. Record the result as
> `Knife: not tested — equipment limitation` in the outcome record at the end of
> this document, and leave section B out of the CSV. An untested criterion is
> neither a pass nor a fail — it is a documented gap. A blunt table knife or
> butter knife is a reasonable and safe substitute for a sharp one; a photo is
> not.

| # | Object | Scenario | Distance | Lighting | Background |
|---|--------|----------|----------|----------|------------|
| B1 | knife | flat on table, blade visible | 0.5 m | bright | plain |
| B2 | knife | flat on table, blade visible | 1.0 m | bright | plain |
| B3 | knife | held in hand, blade down | 1.0 m | bright | plain |
| B4 | knife | held in hand, blade up | 1.0 m | bright | plain |
| B5 | knife | held, blade partly occluded | 1.0 m | bright | plain |
| B6 | knife | flat on table | 2.0 m | bright | plain |
| B7 | knife | held in hand | 2.0 m | bright | plain |
| B8 | knife | flat on table | 1.0 m | dim | plain |
| B9 | knife | flat on table | 1.0 m | bright | cluttered |
| B10 | knife | angled 45° to camera | 1.0 m | bright | plain |

### C. Person — 4 trials (sanity check, not part of the gate)

| # | Object | Scenario | Distance | Lighting | Background |
|---|--------|----------|----------|----------|------------|
| C1 | person | standing, full body | 2.0 m | bright | plain |
| C2 | person | seated, upper body only | 1.0 m | bright | plain |
| C3 | person | partly out of frame | 1.5 m | bright | plain |
| C4 | person | crouching / low to floor | 1.5 m | bright | plain |

### D. False-positive control — 3 minutes, NO scissors or knife present

Point the camera at ordinary nursery-like objects and leave it running for a
**full 3 minutes**. There must be no scissors and no knife anywhere in frame.

Objects to include: pens, a TV remote, a phone, a ruler, plastic toys, a
spoon, crayons, a book.

Procedure:

1. Note the wall-clock start time.
2. Move the objects slowly through the frame, at 0.5 m, 1 m and 2 m.
3. Every time a `scissors` or `knife` box appears, press `s` immediately and
   write the real object in `notes` (e.g. "actually a ruler").
4. Note the wall-clock end time. Confirm it was ≥ 3 minutes.
5. Count the rows where `object` is `scissors` or `knife`. That count is the
   false-positive number for the gate.

A `person` detection during this test is **not** a false positive.

### E. Speed measurement

A speed number counts toward the gate only if **all four** conditions hold:

1. **Display ON.** Never use `--no-display` for a gate number — it skips window
   drawing and `waitKey` entirely, so the figure is optimistic. Those runs are
   diagnostic only.
2. **Same `--width`, `--height` and `--imgsz` as the scissors matrix (A).** A
   fast number at a smaller input size does not describe the configuration you
   measured accuracy on.
3. **Run continuously for ≥ 60 seconds** before reading.
4. Read the settled `loop FPS` from the overlay, not the opening seconds.

Procedure:

```bash
# use the SAME values here as in test A
python detect.py --source 0 --imgsz 640 --width 1280 --height 720
```

Leave it running, pointed at a normal scene, and do not touch the keys. Watch
the third overlay line:

- `FPS reading: warming up NN/60s` — not yet usable.
- `FPS reading: GATE-VALID` — from this point the `loop FPS` value may be quoted.

Once it reads `GATE-VALID`, note the `loop FPS`. On quitting, confirm the summary
prints `FPS gate status : VALID`. If it prints `NOT VALID`, the run does not
count — repeat it.

Record the FPS value, the `imgsz`, and the actual capture resolution shown in the
exit summary. If the camera refused the requested resolution the program prints a
`[WARN]` and records what was really used — report that, not what you asked for.

## Filling in the results

For each trial, the program fills: `timestamp`, `object`, `detected(y/n)`,
`confidence`, `bbox_w_px`, `bbox_h_px`, `conf_threshold`, `inference_ms`,
`fps`, `model`, `device`, `frame_path`.

You fill: `scenario`, `distance_m`, `lighting`, `background`, `notes`.

## Evaluating the gate

Count from the completed CSV:

| Criterion | Threshold | How to count | Weight |
|-----------|-----------|--------------|--------|
| Scissors | ≥ 8 of 10 | rows with `object=scissors`, `distance_m ≤ 2`, `confidence ≥ 0.40` | **PRIMARY** |
| Knife | ≥ 7 of 10 | rows with `object=knife`, `distance_m ≤ 2`, `confidence ≥ 0.40` | Only if a real knife was available |
| Speed | ≥ 15 FPS | settled end-to-end FPS from test E — display ON, same width/height/imgsz as matrix A, after ≥ 60 s | Required |
| False positives | ≤ 2 | scissors/knife rows in the 3-minute test D | Recorded, does not block |

**Scissors is the primary decider.** If scissors passes **and** speed passes,
Phases 3 and 4 can be skipped and work moves to the Pose/Wrist layer.

If the knife was not tested, that does **not** count as a pass and does **not**
count as a fail. The skip decision still rests on scissors, but the untested
knife is carried forward as an open risk that must be resolved before the system
is trusted on knives.

If scissors fails, or speed fails, or a *tested* knife fails, that specific
criterion is the justification for building a dataset and fine-tuning.

## Outcome record

Fill this in after the run and keep it with the CSV.

```
Date tested        : ____________________
Machine / device   : ____________________  (e.g. MacBook, mps)
Model              : yolo11n.pt
Conf threshold     : 0.25
imgsz              : ______   (same for ALL tests below)
Capture resolution : ______ x ______  (actual, as reported at exit)

Scissors (PRIMARY) : ____ / 10 at <= 2 m with conf >= 0.40   -> PASS / FAIL
Knife              : ____ / 10 at <= 2 m with conf >= 0.40   -> PASS / FAIL
                     or: "not tested - equipment limitation"
Speed (end-to-end) : ______ FPS                              -> PASS / FAIL
    display ON during measurement                : YES / NO
    same width/height/imgsz as matrix A          : YES / NO
    ran >= 60 s, overlay showed GATE-VALID       : YES / NO
    exit summary said "FPS gate status : VALID"  : YES / NO
False positives    : ____ in 3 minutes                       -> PASS / FAIL

All trials used real physical objects, no screen images:  YES / NO

DECISION          : skip Phase 3+4  /  proceed with Phase 3+4
Reason            : ____________________________________________
Open risks carried forward : ___________________________________
```
