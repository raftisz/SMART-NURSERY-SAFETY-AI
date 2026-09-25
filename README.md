# NurseryGuard AI

> **CCTV Core All Detection — โครงสร้างรีโป (กำลังปรับเป็น Modular CCTV Core)**
>
> | โฟลเดอร์ / ไฟล์ | เนื้อหา |
> |---|---|
> | `main.py`, `detect.py`, `config.yaml`, `src/`, `tools/`, `docs/`, `data/` | NurseryGuard เดิม (ย้ายจาก `nursery-guard-ai-main/` มาไว้ที่ root, เนื้อหาไม่เปลี่ยน) |
> | `models/` | weight ของโปรเจกต์ + ทะเบียนโมเดล ดู [`models/README.md`](models/README.md) |
> | `legacy/` | โค้ดต้นฉบับ smoking / climbing / Streamlit ดู [`legacy/README.md`](legacy/README.md) |
> | `evidence/` | ภาพหน้าจอแจ้งเตือน LINE จากการทดสอบเดิม |
> | `src/cctv_core/`, `config/core.yaml` | CCTV Core ใหม่ ดู [`docs/architecture.md`](docs/architecture.md) |
> | `tests/` | unit test ของ Core: `python -m unittest discover -s tests -t .` |
> | `.env.example` | ตัวอย่างตัวแปร LINE — คัดลอกเป็น `.env` แล้วใส่ค่าจริง (ห้าม commit `.env`) |
>
> เอกสารด้านล่างทั้งหมดเป็นของ NurseryGuard เดิม คำว่า `nursery_guard_ai/` ในเอกสารหมายถึง root ของรีโปนี้
> `detect.py` คือเครื่องมือทดสอบ Phase 2 และผูกกับ Decision Gate จึงคงไว้ตามเดิม (ยังส่ง LINE ตรง) ไม่ใช่ส่วนของ CCTV Core

A real-time dangerous object and child risk detection system for nursery environments.

The system watches a live webcam or CCTV stream, detects dangerous objects and the
hands of people in the monitored zone, and raises an alert when a hand gets close to
a dangerous object.

## Concept

| Stage | What happens |
|-------|--------------|
| 1 | Read frames from a webcam / CCTV stream |
| 2 | Detect dangerous objects (**scissors**, **knife**) with a pretrained YOLO model |
| 3 | Detect people with the same YOLO model |
| 4 | Detect hand landmarks with MediaPipe Hands |
| 5 | Measure the spatial relationship between hands and dangerous objects |
| 6 | Classify the risk level |
| 7 | Alert, screenshot, and log HIGH-risk events |

### Risk levels

| Level | Condition |
|-------|-----------|
| **LOW** | A dangerous object is detected, no hand nearby |
| **MEDIUM** | A hand is near a dangerous object |
| **HIGH** | A hand is very close to, or overlapping, a dangerous object |

A HIGH-risk event triggers a real-time alert, saves a screenshot with a timestamp,
and writes a record to a local SQLite log.

## Scope

**In scope:** pretrained YOLO object/person detection, MediaPipe hand detection,
distance-based risk rules, local alerting, screenshots, SQLite event log.

**Deliberately out of scope:** custom dataset collection, child age classification,
face recognition, action recognition models (LSTM / 3D CNN / SlowFast),
multi-camera tracking, and edge deployment.

## Project structure

```
nursery_guard_ai/
├── main.py              # entry point (Phase 1: environment + webcam test)
├── requirements.txt     # Python dependencies
├── README.md
│
├── src/
│   ├── detection/       # YOLO object/person detection, MediaPipe hands
│   ├── risk_analysis/   # hand-to-object distance and risk classification
│   ├── alerts/          # real-time alerting and screenshot saving
│   └── utils/           # config, drawing helpers, SQLite logging
│
├── screenshots/         # saved HIGH-risk event images
└── database/            # SQLite event log
```

The `src/` sub-packages are empty placeholders in Phase 1 and are filled in later phases.

## Requirements

- **Python 3.12** — required. MediaPipe does not publish wheels for Python 3.13 or 3.14.
- A working webcam.
- Roughly **4 GB of free disk space** (PyTorch, which Ultralytics YOLO depends on, is large).

## Setup

```bash
cd nursery_guard_ai

# 1. Create a virtual environment on Python 3.12
python3.12 -m venv venv

# 2. Activate it
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

On macOS, the first run will ask for camera permission. If your terminal was never
granted access, enable it under
**System Settings → Privacy & Security → Camera**, then restart the terminal.

## Running the Phase 1 check

```bash
source venv/bin/activate
python main.py
```

This verifies three things and prints the result of each:

1. The Python environment and package versions.
2. That OpenCV can open the webcam.
3. That frames can be read and displayed in a window.

A live window opens showing the camera feed. Press **`q`** or **ESC** in that window
to quit. On success the script prints `PHASE 1 PASSED` and exits with code 0.

If you have more than one camera, select it by index:

```bash
python main.py --camera 1
```

## Roadmap

- [x] **Phase 1** — project structure, environment, webcam verification
- [ ] **Phase 2** — YOLO detection of dangerous objects and people
- [ ] **Phase 3** — MediaPipe hand detection
- [ ] **Phase 4** — hand-to-object risk classification
- [ ] **Phase 5** — alerts, screenshots, SQLite event logging

---

# Phase 2 — Pretrained Detection

Phase 2 tests whether a **pretrained** model detects `scissors` and `knife` well
enough that building a custom dataset (Phase 3) and fine-tuning (Phase 4) can be
skipped. Nothing is trained in this phase.

Phase 2 is a **separate entry point**. `main.py` is unchanged and remains the
Phase 1 webcam check.

## Model

- **YOLO11 nano** (`yolo11n.pt`), downloaded automatically on first run.
- Pretrained on **COCO**, 80 classes.
- Class IDs are resolved at runtime from `model.names` and printed at startup —
  never hardcoded, because the mapping belongs to the weights, not to the code.

Resolved mapping for `yolo11n.pt`:

| Class | ID |
|-------|-----|
| `person` | 0 |
| `knife` | 43 |
| `scissors` | 76 |

Filtering is done inside the model via `predict(classes=[0, 43, 76])`, so
non-target classes are discarded before they reach Python.

### Supported classes and the Cutter gap

COCO has no `cutter` / box-cutter / craft-knife class. A cutter can only ever be
detected here by being misread as `knife`. **If the Cutter class is required,
Phases 3 and 4 cannot be skipped regardless of how the scissors and knife tests
turn out** — no threshold tuning adds a class the weights do not contain.

## How to run

```bash
source venv/bin/activate

python detect.py                       # webcam 0 (default)
python detect.py --source 1            # a different webcam
python detect.py --source clip.mp4     # video file
python detect.py --source photo.jpg    # still image
```

Model size is selectable with `--model` (also `model.weights` in
`config.yaml`). Weights download automatically on first use, and class IDs are
re-resolved from `model.names` for whichever model is loaded — never assumed to
match another model. The filename appears in the startup banner, the exit
summary, and the `model` column of every CSV row, so trials stay traceable.

```bash
python detect.py --model yolo11s.pt              # larger model
python detect.py --compare-models yolo11s.pt     # run BOTH, observation only
```

`--compare-models` runs a second model on the same frames and prints a
side-by-side line every 5 s. Display and CSV keep using the primary `--model`
only. **FPS in this mode is not gate-valid** — two models run per frame.

Detection labels include the box area (`scissors 0.52 [2310px]`), and each `s`
press prints the smallest and largest box in that frame, so the
confidence-vs-object-size relationship can be read off directly.

Useful flags: `--conf 0.4` (starting threshold), `--imgsz 640` (model input
size), `--width` / `--height` (webcam capture resolution), `--display-scale 0.5`
(shrinks the window only), `--buffersize 1`, `--backend`,
`--device cpu|mps|cuda|auto`, `--flip` / `--no-flip`, and `--no-display` /
`--max-frames N` for headless checks.

`--display-scale` affects **only** the image passed to `cv2.imshow`. Inference
input, box coordinates, saved frames and every CSV value are unchanged by it.

For long runs the loop prints a timing line every 60 s, flagged `<- degrading`
if FPS has fallen more than 10% below the first logged minute, so thermal drift
is visible while testing rather than only at exit.

To choose a capture resolution on evidence rather than theory:

```bash
python tools/bench_capture.py       # measures cap.read() alone, no inference
```

Keep `--width`, `--height` and `--imgsz` **identical** across the whole test
matrix, and use those same values for the speed measurement. All three are
recorded in every CSV row so the run can be audited afterwards.

## Keyboard controls

| Key | Action |
|-----|--------|
| `s` | Save the annotated frame to `data/test_frames/` and append one row per detection to `docs/phase2_test_results.csv` |
| `+` | Raise the confidence threshold by 0.05 |
| `-` | Lower the confidence threshold by 0.05 |
| `h` | Toggle the HUD bar |
| `q` / `ESC` | Quit |

> **The keyboard input source must be English/ABC.** `cv2.waitKey` returns the
> Unicode codepoint of the character produced, not the physical key. With a Thai
> layout, `s` sends ห (U+0E2B, 3627), and the usual `& 0xFF` idiom folds that to
> 43 — which is `+`. The program now rejects non-ASCII codes and prints a
> `[WARN]` instead, but the hotkeys still will not work until you switch layout.

Every accepted key is echoed as `[KEY] <char>`, and `+`/`-` print
`[CONF] old -> new`. Only `+` and `-` change the threshold — `=` and `_` are not
aliases.

The HUD is a solid bar **above** the video, not an overlay, so its text can never
cover a bounding box. It shows a per-stage timing breakdown (capture / inference
/ draw / display, rolling medians over 30 frames), which is how you find time
that is not inference.

A second class-wise NMS pass (`nms_iou` in `config.yaml`, default 0.5) removes
duplicate boxes on one object. Ultralytics' own NMS runs at IoU 0.7, which is
loose enough to leave two boxes on a single object; the HUD shows `(NMS -N)`
when duplicates were suppressed.

Pressing `s` with **no** detections writes a single row with `detected(y/n)=n`,
so a miss is recorded rather than lost.

## Configuration

Thresholds and target classes live in `config.yaml`, not in the source.

## On-screen overlay

Both figures are rolling averages over the last **30 frames**:

- `infer … ms` — model inference time only.
- `loop … FPS` — end-to-end rate, including capture, drawing and display.

The two differ substantially; the Decision Gate uses the **end-to-end** figure.
Let it run ~30 seconds before reading it, or start-up cost skews the average.

## Current limitations

- **No `cutter` class exists in COCO** (see above).
- COCO `scissors` and `knife` are photographed mostly as kitchen/desk objects on
  plain surfaces. Detection of a *held*, partly-occluded, or angled blade — which
  is exactly the nursery risk case — is expected to be the weak point.
- YOLO11n is the smallest variant, chosen for speed. If accuracy fails the gate
  but speed passes comfortably, trying `yolo11s.pt` is cheaper than building a
  dataset and should be ruled out before committing to Phase 3.
- Small objects at ≥ 2 m occupy few pixels and are the likeliest misses.
- Every measurement so far is from **file input on the developer machine**. No
  webcam numbers exist yet.

## Decision Gate

Fixed before data collection. Evaluated on results you collect yourself, using
`docs/phase2_test_protocol.md`.

| Criterion | Pass condition | Weight |
|-----------|----------------|--------|
| **Scissors** | detected in **≥ 8 of 10** trials at **≤ 2 m** with **confidence ≥ 0.40** | **PRIMARY — this decides the gate** |
| **Knife** | detected in **≥ 7 of 10** trials at **≤ 2 m** with **confidence ≥ 0.40** | Conditional — only if a real knife is available |
| **Speed** | **end-to-end ≥ 15 FPS**, measured **with the display ON**, at the **same `--width`/`--height`/`--imgsz` as the scissors matrix**, after **≥ 60 s** of continuous running | Required |
| **False positives** | **≤ 2** false detections during a **3-minute** control test | Recorded, does not block |

### Speed-measurement rule

An FPS number counts toward the gate only if **all four** hold:

1. **Display ON.** `--no-display` skips window drawing and `waitKey`, so it
   flatters the result. Those runs are diagnostic only and must never be quoted.
2. **Same settings as the scissors matrix** — identical `--width`, `--height`
   and `--imgsz`. Changing input size changes both speed and accuracy, so a fast
   number taken at a smaller size does not describe the configuration that was
   actually tested for accuracy.
3. **≥ 60 s of continuous running** before the reading is taken.
4. Read the settled rolling average, not the first few seconds.

The program enforces this rather than relying on memory. The overlay shows
`warming up NN/60s` and turns to `GATE-VALID` only once the run qualifies; the
exit summary prints `FPS gate status: VALID / NOT VALID`, and every saved CSV row
carries `fps_gate_valid` plus the `imgsz`, `capture_w` and `capture_h` in force.

### Physical-object rule (applies to every trial)

Every trial must use a **real physical object** in the room. Pointing the camera
at a photograph of scissors or a knife — on a phone, a monitor, a printout — is
**not a valid trial** and must never be recorded as one. A screen image has
different lighting, no depth, and screen texture, so it does not tell you whether
the model works on the real thing. Any such run is void.

### If no real knife is available

Do **not** substitute a photo, and do **not** leave the knife result blank or
implied. Record it literally as:

> **Knife: not tested — equipment limitation**

An untested criterion is **not** a pass and **not** a fail. It is carried forward
as a known, documented gap.

### Outcome

- **Scissors PASS + Speed PASS** → **skip Phases 3 and 4**, go to the
  Pose/Wrist layer. Scissors is the primary decider.
  - If knife was tested and passed, record it as confirmed.
  - If knife was *not tested*, the skip still stands on the scissors result, but
    the report must carry "knife: not tested — equipment limitation" as an open
    risk, to be resolved before the system is trusted on knives.
- **Scissors FAIL** → Phases 3 and 4 are justified. Name the exact shortfall
  (e.g. "6/10 at ≥ 0.40").
- **Speed FAIL** → a smaller input size or a lighter pipeline is needed before
  any further layers are added.
- **Knife tested and FAILED** → recorded as a real fail, and justifies Phase 3/4
  for the knife class specifically, even if scissors passes.
- **Cutter class required → Phases 3 and 4 are required regardless**, since COCO
  cannot supply that class.

The false-positive count does not by itself block skipping to the Pose layer, but
a failure there must be recorded, since false alarms drive alert fatigue.

---

## LINE alerts

Pushes a LINE message when **scissors** or **knife** is detected. A `person`
detection alone never triggers an alert.

> Uses the **LINE Messaging API** push endpoint. LINE Notify was discontinued in
> 2025 and is not used.

### 1. Create a Messaging API channel

1. Sign in at [developers.line.biz](https://developers.line.biz/console/).
2. Create a **Provider**, then a channel of type **Messaging API**.
3. Open the channel → **Messaging API** tab → issue a **Channel access token
   (long-lived)**. Copy it.
4. On the same tab, scan the QR code with the LINE app to **add the bot as a
   friend**. The push will fail with `403` if you are not a friend of it.
5. Get your **User ID** (starts with `U`): it is shown under the **Basic
   settings** tab as *Your user ID*.

### 2. Put the credentials in `.env`

```bash
cp .env.example .env
```

Then edit `.env`:

```
LINE_CHANNEL_ACCESS_TOKEN=<the long-lived channel access token>
LINE_USER_ID=U1234567890abcdef...
LINE_IMAGE_BASE_URL=
```

`.env` is gitignored and must never be committed. Only `.env.example` is
tracked, and it holds no values.

`LINE_IMAGE_BASE_URL` is optional and normally left blank: LINE can only attach
an image by **public HTTPS URL**, so a locally saved frame cannot be uploaded
through this endpoint. Left blank, the alert text names the saved frame instead.

### 3. Verify the credentials without the camera

```bash
source venv/bin/activate
python detect.py --line-test
```

This sends one message and exits — no camera, no model. It prints
`LINE test PASSED` (exit 0) or the exact failure reason (exit 1). Common causes:
`401` = wrong token, `403` = you have not added the bot as a friend.

### 4. Run detection with alerts on

```bash
python detect.py --source 0 --line
```

Alerts are **off** unless `--line` is passed; without it behaviour is unchanged.

| Log line | Meaning |
|---|---|
| `[LINE] sent` | Message delivered |
| `[LINE] cooldown Ns remaining` | Suppressed — still inside the cooldown |
| `[LINE] error <reason>` | Failed; detection continues regardless |

### Cooldown

`alerts.line_cooldown_seconds` in `config.yaml` (default **30**). Without it a
14 FPS loop would push hundreds of messages a minute. The slot is claimed before
the request is sent, so a burst of frames cannot each start their own send.

Sending happens on a daemon thread, so the video loop never blocks on the
network, and every failure is caught — an unreachable LINE cannot stop detection.
