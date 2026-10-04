# CCTV Core — Architecture

## ภาพรวม

```
กล้อง → Frame → Detector ──→ DetectorOutput ──→ EventPolicy ──→ EventCandidate
                                                                     │
                                                              EventManager
                                           (ยืนยันเหตุการณ์ / รวม / ปิด / cooldown)
                                                  │            │            │
                                              Snapshot      SQLite    NotificationRouter
                                                                       ├── Console
                                                                       └── LINE (LineNotifier เดิม)
```

กฎหลัก
- **Detector** บอกแค่ว่า "เห็นอะไรในเฟรม" ห้ามส่ง LINE ห้ามบันทึกไฟล์ ห้ามทำ cooldown (มี test ตรวจกฎนี้)
- **EventPolicy** ตัดสินว่าสิ่งที่เห็นนับเป็นเหตุการณ์ประเภทไหน
- **EventManager** เป็นทางเดียวที่นำไปสู่การแจ้งเตือน
- การแจ้งเตือนทำใน thread แยก ลูปวิดีโอไม่ต้องรอเน็ต

## ไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `src/cctv_core/schemas.py` | ชนิดข้อมูลกลาง: `Frame`, `Detection`, `DetectorOutput`, `EventCandidate`, `Event` |
| `src/cctv_core/detector_base.py` | `BaseDetector`, `EventPolicy` (interface ที่ทุก module ต้องทำตาม) |
| `src/cctv_core/registry.py` | สร้าง detector / policy จากชื่อ class ใน config |
| `src/cctv_core/events/policies.py` | `LabelMatchPolicy` ใช้ซ้ำได้: เจอ label ที่กำหนด = เหตุการณ์ |
| `src/cctv_core/events/manager.py` | Event Manager |
| `src/cctv_core/events/store.py` | บันทึกเหตุการณ์ลง SQLite (`data/events.db`) |
| `src/cctv_core/events/snapshot.py` | บันทึกภาพหลักฐาน (`data/snapshots/`) |
| `src/cctv_core/notifications/` | router, console, LINE, ข้อความภาษาไทย |
| `src/cctv_core/factory.py` | ประกอบระบบจาก `config/core.yaml` |
| `config/core.yaml` | ตั้งค่ากล้อง, module, เหตุการณ์, การแจ้งเตือน |

## วงจรชีวิตของ Event

| สถานะ | ความหมาย |
|---|---|
| `CANDIDATE` | เห็นแล้ว แต่ยังไม่ครบเงื่อนไข (ไม่บันทึก ถ้าหายไปก่อนจะถูกทิ้ง) |
| `CONFIRMED` | เห็นครบ `confirm_hits` ครั้งภายใน `confirm_window_s` วินาที → บันทึก + แจ้งเตือน |
| `SUPPRESSED` | ยืนยันแล้ว แต่ยังอยู่ใน `cooldown_s` → บันทึก แต่ไม่แจ้งเตือน (ถ้ายังเห็นอยู่หลังพ้น cooldown จะเปลี่ยนเป็น `CONFIRMED` และแจ้งเตือน 1 ครั้ง) |
| `CLOSED` | ไม่เห็นนานเกิน `close_after_s` วินาที |

นับแยกตาม (กล้อง, ประเภทเหตุการณ์) ถ้ากรรไกรอยู่ในภาพต่อเนื่อง 1 นาที จะได้ 1 เหตุการณ์ ไม่ใช่ 840 ข้อความ

## วิธีเพิ่ม Detection Module ใหม่

1. สร้างไฟล์ใน `src/detectors/` เป็น subclass ของ `BaseDetector` (เขียน `load()` และ `process()`)
2. ถ้ากฎเป็นแบบ "เจอ label นี้ = เหตุการณ์" ใช้ `LabelMatchPolicy` ได้เลย ถ้าซับซ้อนกว่านั้น เขียน subclass ของ `EventPolicy`
3. เพิ่มรายการใน `modules:` ของ `config/core.yaml`
4. (ไม่บังคับ) เพิ่มชื่อภาษาไทยของ event_type ใน `notifications/templates.py` ถ้าไม่เพิ่มจะแสดงเป็น "ตรวจพบเหตุการณ์: <ชื่อ>"

ไม่ต้องแก้ Event Manager หรือระบบแจ้งเตือน

## Module ที่ต่อเข้า Core แล้ว (ขั้น 8)

| Module | Detector | โมเดล | Policy | สถานะใน config |
|---|---|---|---|---|
| `hazard_object` | `src/detectors/hazard_object.py` | `yolo11n.pt` (COCO, ดาวน์โหลดอัตโนมัติ) | `LabelMatchPolicy` scissors/knife (person อย่างเดียวไม่นับ) | เปิด |
| `smoking` | `src/detectors/smoking.py` | `models/smoking/best.pt` | `LabelMatchPolicy` smoking/cigarette | เปิด (รันทุก 2 เฟรม) |
| `climbing` | `src/detectors/climbing_pose.py` | `yolo11n-pose.pt` (pretrained ชั่วคราว) | `ClimbingPosePolicy` กฎเวอร์ชันแรกจาก `legacy/kids_acsident.py` (เพิ่ม: ท่านอนราบไม่นับ, ไม่ใช้สะโพกที่ confidence ต่ำ) | ปิด |
| `fight` | ใช้ผล Pose ชุดเดียวกับ `climbing` (ไม่รันโมเดลเพิ่ม) | `yolo11n-pose.pt` | `FightPolicy` ใน `src/policies/fight.py` | ปิด (เปิดพร้อม climbing ด้วย `--modules climbing_pose`) |
| `out_of_area` | ใช้กรอบ `person` จาก `hazard_object` (ไม่รันโมเดลเพิ่ม) | – | `OutOfAreaPolicy` ใน `src/policies/out_of_area.py` | เปิด (ทำงานเมื่อมีโซนใน `config/zones.yaml`) |

ทั้งสอง detector แบบ YOLO ใช้ `src/detection/yolo_detector.py` เดิมผ่าน `src/detectors/yolo_label.py` โดยไม่แก้ไฟล์เดิม
ค่าตั้งต้นเท่ากับของเดิม: hazard ใช้ค่าจาก `config.yaml` ของ Phase 2, smoking ใช้ conf 0.5 ตามแอป Streamlit เดิม
ยังไม่มี metric ความแม่นยำของทุกโมเดล (ดู `models/README.md`)

ตรวจว่า adapter ให้ผลตรงกับโค้ดเดิม (รันบน Mac):

```bash
python tools/check_adapters.py
```

## รันกับกล้อง 1 ตัว (ขั้น 9)

```bash
python run_core.py                      # ทุก module ที่เปิดใน config + หน้าต่างภาพ
python run_core.py --modules smoking    # เลือกบาง module (ระบุชื่อ module ที่ปิดไว้ = เปิดเฉพาะรอบนี้)
python run_core.py --source clip.mp4    # ใช้ไฟล์วิดีโอ / ภาพ / rtsp:// แทนกล้อง
python run_core.py --console-only       # บังคับไม่ส่ง LINE ในรอบนี้
```

กด `q` หรือ `ESC` ในหน้าต่างภาพเพื่อหยุด (ใช้ได้ทั้งแป้นไทยและอังกฤษ) เมื่อหยุดจะสรุป FPS และเวลาของแต่ละ module
เหตุการณ์บันทึกที่ `data/events.db` และภาพหลักฐานที่ `data/snapshots/` (ทั้งสองถูก ignore ใน git)

## เด็กออกนอกพื้นที่ (Out-of-Area)

ตรวจเด็กที่ไปอยู่ในโซนทางออก (เช่นประตูรั้ว) ในช่วงเวลาที่กำหนด

1. **จุดเท้า** = กึ่งกลางขอบล่างของกรอบ person ต้องอยู่ในโซน
2. **เด็กหรือผู้ใหญ่** = ความสูงในภาพ ÷ ความสูงที่ผู้ใหญ่ยืนตรงจุดนั้นควรเป็น ถ้าน้อยกว่า `child_ratio` (ค่าเริ่มต้น 0.75) นับเป็นเด็ก
3. **ช่วงเวลา** = `active_hours` ของโซน (ว่าง = ตลอดเวลา)
4. **อยู่นานพอ** = `min_duration_s` ของ `out_of_area` ใน `config/core.yaml` (ค่าเริ่มต้น 3 วินาที)

สร้างโซน (ต้องรันใหม่ทุกครั้งที่ย้ายกล้อง):

```bash
python tools/define_zone.py --name main_gate --hours 09:00-15:00
python tools/define_zone.py --name main_gate --source qa_clip.mp4   # วาดบนคลิปของ QA
python tools/define_zone.py --name test --any-person                # ทดสอบเร็ว: ไม่แยกเด็ก
```

ขั้นตอนในหน้าต่าง: คลิกมุมโซนบนพื้น → Enter → ให้ผู้ใหญ่ยืนตรงใกล้กล้อง กด Spacebar หยุดภาพ คลิกศีรษะแล้วคลิกเท้า → Enter → ทำซ้ำตอนยืนไกลกล้อง → Enter
Backspace หรือคลิกขวา = ลบจุดล่าสุด, ESC = ยกเลิก

ข้อจำกัด v1: ผู้ใหญ่ที่นั่งหรือย่อตัวในโซนจะถูกนับเป็นเด็ก, เด็กที่เดิน**เข้า**มาจากนอกโซนก็แจ้งเตือน (ยังไม่ดูทิศทาง), คนที่กรอบถูกขอบภาพตัดจะไม่ถูกวัด, กล้องต้องเห็นเท้า และยังไม่มีผลวัดความแม่นยำ

## ทะเลาะวิวาท (Fight v1)

แบบกฎ ไม่ต้องเทรน ใช้ keypoint จากโมเดล Pose และติดตามแต่ละคนข้ามเฟรมด้วย IoU แบบง่าย นับเป็นท่าทะเลาะเมื่อ

1. คน 2 คนอยู่ชิดกัน (ระยะห่างของกรอบ ≤ `close_gap` × ความสูงเฉลี่ย ค่าเริ่มต้น 0.15)
2. ข้อมือของคนหนึ่งเคลื่อนเร็ว ≥ `speed_threshold` ความสูงลำตัวต่อวินาที (ค่าเริ่มต้น 1.5)
3. ข้อมือที่เร็วนั้นอยู่ในกรอบของอีกคน

และต้องเกิดต่อเนื่อง ≥ 1 วินาที (`min_duration_s` ของ `fight`) กอดกัน = ชิดแต่ช้า, เต้นข้างกัน = เร็วแต่มือไม่เข้าตัวอีกคน, ไฮไฟว์ครั้งเดียว = สั้นเกิน จึงไม่แจ้งเตือน

บนจอจะเห็น `id<N> spd <ความเร็ว>` บนกรอบคน ใช้ตัวเลขนี้จากคลิป QA ปรับ `speed_threshold` ใน `config/core.yaml`

ข้อจำกัด: การเล่นปล้ำ หยอกล้อ จั๊กจี้ จะดูเหมือนทะเลาะ, คนที่บังกันจะไม่มี keypoint, ค่าตั้งต้นยังไม่ได้ปรับ และยังไม่มีผลวัดความแม่นยำ

## Event Schema 1.1

เพิ่มช่อง `subject` (เช่นชื่อโซน) เหตุการณ์ถูกนับแยกตาม (กล้อง, ประเภท, subject) และเพิ่ม `min_duration_s` / `max_gap_s` ให้ประเภทเหตุการณ์ที่ต้องเห็นต่อเนื่องนาน ๆ ก่อนยืนยัน ค่าเริ่มต้นปิด พฤติกรรมเดิมไม่เปลี่ยน

## รัน test

จาก root ของรีโป:

```bash
python -m unittest discover -s tests -t .
```

test ไม่ใช้กล้อง ไม่โหลดโมเดล และไม่ส่ง LINE จริง
