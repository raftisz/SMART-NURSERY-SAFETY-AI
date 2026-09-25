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
| `SUPPRESSED` | ยืนยันแล้ว แต่ยังอยู่ใน `cooldown_s` → บันทึก แต่ไม่แจ้งเตือน |
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
| `climbing` | `src/detectors/climbing_pose.py` | `yolo11n-pose.pt` (pretrained ชั่วคราว) | `ClimbingPosePolicy` กฎเวอร์ชันแรกจาก `legacy/kids_acsident.py` | ปิด |

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
python run_core.py --modules smoking    # เลือกบาง module
python run_core.py --source clip.mp4    # ใช้ไฟล์วิดีโอ / ภาพ / rtsp:// แทนกล้อง
python run_core.py --console-only       # บังคับไม่ส่ง LINE ในรอบนี้
```

กด `q` หรือ `ESC` ในหน้าต่างภาพเพื่อหยุด (ใช้ได้ทั้งแป้นไทยและอังกฤษ) เมื่อหยุดจะสรุป FPS และเวลาของแต่ละ module
เหตุการณ์บันทึกที่ `data/events.db` และภาพหลักฐานที่ `data/snapshots/` (ทั้งสองถูก ignore ใน git)

## รัน test

จาก root ของรีโป:

```bash
python -m unittest discover -s tests -t .
```

test ไม่ใช้กล้อง ไม่โหลดโมเดล และไม่ส่ง LINE จริง
