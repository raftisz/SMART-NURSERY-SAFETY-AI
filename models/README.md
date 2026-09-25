# Model Registry

ทะเบียนโมเดลของโปรเจกต์ ห้ามแก้ไข / เทรนทับ / เปลี่ยน architecture ของ weight ในโฟลเดอร์นี้
ถ้าจะเพิ่มหรือเปลี่ยนโมเดล ให้เพิ่มแถวใหม่ อย่าเขียนทับของเดิม

## smoking — `models/smoking/best.pt`

| รายการ | ค่า |
|---|---|
| ที่มา | เทรนด้วย `legacy/cigarette.py` (Google Colab) |
| ฐาน | `yolo11n.pt` (YOLO11 nano, detection) |
| Classes | `0: cigarette`, `1: face`, `2: smoking` |
| Training args | epochs 100, imgsz 640, batch 16 |
| Dataset | Roboflow `smoking.v2-2024-06-10` (ตามชื่อไฟล์ zip ใน `cigarette.py`) รายละเอียดอื่น: UNKNOWN |
| วันที่เทรน | 2026-08-31 (จาก metadata ในไฟล์) |
| Ultralytics | 8.4.136 |
| SHA-256 | `fded2de373f09f73628930ea4b7249efba061fadac73ca796dc82ffdf63841e2` |
| Metrics (mAP / Precision / Recall) | **UNKNOWN** — ยังไม่มีผลวัด ห้ามสรุปว่าแม่นหรือไม่แม่น |
| License | AGPL-3.0 (Ultralytics) |
| ตำแหน่งเดิม | `best/best.pt` (ย้ายด้วย `git mv`, เนื้อไฟล์ไม่เปลี่ยน) |

## hazard_object — `yolo11n.pt` (COCO pretrained)

| รายการ | ค่า |
|---|---|
| ที่มา | Ultralytics ดาวน์โหลดอัตโนมัติตอนรันครั้งแรก (ไม่ได้เก็บในรีโป, ถูก ignore) |
| Classes ที่ใช้ | `person` (0), `knife` (43), `scissors` (76) |
| Metrics ในบริบทโปรเจกต์ | อยู่ระหว่างทดสอบตาม `docs/phase2_test_protocol.md` ผล Decision Gate: **UNKNOWN** |

## climbing_pose — ยังไม่มีในรีโป

| รายการ | ค่า |
|---|---|
| ที่มา | `legacy/kids_acsident.py` โหลดจาก Google Drive ของเจ้าของเดิม: `Child_Pose_Project/runs/child_climbing_pose/weights/best.pt` |
| สถานะ | **ไม่มีไฟล์ในรีโป** โค้ดเทรนและ dataset: UNKNOWN |
| ทางเลือกชั่วคราว | `yolo11n-pose.pt` (pretrained, ไม่เทรนเพิ่ม) |
