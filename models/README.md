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

## smoking_cls (ตัวยืนยัน) — `models/smoking_cls/smoking_cls_v1_mac.pt`

| รายการ | ค่า |
|---|---|
| ที่มา | `tools/train_smoking_cls.py` บน Mac (Apple M3, mps), 2026-10-04 |
| ฐาน | `yolo11n-cls.pt` (Ultralytics v8.4.0 release, SHA-256 `c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7`) |
| Classes | `notsmoking`, `smoking` |
| Dataset | Mendeley Smoker Detection **Training เท่านั้น** (716 ภาพ, แบ่ง 85/15 seed 0) CC BY 4.0 |
| Training args | epochs 30 (หยุดที่ 13, best epoch 3), imgsz 224, batch 32, patience 10 |
| Ultralytics / torch | 8.4.136 / 2.13.0 |
| SHA-256 | `4f3f2b4c4997f65626d514cb50fe42b4cc8125ac72947ad88117fb30b846a61f` |
| threshold | 0.10 เลือกบน Validation (`docs/eval/smoking_cls_validation.md`) |
| ผลบน Mendeley Testing | detector อย่างเดียว: เจอ 105/112, แจ้งผิด 41/112 → + ตัวยืนยัน: เจอ 99/112, แจ้งผิด 9/112 (`docs/eval/smoking_cls_mendeley_test.md`) |
| สถานะ | **DO NOT ADOPT** (recall ลด 5.4 จุด เกินงบ 5 จุดที่ตั้งไว้ก่อน) ไม่ได้ใช้ในระบบ ยังไม่ได้วัดบนคลิป QA |
| License | AGPL-3.0 (Ultralytics) |

## hazard_object v1 (fine-tune) — `models/hazard/hazard_v1_full.pt` (สร้างจาก Colab)

| รายการ | ค่า |
|---|---|
| ที่มา | `notebooks/hazard_v1_colab.ipynb` (GPU T4) = `tools/build_hazard_dataset.py` → `tools/train_hazard_model.py` |
| ฐาน | `yolo11n.pt` (COCO) **คง 80 คลาสเดิม** ชื่อคลาสเหมือนเดิม จึงใช้แทนกันได้ทันที |
| Dataset | COCO val2017 (replay) + HOD knife + Sohas (มีดในมือ + hard negatives) รายละเอียด/License: `docs/eval/hazard_v1_dataset.md` |
| Training args | epochs 40, imgsz 960, batch 16, freeze 0, SGD lr0 0.002, scale 0.9, seed 0 |
| SHA-256 / ผลวัด | อยู่ใน `hazard_v1_full.train_info.json` และ `docs/eval/hazard_v1_full_eval.md` ที่ได้จาก Colab |
| สถานะ | **ไม่ใช่ค่าเริ่มต้น**: มีด AP +49 จุด แต่กรรไกร −33 จุด (`docs/eval/hazard_v1_full_eval.md`) ใช้เฉพาะเมื่อเน้นมีด |
| License | AGPL-3.0 (Ultralytics); ข้อมูล HOD ใช้เพื่อการวิจัยเท่านั้น |
