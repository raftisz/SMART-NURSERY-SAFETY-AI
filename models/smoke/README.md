# smoke — `models/smoke/smoke_custom_yolo11n_v2.pt`

โมเดลตรวจควัน (class เดียว) ที่ฝึกเอง **ยังไม่ต่อเข้า pipeline หลัก** (`config/core.yaml` ไม่ได้อ้างถึง)

| รายการ | ค่า |
|---|---|
| ฐาน | `yolo11n.pt` (YOLO11 nano, detection) |
| Classes | `0: smoke` |
| ที่มา | `notebooks/smoke_custom_yolo11n_v2_colab.ipynb` (Colab, Tesla T4) ชุดข้อมูลสร้างด้วย `tools/build_smoke_dataset_v2.py` |
| Training args | epochs 50, imgsz 640, batch 32, seed 0 |
| Dataset | train 2,884 / val 281 รูป แบ่งตามกลุ่ม ไม่มีภาพเกือบซ้ำข้าม train/val (dHash ≤ 4 bit) |
| Ultralytics / torch | 8.4.136 / 2.11.0+cu130 |
| SHA-256 | `2a97455c6f96252dfcedf92c149154db1b32d8eee0ef0028bb2d9c80cad95876` |
| Val split (281 รูป, conf ตามค่า val ของ Ultralytics) | Precision 0.711 / Recall 0.552 / mAP50 0.668 / mAP50-95 0.372 |
| เทสสดด้วยกล้อง (`tools/smoke_live_test.py`, conf 0.25) | A (ไม่มีควัน 60 วิ) = 0 เฟรม, B (มีควัน ครั้งละ 10 วิ) = 4/5 ครั้ง, C (คล้ายควันแต่ไม่ใช่ 60 วิ) = 0 เฟรม |
| License ของ weight | AGPL-3.0 (Ultralytics) |

**ข้อจำกัด:** ตัวเลข val มาจากชุดข้อมูล (ภาพจากเว็บและภาพทางเดิน) ไม่ใช่กล้องของเรา
ผลเทสสดเป็น**ผลที่ผู้ใช้รายงานเอง** ทดสอบ B 5 ครั้งในห้องเดียว ยังไม่มีภาพหลักฐาน
**ยังไม่ใช่ความแม่นยำในการใช้งานจริง**

## ชุดข้อมูลที่ใช้ (CC BY 4.0 ทั้ง 3 แหล่ง)

- **ScPuteri** — "Cigarette Vape Smoke" v11, Roboflow Universe
  (universe.roboflow.com/scputeri/cigarette-vape-smoke-kcg4j), CC BY 4.0 — ใช้เฉพาะกล่อง smoke
- **ITLP-Campus-Indoor** — OPR-Project, Hugging Face
  (huggingface.co/datasets/OPR-Project/ITLP-Campus-Indoor), CC BY 4.0 — 400 รูปทางเดินเป็นภาพไม่มีควัน
- **Indoor Fire Smoke Dataset** — Putra, A.K., Binus University, Zenodo
  DOI 10.5281/zenodo.15826133, CC BY 4.0 — ใช้เฉพาะกล่อง smoke (ตัดกล่อง fire ออก), ไม่ใช้ test split
