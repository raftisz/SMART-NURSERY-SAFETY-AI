# Legacy code

โค้ดต้นฉบับจากช่วงแรกของโปรเจกต์ เก็บไว้เพื่ออ้างอิงและเป็นหลักฐาน **ไม่ใช่ส่วนของ CCTV Core**
เนื้อหาไม่ได้แก้ไข ยกเว้นการลบ LINE credential ที่เคยฝังอยู่ใน `kids_acsident.py`

| ไฟล์ | เดิมอยู่ที่ | คืออะไร | หมายเหตุ |
|---|---|---|---|
| `cigarette.py` | `/cigarette.py` | Colab notebook export: เทรนโมเดล smoking | รันได้เฉพาะบน Google Colab |
| `kids_acsident.py` | `/kids_acsident.py` | Colab notebook export: ตรวจจับเด็กปีนป่ายด้วย YOLO Pose | รันได้เฉพาะบน Colab; weight อยู่ใน Google Drive ไม่อยู่ในรีโป; มีฟังก์ชัน `analyze_and_detect` 3 เวอร์ชันซ้อนกัน |
| `streamlit_smoking_app.py` | `/app/app.py` | Streamlit app ตรวจจับการสูบบุหรี่ + LINE | ส่ง LINE ตรงจากลูปวิดีโอ (ไม่ผ่าน Event Manager) |

## รัน Streamlit app เดิม

ไฟล์นี้โหลดโมเดลจาก `"best.pt"` ในโฟลเดอร์ที่รัน จึงต้องรันจากโฟลเดอร์ `models/smoking/`:

```bash
pip install -r legacy/requirements-legacy.txt
cd models/smoking
streamlit run ../../legacy/streamlit_smoking_app.py
```

(ภาพแจ้งเตือนจะถูกบันทึกลง `models/smoking/alerts/` ซึ่งถูก ignore ใน git)
