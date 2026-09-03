import streamlit as st
import cv2
import os
import time
import requests
from datetime import datetime
from ultralytics import YOLO


# ==============================
# CONFIG
# ==============================

MODEL_PATH = "best.pt"
ALERT_DIR = "alerts"

os.makedirs(ALERT_DIR, exist_ok=True)


# ==============================
# PAGE
# ==============================

st.set_page_config(
    page_title="AI CCTV Smoking Detection",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 AI CCTV Smoking Detection")
st.write("ระบบ AI ตรวจจับการสูบบุหรี่แบบ Real-time")


# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


try:
    model = load_model()
    st.success("✅ โหลด YOLO best.pt สำเร็จ")
except Exception as e:
    st.error(f"❌ โหลดโมเดลไม่ได้: {e}")
    st.stop()


# ==============================
# SIDEBAR
# ==============================

st.sidebar.header("⚙️ ตั้งค่าระบบ")

confidence = st.sidebar.slider(
    "Confidence",
    0.1,
    0.95,
    0.50,
    0.05
)

cooldown = st.sidebar.number_input(
    "LINE Cooldown (วินาที)",
    5,
    300,
    30
)


st.sidebar.header("📱 LINE")

LINE_TOKEN = st.sidebar.text_input(
    "Channel Access Token",
    type="password"
)

USER_ID = st.sidebar.text_input(
    "Your User ID",
    type="password"
)


# ==============================
# SESSION
# ==============================

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if "last_alert" not in st.session_state:
    st.session_state.last_alert = 0

if "detections" not in st.session_state:
    st.session_state.detections = 0

if "alerts" not in st.session_state:
    st.session_state.alerts = 0


# ==============================
# LINE FUNCTION
# ==============================

def send_line(message):

    if not LINE_TOKEN or not USER_ID:
        return False, "ยังไม่ได้กรอก LINE Token / User ID"

    url = "https://api.line.me/v2/bot/message/push"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_TOKEN}"
    }

    data = {
        "to": USER_ID,
        "messages": [
            {
                "type": "text",
                "text": message
            }
        ]
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=10
        )

        if response.status_code == 200:
            return True, "ส่ง LINE สำเร็จ"

        return False, response.text

    except Exception as e:
        return False, str(e)


# ==============================
# DASHBOARD
# ==============================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🚬 ตรวจพบ Smoking",
        st.session_state.detections
    )

with col2:
    st.metric(
        "🚨 LINE Alerts",
        st.session_state.alerts
    )

with col3:

    if st.session_state.camera_running:
        st.metric("📹 Camera", "ONLINE")
    else:
        st.metric("📹 Camera", "OFFLINE")


# ==============================
# BUTTON
# ==============================

start = st.button(
    "▶️ START CAMERA",
    use_container_width=True
)

stop = st.button(
    "⏹️ STOP CAMERA",
    use_container_width=True
)


if start:
    st.session_state.camera_running = True

if stop:
    st.session_state.camera_running = False


# ==============================
# VIDEO
# ==============================

video_placeholder = st.empty()
status_placeholder = st.empty()


# ==============================
# CAMERA
# ==============================

if st.session_state.camera_running:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        st.error("❌ ไม่สามารถเปิดกล้องโน้ตบุ๊กได้")

        st.session_state.camera_running = False

    else:

        status_placeholder.success(
            "🟢 กล้องกำลังทำงาน"
        )

        while st.session_state.camera_running:

            ret, frame = cap.read()

            if not ret:
                st.error("❌ อ่านภาพจากกล้องไม่ได้")
                break


            # ==========================
            # YOLO DETECTION
            # ==========================

            results = model(
                frame,
                conf=confidence,
                verbose=False
            )

            result = results[0]

            smoking_found = False
            smoking_conf = 0


            if result.boxes is not None:

                for box in result.boxes:

                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    class_name = model.names[cls_id]

                    if class_name.lower() in [
                        "smoking",
                        "cigarette"
                    ]:

                        smoking_found = True

                        if conf > smoking_conf:
                            smoking_conf = conf


            # ==========================
            # DRAW RESULT
            # ==========================

            annotated = result.plot()


            # ==========================
            # ALERT
            # ==========================

            if smoking_found:

                st.session_state.detections += 1

                cv2.putText(
                    annotated,
                    "WARNING: SMOKING DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3
                )


                now = time.time()

                # LINE COOLDOWN

                if now - st.session_state.last_alert >= cooldown:

                    timestamp = datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )

                    image_path = os.path.join(
                        ALERT_DIR,
                        f"alert_{timestamp}.jpg"
                    )

                    cv2.imwrite(
                        image_path,
                        annotated
                    )


                    message = (
                        "🚨 AI CCTV ALERT 🚨\n\n"
                        "⚠️ ตรวจพบการสูบบุหรี่\n\n"
                        f"🚬 Class: smoking\n"
                        f"📊 Confidence: {smoking_conf:.2f}\n"
                        f"🕐 เวลา: "
                        f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n"
                        "📸 บันทึกภาพหลักฐานแล้ว"
                    )


                    success, result_message = send_line(
                        message
                    )


                    if success:

                        st.session_state.alerts += 1

                        st.session_state.last_alert = now

                        status_placeholder.error(
                            "🚨 พบการสูบบุหรี่ → ส่ง LINE แล้ว ✅"
                        )

                    else:

                        status_placeholder.warning(
                            f"⚠️ ตรวจพบ แต่ LINE ส่งไม่ได้: "
                            f"{result_message}"
                        )

                else:

                    status_placeholder.warning(
                        "⚠️ พบการสูบบุหรี่ "
                        "แต่ยังอยู่ใน LINE Cooldown"
                    )

            else:

                status_placeholder.success(
                    "🟢 ไม่พบการสูบบุหรี่"
                )


            # ==========================
            # DISPLAY
            # ==========================

            frame_rgb = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            video_placeholder.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True
            )


            time.sleep(0.03)


        cap.release()

else:

    st.info(
        "📹 กด START CAMERA เพื่อเริ่มระบบ"
    )