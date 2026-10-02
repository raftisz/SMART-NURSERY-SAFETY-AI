"""Alert text. Keeps the "🚨 AI CCTV ALERT 🚨" format already used on the LINE OA."""

from __future__ import annotations

from datetime import datetime

from ..schemas import Event

TITLES: dict[str, str] = {
    "smoking": "ตรวจพบการสูบบุหรี่",
    "hazard_object": "ตรวจพบวัตถุอันตราย",
    "climbing": "ตรวจพบพฤติกรรมเสี่ยงปีนป่าย",
    "out_of_area": "ตรวจพบเด็กออกนอกพื้นที่",
}


def title_for(event_type: str) -> str:
    return TITLES.get(event_type, f"ตรวจพบเหตุการณ์: {event_type}")


def format_message(event: Event, camera_names: dict[str, str] | None = None) -> str:
    camera = (camera_names or {}).get(event.camera_id, event.camera_id)
    when = datetime.fromtimestamp(event.confirmed_at or event.last_seen_at)
    lines = [
        "🚨 AI CCTV ALERT 🚨",
        "",
        f"⚠️ {title_for(event.event_type)}",
        "",
    ]
    if event.peak_detection is not None:
        lines.append(f"🎯 ประเภท: {event.peak_detection.label}")
    if event.subject:
        lines.append(f"📍 โซน: {event.subject}")
    lines += [
        f"📊 Confidence: {event.peak_confidence:.2f}",
        f"⚡ ระดับ: {event.severity.name}",
        f"🕐 เวลา: {when.strftime('%d/%m/%Y %H:%M:%S')}",
        f"📹 กล้อง: {camera}",
    ]
    # Only claim evidence was saved when it really was.
    if event.snapshot_path:
        lines.append("📸 บันทึกภาพเหตุการณ์แล้ว")
    return "\n".join(lines)
