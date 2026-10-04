"""Alert text. Keeps the "🚨 AI CCTV ALERT 🚨" format already used on the LINE OA."""

from __future__ import annotations

from datetime import datetime

from ..schemas import Event

TITLES: dict[str, str] = {
    "smoking": "ตรวจพบการสูบบุหรี่",
    "hazard_object": "ตรวจพบของมีคม",
    # pose rules are not measured yet: ask a teacher to check rather than state it happened
    "climbing": "พบท่าทางเสี่ยงปีนป่าย โปรดตรวจสอบ",
    "out_of_area": "ตรวจพบเด็กออกนอกพื้นที่",
    "fight": "พบท่าทางคล้ายการทะเลาะวิวาท โปรดตรวจสอบ",
}

# Event types whose model label is not reliable enough to state as the kind of
# object (the COCO model often calls a held knife "scissors").
GENERIC_KIND: dict[str, str] = {"hazard_object": "ของมีคม"}


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
        label = event.peak_detection.label
        kind = GENERIC_KIND.get(event.event_type)
        lines.append(f"🎯 ประเภท: {kind} (โมเดลอ่านว่า {label})" if kind else f"🎯 ประเภท: {label}")
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
