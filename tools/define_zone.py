"""Draw an Out-of-Area zone on the camera image and calibrate adult height.

Usage (from the repo root):
    python tools/define_zone.py --name main_gate
    python tools/define_zone.py --name main_gate --hours 09:00-15:00
    python tools/define_zone.py --name main_gate --source clip.mp4     # draw on a QA video
    python tools/define_zone.py --name test --any-person               # no calibration, any person counts

Steps in the window (mouse + Enter; works with a Thai keyboard):
    1. ZONE : click the corners of the exit area on the FLOOR, then Enter
    2. NEAR : an adult stands upright inside/near the zone, close to the camera.
              SPACE freezes the picture. Click the top of the head, then the feet. Enter
    3. FAR  : same, with the adult standing far from the camera. Enter
Backspace or right-click = undo last point, ESC = cancel.
The zone is saved to config/zones.yaml (replacing a zone with the same name).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cctv_core.zones import HeightModel, Zone, parse_hours, save_zone  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

ENTER, ESC, SPACE = {13, 10}, 27, 32
UNDO = {8, 127}

THAI_HELP = {
    "zone": "ขั้น 1: คลิกมุมของพื้นที่ทางออกบนพื้น (อย่างน้อย 3 จุด) แล้วกด Enter",
    "near": "ขั้น 2: ให้ผู้ใหญ่ยืนตรงใกล้กล้อง กด Spacebar หยุดภาพ คลิกบนสุดของศีรษะ แล้วคลิกที่เท้า กด Enter",
    "far": "ขั้น 3: ให้ผู้ใหญ่ยืนตรงไกลกล้อง กด Spacebar หยุดภาพ คลิกบนสุดของศีรษะ แล้วคลิกที่เท้า กด Enter",
}
EN_HELP = {
    "zone": "1/3 ZONE: click floor corners, ENTER when done",
    "near": "2/3 NEAR adult: SPACE freeze, click HEAD then FEET, ENTER",
    "far": "3/3 FAR adult: SPACE freeze, click HEAD then FEET, ENTER",
}


def normalise(points, w: int, h: int):
    return [(round(x / w, 4), round(y / h, 4)) for x, y in points]


def calibration_point(head, feet, w: int, h: int) -> tuple[float, float]:
    """(foot_y, height) normalised, from two clicks.

    The feet are the LOWER click (larger y), whichever was clicked first.
    """
    top, bottom = sorted((head[1], feet[1]))
    foot_y = bottom / h
    height = (bottom - top) / h
    if height <= 0:
        raise ValueError("head and feet clicks are at the same height")
    return round(foot_y, 4), round(height, 4)


MIN_CALIBRATION_SPREAD = 0.05  # near and far feet must be >= 5% of the image height apart


def build_zone(name, polygon_px, near_px, far_px, w, h, hours, any_person) -> Zone:
    calibration = None
    if not any_person:
        nf, nh = calibration_point(near_px[0], near_px[1], w, h)
        ff, fh = calibration_point(far_px[0], far_px[1], w, h)
        if nf <= ff:
            raise ValueError("เท้าของจุด 'ใกล้' ต้องอยู่ต่ำกว่าจุด 'ไกล' ในภาพ (อาจสลับขั้น 2 กับ 3)")
        if nf - ff < MIN_CALIBRATION_SPREAD:
            raise ValueError("จุด 'ใกล้' กับ 'ไกล' อยู่ใกล้กันเกินไป ให้ผู้ใหญ่ยืนห่างกันมากกว่านี้")
        if nh <= fh:
            raise ValueError("ผู้ใหญ่ที่ยืน 'ใกล้' ต้องดูสูงกว่าตอนยืน 'ไกล' ลองวาดใหม่")
        calibration = HeightModel(nf, nh, ff, fh)
    return Zone(name=name, polygon=normalise(polygon_px, w, h), active_hours=hours,
                child_filter=not any_person, calibration=calibration)


def run_ui(source, steps):
    import cv2
    import numpy as np

    points = {s: [] for s in steps}
    state = {"step": 0, "frozen": None}
    need = {"zone": None, "near": 2, "far": 2}

    def on_mouse(event, x, y, _flags, _param):
        step = steps[state["step"]]
        if event == cv2.EVENT_LBUTTONDOWN:
            if need[step] is None or len(points[step]) < need[step]:
                points[step].append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points[step]:
            points[step].pop()

    win = "define zone"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    size = None
    last = None
    print("\n" + THAI_HELP[steps[0]])
    try:
        while True:
            if state["frozen"] is None:
                frame = source.read()
                if frame is None:  # video/image ended: keep showing the last frame
                    if last is None:
                        print("ไม่มีภาพจาก source")
                        return None, None
                    frame = last
                    state["frozen"] = frame
                last = frame
            else:
                frame = state["frozen"]
            if size is None:
                size = (frame.shape[1], frame.shape[0])
                cv2.resizeWindow(win, min(1280, size[0]), int(min(1280, size[0]) * size[1] / size[0]))

            view = frame.copy()
            if len(points["zone"]) >= 2:
                cv2.polylines(view, [np.array(points["zone"], dtype=np.int32)], True, (0, 165, 255), 2)
            for s, color in (("zone", (0, 165, 255)), ("near", (0, 255, 0)), ("far", (255, 0, 255))):
                for p in points.get(s, []):
                    cv2.circle(view, p, 6, color, -1)
                if s != "zone" and len(points.get(s, [])) == 2:
                    cv2.line(view, points[s][0], points[s][1], color, 2)
            step = steps[state["step"]]
            label = EN_HELP[step] + ("  [FROZEN]" if state["frozen"] is not None else "")
            cv2.rectangle(view, (0, 0), (view.shape[1], 34), (30, 30, 30), -1)
            cv2.putText(view, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(win, view)

            key = cv2.waitKeyEx(30)
            if key == ESC:
                print("ยกเลิก ไม่ได้บันทึก")
                return None, None
            if key == SPACE:
                state["frozen"] = None if state["frozen"] is not None else frame
            elif key in UNDO and points[step]:
                points[step].pop()
            elif key in ENTER:
                ok = len(points[step]) >= 3 if step == "zone" else len(points[step]) == 2
                if not ok:
                    print("ยังคลิกไม่ครบ" + (" (โซนต้องมีอย่างน้อย 3 จุด)" if step == "zone" else " (ต้องคลิกศีรษะและเท้า 2 จุด)"))
                    continue
                state["step"] += 1
                if state["step"] == len(steps):
                    return points, size
                state["frozen"] = None
                print("\n" + THAI_HELP[steps[state["step"]]])
    finally:
        source.release()
        cv2.destroyAllWindows()


def main() -> int:
    ap = argparse.ArgumentParser(description="Define an Out-of-Area zone")
    ap.add_argument("--name", required=True, help="zone name, e.g. main_gate")
    ap.add_argument("--camera", default=None, help="camera id from config/core.yaml (default: first)")
    ap.add_argument("--source", default=None, help="override: video file, image, or camera index")
    ap.add_argument("--hours", default="", help="active hours, e.g. 09:00-15:00 (comma for several)")
    ap.add_argument("--any-person", action="store_true",
                    help="skip calibration; any person in the zone counts (for testing)")
    ap.add_argument("--config", default="config/core.yaml")
    ap.add_argument("--zones-file", default="config/zones.yaml")
    args = ap.parse_args()

    hours = [h.strip() for h in args.hours.split(",") if h.strip()]
    for h in hours:
        parse_hours(h)  # fail early on a typo

    cfg = load_config(resolve_path(args.config))
    cams = cfg.get("cameras") or [{"id": "cam0", "source": 0}]
    cam = next((c for c in cams if str(c["id"]) == args.camera), None) if args.camera else cams[0]
    if cam is None:
        print(f"ไม่พบกล้อง '{args.camera}' ใน {args.config}")
        return 1

    from src.cctv_core.stream.camera_source import CameraSource
    src_value = args.source if args.source is not None else cam.get("source", 0)
    # a video used for zones must be mirrored exactly as run_core.py will see it
    source = CameraSource(src_value, cam.get("width"), cam.get("height"), cam.get("flip"))

    steps = ["zone"] if args.any_person else ["zone", "near", "far"]
    points, size = run_ui(source, steps)
    if points is None:
        return 1
    w, h = size
    try:
        zone = build_zone(args.name, points["zone"], points.get("near"), points.get("far"),
                          w, h, hours, args.any_person)
    except ValueError as exc:
        print(f"ไม่ได้บันทึก: {exc}")
        return 1
    path = resolve_path(args.zones_file)
    save_zone(path, str(cam["id"]), zone)
    print(f"\nบันทึกโซน '{zone.name}' ของกล้อง {cam['id']} ลง {args.zones_file} แล้ว")
    if zone.calibration:
        c = zone.calibration
        print(f"ปรับเทียบ: ใกล้ สูง {c.near_height:.2f} ของภาพ, ไกล สูง {c.far_height:.2f} ของภาพ")
    print("รันระบบได้ด้วย: python run_core.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
