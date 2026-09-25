"""
LINE alerting for hazardous-object detections.

Uses the LINE **Messaging API** push-message endpoint. LINE Notify was shut
down in 2025 and is deliberately not used here.

Credentials are read from a .env file at the project root and are never
written to logs or printed - only a masked form is ever shown.

Design notes:
  * Sending happens on a daemon thread, so a slow or unreachable LINE endpoint
    cannot stall the video loop.
  * Every network failure is caught and logged. A detection loop must never die
    because a chat service is down.
  * A cooldown suppresses repeat alerts, since a 14 FPS loop would otherwise
    push hundreds of messages a minute.
"""

import os
import threading
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

PUSH_URL = "https://api.line.me/v2/bot/message/push"

# Classes that justify an alert. `person` alone never does.
DEFAULT_HAZARD_CLASSES = ("scissors", "knife")

# Shown on the "📹 กล้อง:" line. Overridden by alerts.camera_name in config.yaml.
DEFAULT_CAMERA_NAME = "Notebook Camera"


def _mask(secret: str) -> str:
    """Render a credential safe to print."""
    if not secret:
        return "(not set)"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]} (len {len(secret)})"


class LineNotifier:
    """Push hazardous-object alerts to LINE, with cooldown and failure tolerance."""

    def __init__(
        self,
        cooldown_seconds: float = 30.0,
        hazard_classes=DEFAULT_HAZARD_CLASSES,
        camera_name: str = DEFAULT_CAMERA_NAME,
        timeout: float = 5.0,
        env_path: Path | None = None,
    ):
        self.camera_name = camera_name
        load_dotenv(env_path or DEFAULT_ENV_PATH, override=False)
        self.token = (os.getenv("LINE_CHANNEL_ACCESS_TOKEN") or "").strip()
        self.user_id = (os.getenv("LINE_USER_ID") or "").strip()
        # LINE can only attach an image by public HTTPS URL - it cannot accept a
        # local file. Set this only if the saved frames are served somewhere.
        self.image_base_url = (os.getenv("LINE_IMAGE_BASE_URL") or "").strip().rstrip("/")

        self.cooldown_seconds = float(cooldown_seconds)
        self.hazard_classes = {c.lower() for c in hazard_classes}
        self.timeout = timeout

        self._last_sent_at: float | None = None
        self._lock = threading.Lock()

    # -- configuration -----------------------------------------------------

    @property
    def configured(self) -> bool:
        return bool(self.token and self.user_id)

    def describe(self) -> str:
        lines = [
            "LINE alerts  : ENABLED",
            f"  token      : {_mask(self.token)}",
            f"  user id    : {_mask(self.user_id)}",
            f"  cooldown   : {self.cooldown_seconds:.0f}s",
            f"  alert on   : {', '.join(sorted(self.hazard_classes))} (never person alone)",
        ]
        if self.image_base_url:
            lines.append(f"  image base : {self.image_base_url}")
        if not self.configured:
            missing = []
            if not self.token:
                missing.append("LINE_CHANNEL_ACCESS_TOKEN")
            if not self.user_id:
                missing.append("LINE_USER_ID")
            lines.append(f"  [WARN] missing in .env: {', '.join(missing)}")
            lines.append("  [WARN] alerts will be SKIPPED. Detection continues normally.")
        return "\n".join(lines)

    # -- cooldown ----------------------------------------------------------

    def cooldown_remaining(self, now: float | None = None) -> float:
        """Seconds left before another alert may be sent. 0 means ready."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._last_sent_at is None:
                return 0.0
            remaining = self.cooldown_seconds - (now - self._last_sent_at)
        return max(0.0, remaining)

    def _claim_send_slot(self, now: float | None = None) -> bool:
        """Atomically take the cooldown slot. False if still cooling down.

        Claimed *before* the request is made, so a burst of frames cannot each
        start their own thread while the first is still in flight.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._last_sent_at is not None and \
                    (now - self._last_sent_at) < self.cooldown_seconds:
                return False
            self._last_sent_at = now
            return True

    # -- message building --------------------------------------------------

    def build_message(self, detections, extra: str = "") -> str:
        """Alert text, matching the format used by the other system on this OA.

        The format carries a single class/confidence pair, so when a frame holds
        more than one hazard the highest-confidence one is reported.
        The date is DD/MM/YYYY, not ISO, to match that system.
        """
        top = max(detections, key=lambda d: d.confidence)
        stamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        text = (
            "🚨 AI CCTV ALERT 🚨\n"
            "\n"
            "⚠️ ตรวจพบวัตถุอันตราย\n"
            "\n"
            f"🎯 ประเภท: {top.class_name}\n"
            f"📊 Confidence: {top.confidence:.2f}\n"
            f"🕐 เวลา: {stamp}\n"
            f"📹 กล้อง: {self.camera_name}\n"
            "📸 บันทึกภาพเหตุการณ์แล้ว"
        )
        return f"{text}\n{extra}" if extra else text

    # -- sending -----------------------------------------------------------

    def send_alert(self, message: str, image_path=None) -> bool:
        """POST one push message. Blocking. Returns True on success.

        Never raises: every failure is reported through the return value and a
        printed reason, so a caller in a video loop cannot be interrupted.
        """
        if not self.configured:
            print("[LINE] error credentials missing (LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID)")
            return False

        messages = [{"type": "text", "text": message[:4900]}]

        # LINE requires a publicly reachable HTTPS URL for an image; a local
        # path cannot be uploaded through this endpoint.
        if image_path:
            name = Path(image_path).name
            if self.image_base_url:
                url = f"{self.image_base_url}/{name}"
                messages.append({
                    "type": "image",
                    "originalContentUrl": url,
                    "previewImageUrl": url,
                })
            else:
                messages[0]["text"] += f"\nFrame saved locally: {name}"

        try:
            response = requests.post(
                PUSH_URL,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
                json={"to": self.user_id, "messages": messages},
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as exc:
            print(f"[LINE] error network: {type(exc).__name__}: {exc}")
            return False
        except Exception as exc:  # never let the video loop die
            print(f"[LINE] error unexpected: {type(exc).__name__}: {exc}")
            return False

        if response.status_code == 200:
            print("[LINE] sent")
            return True

        reason = response.text.strip()[:200] if response.text else "(no body)"
        if response.status_code == 401:
            reason = f"401 unauthorized - check LINE_CHANNEL_ACCESS_TOKEN. {reason}"
        elif response.status_code == 403:
            reason = f"403 forbidden - is the user a friend of the channel? {reason}"
        elif response.status_code == 429:
            reason = f"429 rate limited by LINE. {reason}"
        else:
            reason = f"HTTP {response.status_code}. {reason}"
        print(f"[LINE] error {reason}")
        return False

    def send_alert_async(self, message: str, image_path=None) -> threading.Thread:
        """Send on a daemon thread so the video loop never blocks."""
        thread = threading.Thread(
            target=self.send_alert,
            args=(message,),
            kwargs={"image_path": image_path},
            daemon=True,
            name="line-alert",
        )
        thread.start()
        return thread

    # -- loop entry point --------------------------------------------------

    def maybe_alert(self, detections, image_path=None, extra: str = "") -> bool:
        """Alert if a hazardous object is present and the cooldown has expired.

        Returns True if a send was started. Person-only frames never alert.
        """
        hazards = [d for d in detections if d.class_name.lower() in self.hazard_classes]
        if not hazards:
            return False

        if not self.configured:
            print("[LINE] error credentials missing - alert skipped")
            return False

        if not self._claim_send_slot():
            print(f"[LINE] cooldown {self.cooldown_remaining():.0f}s remaining "
                  f"({len(hazards)} hazard detection(s) suppressed)")
            return False

        self.send_alert_async(self.build_message(hazards, extra=extra),
                              image_path=image_path)
        return True
