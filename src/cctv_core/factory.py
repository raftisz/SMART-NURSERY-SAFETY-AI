"""Build the event/notification side of the Core from config/core.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils.config import resolve_path

from .events.manager import EventManager, EventTypeConfig
from .events.snapshot import SnapshotWriter
from .events.store import EventStore
from .notifications.console_channel import ConsoleChannel
from .notifications.router import NotificationRouter, Route

DEFAULT_CORE_CONFIG = "config/core.yaml"


@dataclass
class EventSystem:
    manager: EventManager
    router: NotificationRouter
    store: EventStore
    camera_names: dict[str, str]

    def shutdown(self, now: float) -> None:
        self.manager.close_all(now)
        self.router.close()
        self.store.close()


def camera_names_from(cfg: dict[str, Any]) -> dict[str, str]:
    return {str(c["id"]): str(c.get("name", c["id"])) for c in cfg.get("cameras") or []}


def build_channels(cfg: dict[str, Any], camera_names: dict[str, str]) -> dict[str, Any]:
    ch_cfg = (cfg.get("notifications") or {}).get("channels") or {}
    channels: dict[str, Any] = {}
    if (ch_cfg.get("console") or {}).get("enabled", True):
        channels["console"] = ConsoleChannel(camera_names)
    if (ch_cfg.get("line") or {}).get("enabled", False):
        from .notifications.line_channel import LineChannel

        line = LineChannel(camera_names)
        if not line.configured:
            print("[NOTIFY] LINE enabled but .env credentials are missing - LINE sends will fail")
        channels["line"] = line
    return channels


def build_event_system(
    cfg: dict[str, Any],
    database: str | None = None,
    async_notifications: bool = True,
    snapshots_dir: str | None = None,
) -> EventSystem:
    """`database` / `snapshots_dir` override config storage (tests use ':memory:' and a temp dir)."""
    ev_cfg = cfg.get("events") or {}
    default = EventTypeConfig.from_dict(ev_cfg.get("default") or {})
    types = {
        name: EventTypeConfig.from_dict(data or {}, base=default)
        for name, data in (ev_cfg.get("types") or {}).items()
    }

    storage = cfg.get("storage") or {}
    db_path = database or str(resolve_path(storage.get("database", "data/events.db")))
    store = EventStore(db_path)
    snapshots = SnapshotWriter(snapshots_dir or resolve_path(storage.get("snapshots_dir", "data/snapshots")))

    camera_names = camera_names_from(cfg)
    channels = build_channels(cfg, camera_names)
    routes = [Route.from_dict(r) for r in (cfg.get("notifications") or {}).get("routes") or []]
    router = NotificationRouter(channels, routes, store=store, async_mode=async_notifications)

    manager = EventManager(types, default, store=store, snapshots=snapshots, dispatcher=router)
    return EventSystem(manager, router, store, camera_names)
