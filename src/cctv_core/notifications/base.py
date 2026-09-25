"""Notification channel interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import DeliveryRecord, Event


class NotificationChannel(ABC):
    name: str = "unnamed"

    @abstractmethod
    def send(self, event: Event) -> DeliveryRecord:
        """Deliver one event. Must not raise for ordinary delivery failures."""
