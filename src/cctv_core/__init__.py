"""CCTV Core: modular pipeline  Frame -> Detector -> EventPolicy -> EventManager -> Notification.

Detectors only report what they see. They never send notifications, write
files, or apply cooldowns - those are the Event Manager's job.
"""
