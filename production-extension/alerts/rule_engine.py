"""
alerts/rule_engine.py

Configurable rule engine for flagging dangerous or noteworthy events.

Rules are defined as simple dicts (or loaded from YAML) — no code changes
needed to add new alert types. Each rule specifies which attribute
combinations on which road user types should trigger an alert.

Example rule (YAML):
    - id: distracted_ped_low_light
      name: Distracted pedestrian in low light
      severity: high
      conditions:
        road_user_type: pedestrian
        attributes:
          attention: distracted
          lighting: low_light
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger


SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

DEFAULT_RULES = [
    {
        "id": "distracted_ped_low_light",
        "name": "Distracted pedestrian in low light",
        "severity": "high",
        "conditions": {
            "road_user_type": "pedestrian",
            "attributes": {"attention": "distracted", "lighting": "low_light"},
        },
    },
    {
        "id": "phone_use_crossing",
        "name": "Pedestrian using phone",
        "severity": "high",
        "conditions": {
            "road_user_type": "pedestrian",
            "attributes": {"attention": "phone_use"},
        },
    },
    {
        "id": "heavily_occluded_crossing",
        "name": "Heavily occluded road user",
        "severity": "medium",
        "conditions": {
            "road_user_type": "all",
            "attributes": {"occlusion": "heavy"},
        },
    },
    {
        "id": "cyclist_low_light",
        "name": "Cyclist in low light",
        "severity": "medium",
        "conditions": {
            "road_user_type": "cyclist",
            "attributes": {"lighting": "low_light"},
        },
    },
    {
        "id": "group_crossing",
        "name": "Group of pedestrians crossing",
        "severity": "low",
        "conditions": {
            "road_user_type": "pedestrian",
            "attributes": {"group": "group", "mobility": "moving"},
        },
    },
    {
        "id": "slow_vehicle_backlit",
        "name": "Slow-moving vehicle with poor visibility",
        "severity": "medium",
        "conditions": {
            "road_user_type": "vehicle",
            "attributes": {"mobility": "slow_moving", "lighting": "backlit"},
        },
    },
]


@dataclass
class Alert:
    alert_id: str
    rule_id: str
    rule_name: str
    severity: str
    track_id: int
    road_user_type: str
    frame_idx: int
    timestamp_ms: float
    attributes: Dict[str, str]
    bbox: tuple
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "severity": self.severity,
            "track_id": self.track_id,
            "road_user_type": self.road_user_type,
            "frame_idx": self.frame_idx,
            "timestamp_ms": self.timestamp_ms,
            "timestamp_readable": f"{self.timestamp_ms / 1000:.2f}s",
            "attributes": self.attributes,
            "bbox": list(self.bbox),
            "created_at": self.created_at,
        }


class RuleEngine:
    """
    Evaluates attribute predictions against alert rules.

    Call evaluate() on each classified track each frame.
    Deduplicates alerts per (track_id, rule_id) so the same event
    is not fired on every frame for the same object.
    """

    def __init__(self, rules: Optional[List[Dict]] = None, cooldown_frames: int = 30):
        self.rules = rules or DEFAULT_RULES
        self.cooldown_frames = cooldown_frames
        # (track_id, rule_id) -> last frame the alert fired
        self._last_fired: Dict[tuple, int] = {}
        self.alert_log: List[Alert] = []

    def evaluate(
        self,
        track_id: int,
        road_user_type: str,
        attributes: Dict[str, str],
        frame_idx: int,
        timestamp_ms: float,
        bbox: tuple,
    ) -> List[Alert]:
        new_alerts = []

        for rule in self.rules:
            cond = rule["conditions"]
            rule_user_type = cond.get("road_user_type", "all")

            # check road user type match
            if rule_user_type != "all" and rule_user_type != road_user_type:
                continue

            # check all attribute conditions
            required_attrs = cond.get("attributes", {})
            if not all(attributes.get(k) == v for k, v in required_attrs.items()):
                continue

            # check cooldown
            cooldown_key = (track_id, rule["id"])
            last_frame = self._last_fired.get(cooldown_key, -self.cooldown_frames)
            if frame_idx - last_frame < self.cooldown_frames:
                continue

            alert = Alert(
                alert_id=str(uuid.uuid4()),
                rule_id=rule["id"],
                rule_name=rule["name"],
                severity=rule["severity"],
                track_id=track_id,
                road_user_type=road_user_type,
                frame_idx=frame_idx,
                timestamp_ms=timestamp_ms,
                attributes=dict(attributes),
                bbox=bbox,
            )
            self._last_fired[cooldown_key] = frame_idx
            self.alert_log.append(alert)
            new_alerts.append(alert)
            logger.info(
                f"ALERT [{alert.severity.upper()}] {alert.rule_name} "
                f"| track={track_id} frame={frame_idx}"
            )

        return new_alerts

    def get_alerts(
        self,
        severity_filter: Optional[str] = None,
        road_user_filter: Optional[str] = None,
    ) -> List[Dict]:
        alerts = self.alert_log
        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]
        if road_user_filter:
            alerts = [a for a in alerts if a.road_user_type == road_user_filter]
        return [a.to_dict() for a in alerts]

    def clear_stale_cooldowns(self, current_frame: int):
        """Prune old cooldown entries to prevent memory growth on long streams."""
        stale = [
            k for k, f in self._last_fired.items()
            if current_frame - f > self.cooldown_frames * 10
        ]
        for k in stale:
            del self._last_fired[k]
