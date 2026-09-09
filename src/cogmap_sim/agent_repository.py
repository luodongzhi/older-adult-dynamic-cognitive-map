from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from .models import AgentProfile


class AgentRepository:
    """Filesystem repository for reusable trained or prompt-configured agents."""

    def __init__(self, directory: str | Path = "agents"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[AgentProfile]:
        return [self._read(path) for path in sorted(self.directory.glob("*.json"))]

    def get(self, agent_id: str) -> AgentProfile:
        path = self._path(agent_id)
        if not path.exists():
            raise KeyError(f"Agent not found: {agent_id}")
        return self._read(path)

    def add(self, profile: AgentProfile | dict[str, Any], overwrite: bool = False) -> AgentProfile:
        if isinstance(profile, dict):
            allowed = {field.name for field in fields(AgentProfile)}
            unknown = set(profile) - allowed
            if unknown:
                raise ValueError(f"Unknown AgentProfile fields: {sorted(unknown)}")
            profile = AgentProfile(**profile)
        self._validate(profile)
        path = self._path(profile.agent_id)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Agent already exists: {profile.agent_id}; use --overwrite")
        path.write_text(json.dumps(asdict(profile), ensure_ascii=False, indent=2), encoding="utf-8")
        return profile

    def add_from_file(self, source: str | Path, overwrite: bool = False) -> AgentProfile:
        return self.add(json.loads(Path(source).read_text(encoding="utf-8")), overwrite)

    def remove(self, agent_id: str) -> None:
        path = self._path(agent_id)
        if not path.exists():
            raise KeyError(f"Agent not found: {agent_id}")
        path.unlink()

    def validate_for_graph(self, profiles: list[AgentProfile], graph: Any, minimum_age: int = 60) -> None:
        for profile in profiles:
            referenced = (
                profile.home_aoi_id,
                profile.routine_anchor_aoi_id,
                *profile.regular_meeting_aoi_ids,
            )
            missing = [aoi_id for aoi_id in referenced if aoi_id not in graph.aois]
            if missing:
                raise ValueError(f"Agent {profile.agent_id} references missing AOIs: {missing}")
            if profile.age < minimum_age:
                raise ValueError(
                    f"Agent {profile.agent_id} is age {profile.age}; this experiment requires age >= {minimum_age}."
                )

    def _path(self, agent_id: str) -> Path:
        valid = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        if not agent_id or any(character not in valid for character in agent_id):
            raise ValueError("agent_id may contain only letters, digits, '_' and '-'")
        return self.directory / f"{agent_id}.json"

    @staticmethod
    def _read(path: Path) -> AgentProfile:
        profile = AgentProfile(**json.loads(path.read_text(encoding="utf-8")))
        AgentRepository._validate(profile)
        return profile

    @staticmethod
    def _validate(profile: AgentProfile) -> None:
        probabilities = (
            "exploration_tendency", "map_search_tendency", "social_participation",
            "safety_sensitivity", "route_habit", "visual_attention",
            "social_contact_frequency", "visual_detection_skill",
            "information_sharing_propensity", "work_from_home_probability",
            "schedule_flexibility", "spontaneity", "stay_home_preference",
            "risk_aversion", "physical_cost_sensitivity",
        )
        invalid = [name for name in probabilities if not 0.0 <= float(getattr(profile, name)) <= 1.0]
        if invalid:
            raise ValueError(f"Agent {profile.agent_id} parameters must be in [0,1]: {invalid}")
        if not 1.0 <= profile.visual_field_of_view_deg <= 358.0:
            raise ValueError(f"Agent {profile.agent_id} visual_field_of_view_deg must be in [1,358]")
        if profile.visual_search_radius <= 0 or profile.visual_sample_spacing_ratio <= 0:
            raise ValueError(f"Agent {profile.agent_id} visual radius and sample spacing must be positive")
        if not 0 <= int(profile.age) <= 110:
            raise ValueError(f"Agent {profile.agent_id} age must be in [0,110]")
        if profile.chronotype not in {"morning", "intermediate", "evening"}:
            raise ValueError(
                f"Agent {profile.agent_id} chronotype must be morning, intermediate or evening"
            )
        if any(not isinstance(day, int) or day < 1 or day > 7 for day in profile.work_days):
            raise ValueError(f"Agent {profile.agent_id} work_days must contain integers in [1,7]")
        if not 0 <= profile.work_start_mean < 24 or not 0 <= profile.work_start_std <= 6:
            raise ValueError(f"Agent {profile.agent_id} work start parameters are invalid")
        if not 0 < profile.work_duration_mean <= 24 or not 0 <= profile.work_duration_std <= 12:
            raise ValueError(f"Agent {profile.agent_id} work duration parameters are invalid")
        if profile.years_residence < 0 or profile.walking_speed_m_per_min <= 0 or profile.activity_radius_m <= 0:
            raise ValueError(f"Agent {profile.agent_id} residence and mobility parameters must be positive")
        if not 0 <= profile.meeting_frequency_weekly <= 21:
            raise ValueError(f"Agent {profile.agent_id} meeting_frequency_weekly must be in [0,21]")
        if not 0 <= profile.routine_anchor_frequency_weekly <= 21:
            raise ValueError(f"Agent {profile.agent_id} routine_anchor_frequency_weekly must be in [0,21]")
        if not 0 <= profile.routine_anchor_start_mean < 24 or not 0 <= profile.routine_anchor_start_std <= 6:
            raise ValueError(f"Agent {profile.agent_id} routine anchor time parameters are invalid")
        if (
            len(profile.routine_anchor_duration_minutes) != 2
            or float(profile.routine_anchor_duration_minutes[0]) <= 0
            or float(profile.routine_anchor_duration_minutes[1]) < float(profile.routine_anchor_duration_minutes[0])
        ):
            raise ValueError(f"Agent {profile.agent_id} routine_anchor_duration_minutes is invalid")
        if not 0 <= profile.meeting_start_mean < 24 or not 0 <= profile.meeting_start_std <= 6:
            raise ValueError(f"Agent {profile.agent_id} meeting time parameters are invalid")
        if (
            len(profile.meeting_duration_minutes) != 2
            or float(profile.meeting_duration_minutes[0]) <= 0
            or float(profile.meeting_duration_minutes[1]) < float(profile.meeting_duration_minutes[0])
        ):
            raise ValueError(f"Agent {profile.agent_id} meeting_duration_minutes is invalid")
        supported_activities = {"CAFE", "PARK", "SHOP"}
        if set(profile.activity_frequency_weekly) - supported_activities:
            raise ValueError(
                f"Agent {profile.agent_id} activity_frequency_weekly supports only {sorted(supported_activities)}"
            )
        if any(float(value) < 0 or float(value) > 21 for value in profile.activity_frequency_weekly.values()):
            raise ValueError(f"Agent {profile.agent_id} weekly activity frequencies must be in [0,21]")
        if set(profile.activity_duration_minutes) - supported_activities:
            raise ValueError(
                f"Agent {profile.agent_id} activity_duration_minutes supports only {sorted(supported_activities)}"
            )
        invalid_durations = []
        for activity, window in profile.activity_duration_minutes.items():
            if (
                not isinstance(window, (list, tuple)) or len(window) != 2
                or float(window[0]) <= 0 or float(window[1]) < float(window[0])
                or float(window[1]) > 1440
            ):
                invalid_durations.append(activity)
        if invalid_durations:
            raise ValueError(
                f"Agent {profile.agent_id} activity duration windows are invalid: {invalid_durations}"
            )
        invalid_needs = [
            key for key, value in profile.initial_needs.items()
            if not 0.0 <= float(value) <= 1.0
        ]
        invalid_decay = [
            key for key, value in profile.need_decay_per_hour.items()
            if not 0.0 <= float(value) <= 1.0
        ]
        if invalid_needs or invalid_decay:
            raise ValueError(
                f"Agent {profile.agent_id} need values must be in [0,1]: "
                f"needs={invalid_needs}, decay={invalid_decay}"
            )
