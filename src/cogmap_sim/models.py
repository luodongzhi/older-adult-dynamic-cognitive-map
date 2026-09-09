from __future__ import annotations

from dataclasses import asdict, dataclass, field
try:
    from enum import StrEnum
except ImportError:  # Python 3.10 compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of enum.StrEnum for Python 3.10."""

        def __str__(self) -> str:
            return self.value
from typing import Any


class EdgeStatus(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    POSSIBLY_OPEN = "POSSIBLY_OPEN"
    POSSIBLY_CLOSED = "POSSIBLY_CLOSED"
    UNKNOWN = "UNKNOWN"


class AdoptionState(StrEnum):
    UNKNOWN = "UNKNOWN"
    AWARE = "AWARE"
    TRIED = "TRIED"
    ADOPTED = "ADOPTED"
    REJECTED = "REJECTED"


class SourceChannel(StrEnum):
    VISUAL_SEARCH = "VISUAL_SEARCH"
    PASSIVE_VISUAL = "PASSIVE_VISUAL"
    ACTIVE_SEARCH = "ACTIVE_SEARCH"
    INTERPERSONAL = "INTERPERSONAL"
    PUBLIC_ANNOUNCEMENT = "PUBLIC_ANNOUNCEMENT"
    DIRECT_EXPERIENCE = "DIRECT_EXPERIENCE"
    ROUTE_FAILURE = "ROUTE_FAILURE"


@dataclass(slots=True)
class ScenarioConfig:
    scenario_id: str = "demo_central_link"
    experiment_id: str = "demo"
    start_day: int = 0
    end_day: int = 30
    intervention_day: int = 0
    random_seed: int = 42
    model_type: str = "dynamic_cognitive"
    agent_count: int = 3
    snapshot_interval: int = 5
    parameter_set_id: str = "default_v1"
    behavior_driver: str = "rule_based"
    llm_model: str = "doubao-seed-2-1-pro-260628"
    paired_seed: int | None = None
    warmup_days: int = 0
    directional_vision_enabled: bool = True
    direct_experience_enabled: bool = True
    face_to_face_enabled: bool = True
    copresence_min_overlap_minutes: float = 8.0
    max_reports_per_conversation: int = 2
    minimum_agent_age: int = 60
    export_research_csv: bool = True
    visual_sample_spacing_ratio: float = 0.18
    code_version: str = "3.0.0-binary-cognition"

    @property
    def canonical_model_type(self) -> str:
        aliases = {
            "m": "dynamic_cognitive",
            "cognitive": "dynamic_cognitive",
            "dynamic_cognitive": "dynamic_cognitive",
            "b0": "objective_accessibility",
            "accessibility": "objective_accessibility",
            "objective_accessibility": "objective_accessibility",
            "b1": "omniscient",
            "omniscient": "omniscient",
            "b2": "static_partial",
            "static_partial": "static_partial",
            "no_change": "no_change",
        }
        key = str(self.model_type).strip().lower()
        if key not in aliases:
            raise ValueError(f"Unknown model type: {self.model_type}")
        return aliases[key]


@dataclass(slots=True)
class Node:
    node_id: str
    x: float
    y: float


@dataclass(slots=True)
class Edge:
    edge_id: str
    from_node: str
    to_node: str
    length: float
    base_travel_time: float
    status: EdgeStatus = EdgeStatus.OPEN
    visibility: float = 0.65
    safety: float = 0.75
    comfort: float = 0.7
    is_intervention: bool = False
    geometry: list[tuple[float, float]] = field(default_factory=list)
    name: str = ""
    road_class: str = ""


@dataclass(slots=True)
class AOI:
    aoi_id: str
    name: str
    node_id: str
    function: str
    attractiveness: float
    safety: float = 0.75
    comfort: float = 0.7
    capacity: int = 100
    is_intervention: bool = False
    geometry: list[tuple[float, float]] = field(default_factory=list)
    feature_kind: str = "AOI"


@dataclass(slots=True)
class Entrance:
    entrance_id: str
    aoi_id: str
    node_id: str
    x: float
    y: float
    status: EdgeStatus = EdgeStatus.OPEN
    conspicuity: float = 0.7
    is_intervention: bool = False


@dataclass(slots=True)
class Occluder:
    object_id: str
    geometry: list[tuple[float, float]]
    opacity: float = 1.0
    object_type: str = "BUILDING"


@dataclass(slots=True)
class Signage:
    sign_id: str
    target_id: str
    x: float
    y: float
    facing_deg: float = 0.0
    range_m: float = 60.0
    credibility: float = 0.8
    active: bool = True


@dataclass(slots=True)
class InterventionEvent:
    intervention_id: str
    effective_day: int
    operation_type: str
    target_type: str
    target_id: str
    before_value: str
    after_value: str
    visibility_level: float = 0.7
    announcement_level: float = 0.25
    visual_salience: float | None = None
    official_announcement: float | None = None
    signage_support: float = 0.0

    @property
    def effective_visual_salience(self) -> float:
        value = self.visibility_level if self.visual_salience is None else self.visual_salience
        return max(0.0, min(1.0, float(value)))

    @property
    def effective_announcement(self) -> float:
        value = self.announcement_level if self.official_announcement is None else self.official_announcement
        return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class AgentProfile:
    agent_id: str
    group_id: str
    home_aoi_id: str
    routine_anchor_aoi_id: str
    exploration_tendency: float
    map_search_tendency: float
    social_participation: float
    safety_sensitivity: float
    route_habit: float
    visual_attention: float
    social_contact_frequency: float = 0.5
    visual_search_radius: float = 1.25
    visual_detection_skill: float = 0.7
    llm_model: str = "doubao-seed-2-1-pro-260628"
    system_prompt: str = "Act as a boundedly rational urban resident."
    training_label: str = "prompt-configured"
    visual_field_of_view_deg: float = 100.0
    visual_sample_spacing_ratio: float = 0.18
    information_sharing_propensity: float = 0.55
    years_residence: float = 15.0
    risk_aversion: float = 0.6
    walking_speed_m_per_min: float = 65.0
    activity_radius_m: float = 900.0
    physical_cost_sensitivity: float = 0.6
    mobility_aid: str = "none"
    stable_contact_ids: list[str] = field(default_factory=list)
    regular_meeting_aoi_ids: list[str] = field(default_factory=list)
    routine_anchor_frequency_weekly: float = 5.0
    routine_anchor_start_mean: float = 9.5
    routine_anchor_start_std: float = 0.45
    routine_anchor_duration_minutes: list[float] = field(default_factory=lambda: [35.0, 90.0])
    meeting_frequency_weekly: float = 4.0
    meeting_start_mean: float = 15.0
    meeting_start_std: float = 0.35
    meeting_duration_minutes: list[float] = field(default_factory=lambda: [45.0, 90.0])
    # AgentSociety/CitySim-inspired schedule configuration.  These values describe
    # behavioral priors; the LLM may propose intentions but Python enforces time,
    # map-knowledge and destination constraints.
    profile_version: str = "older-adult-information-arrival-v2"
    age: int = 70
    occupation: str = "retired"
    employment_status: str = "retired"
    household_role: str = "older_household"
    chronotype: str = "morning"
    work_days: list[int] = field(default_factory=list)
    work_start_mean: float = 8.0
    work_start_std: float = 0.45
    work_duration_mean: float = 8.0
    work_duration_std: float = 0.65
    work_from_home_probability: float = 0.0
    schedule_flexibility: float = 0.5
    spontaneity: float = 0.5
    stay_home_preference: float = 0.0
    activity_frequency_weekly: dict[str, float] = field(default_factory=lambda: {
        "CAFE": 7.0, "PARK": 7.0, "SHOP": 7.0,
    })
    activity_duration_minutes: dict[str, list[float]] = field(default_factory=lambda: {
        "CAFE": [30.0, 75.0], "PARK": [35.0, 120.0], "SHOP": [25.0, 90.0],
    })
    initial_needs: dict[str, float] = field(default_factory=lambda: {
        "energy": 0.75, "social": 0.55, "leisure": 0.5, "errand": 0.35,
    })
    need_decay_per_hour: dict[str, float] = field(default_factory=lambda: {
        "energy": 0.04, "social": 0.025, "leisure": 0.03, "errand": 0.015,
    })
    long_term_goals: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentDailyState:
    day: int
    agent_id: str
    start_location: str
    end_location: str = ""
    episode_count: int = 0
    total_travel_time: float = 0.0
    total_distance: float = 0.0
    failed_route_count: int = 0
    reroute_count: int = 0
    new_object_count: int = 0
    daily_satisfaction: float = 0.0


@dataclass(slots=True)
class EdgeBelief:
    edge_id: str
    believed_status: EdgeStatus = EdgeStatus.OPEN
    perceived_travel_time: float = 1.0
    perceived_safety: float = 0.7
    information_source: str = "BASELINE_KNOWLEDGE"
    first_known_day: int = 0
    last_update_day: int = 0
    adoption_state: AdoptionState = AdoptionState.AWARE
    use_count: int = 0


@dataclass(slots=True)
class AOIBelief:
    aoi_id: str
    believed_function: str
    perceived_accessibility: float = 0.7
    perceived_safety: float = 0.7
    perceived_comfort: float = 0.7
    perceived_attractiveness: float = 0.6
    information_source: str = "BASELINE_KNOWLEDGE"
    first_known_day: int = 0
    last_update_day: int = 0
    visit_count: int = 0
    adoption_state: AdoptionState = AdoptionState.AWARE


@dataclass(slots=True)
class EntranceBelief:
    entrance_id: str
    believed_status: EdgeStatus = EdgeStatus.OPEN
    conspicuity: float = 0.6
    information_source: str = "BASELINE_KNOWLEDGE"
    first_known_day: int = 0
    last_update_day: int = 0


@dataclass(slots=True)
class CognitiveBeliefGraph:
    agent_id: str
    edges: dict[str, EdgeBelief] = field(default_factory=dict)
    aois: dict[str, AOIBelief] = field(default_factory=dict)
    entrances: dict[str, EntranceBelief] = field(default_factory=dict)


@dataclass(slots=True)
class ActivityEpisode:
    day: int
    agent_id: str
    episode_id: str
    sequence: int
    activity_type: str
    origin_node: str
    planned_start: float
    mandatory_level: str
    expected_duration: float
    travel_budget: float
    fixed_destination_aoi: str | None = None


@dataclass(slots=True)
class DestinationDecision:
    day: int
    agent_id: str
    episode_id: str
    candidate_aoi_ids: list[str]
    selected_aoi_id: str | None
    expected_utility: float
    choice_probability: float
    decision_factors: dict[str, float]
    status: str


@dataclass(slots=True)
class TrajectoryStep:
    edge_id: str
    sequence: int
    enter_time: float
    exit_time: float
    status: str = "COMPLETED"
    reroute_id: str | None = None


@dataclass(slots=True)
class ExecutionResult:
    success: bool
    steps: list[TrajectoryStep]
    failed_edge_id: str | None
    rerouted: bool
    arrival_node: str
    travel_time: float
    distance: float
    activity_success: bool
    satisfaction: float
    failure_reason: str | None = None


@dataclass(slots=True)
class ObservationEvent:
    event_id: str
    day: int
    event_time: float
    agent_id: str
    episode_id: str
    object_id: str
    object_type: str
    source_channel: SourceChannel
    observed_attribute: str
    observed_value: str
    source_reliability: float
    salience: float
    distance_to_object: float = 0.0
    observer_x: float | None = None
    observer_y: float | None = None
    heading_deg: float | None = None
    relative_angle_deg: float | None = None
    line_of_sight: bool | None = None
    detection_probability: float | None = None
    detection_draw: float | None = None
    source_event_id: str | None = None
    independence_group: str | None = None
    source_agent_id: str | None = None
    copresence_event_id: str | None = None
    first_hand_source: bool | None = None


@dataclass(slots=True)
class InformationItem:
    information_id: str
    day: int
    agent_id: str
    object_id: str
    object_type: str
    attribute_name: str
    observed_value: str
    source_channel: SourceChannel
    source_event_id: str
    reliability: float
    salience: float
    independence_group: str = ""
    source_agent_id: str | None = None
    copresence_event_id: str | None = None
    first_hand_source: bool | None = None


@dataclass(slots=True)
class CognitiveUpdateRecord:
    update_id: str
    day: int
    agent_id: str
    object_id: str
    object_type: str
    attribute_name: str
    old_value: str
    new_value: str
    update_reason: str
    source_event_id: str
    event_time: float
    source_agent_id: str | None = None
    first_hand_source: bool | None = None


@dataclass(slots=True)
class CognitiveDetectionRecord:
    detection_id: str
    day: int
    event_time: float
    agent_id: str
    object_id: str
    object_type: str
    prior_value: str
    observed_value: str
    source_channel: SourceChannel
    source_event_id: str
    first_detection: bool


@dataclass(slots=True)
class ActivityStop:
    day: int
    agent_id: str
    episode_id: str
    place_id: str
    activity_type: str
    start_time: float
    end_time: float


@dataclass(slots=True)
class CopresenceEvent:
    event_id: str
    day: int
    agent_a: str
    agent_b: str
    place_id: str
    overlap_start: float
    overlap_end: float
    overlap_minutes: float
    relationship_strength: float
    conversation_probability: float
    conversation_draw: float
    conversation_flag: bool


@dataclass(slots=True)
class InterpersonalExchange:
    exchange_id: str
    day: int
    copresence_event_id: str
    speaker_agent_id: str
    listener_agent_id: str
    place_id: str
    report_id: str
    object_id: str
    object_type: str
    observed_attribute: str
    observed_value: str
    source_event_id: str
    independence_group: str
    first_hand_source: bool
    delivered: bool = True


@dataclass(slots=True)
class DayResult:
    day: int
    agent_states: list[AgentDailyState] = field(default_factory=list)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[DestinationDecision] = field(default_factory=list)
    planned: list[dict[str, Any]] = field(default_factory=list)
    executed: list[dict[str, Any]] = field(default_factory=list)
    observations: list[ObservationEvent] = field(default_factory=list)
    temporary_memories: list[dict[str, Any]] = field(default_factory=list)
    updates: list[CognitiveUpdateRecord] = field(default_factory=list)
    llm_decisions: list[dict[str, Any]] = field(default_factory=list)
    activity_stops: list[ActivityStop] = field(default_factory=list)
    copresence_events: list[CopresenceEvent] = field(default_factory=list)
    interpersonal_exchanges: list[InterpersonalExchange] = field(default_factory=list)
    detections: list[CognitiveDetectionRecord] = field(default_factory=list)
    information_items: list[InformationItem] = field(default_factory=list)
    perception_diagnostics: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class DailyAgentPlan:
    flexible_activity_type: str
    destination_preferences: dict[str, list[str]]
    reasoning: str = ""


def to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
