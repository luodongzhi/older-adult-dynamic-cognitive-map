from __future__ import annotations

import math
import random
import uuid
from typing import Any

from .information import InformationLedger
from .graph import UrbanGraph
from .models import (
    AOIBelief,
    ActivityEpisode,
    ActivityStop,
    AdoptionState,
    AgentDailyState,
    AgentProfile,
    CognitiveBeliefGraph,
    CognitiveDetectionRecord,
    CognitiveUpdateRecord,
    DestinationDecision,
    EdgeBelief,
    EdgeStatus,
    EntranceBelief,
    InformationItem,
    ExecutionResult,
    InterventionEvent,
    ObservationEvent,
    SourceChannel,
    TrajectoryStep,
)
from .spatial import DirectionalVisionEngine


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class M00ExperimentController:
    """M00: owns the t0 -> tT clock and deterministic seed derivation."""

    def __init__(self, seed: int):
        self.seed = seed

    def random_for(self, day: int, agent_id: str = "") -> random.Random:
        stable = sum((i + 1) * ord(ch) for i, ch in enumerate(agent_id))
        return random.Random(self.seed * 1_000_003 + day * 10_007 + stable)


class M01ActualEnvironmentManager:
    """M01: maintains the shared objective graph."""

    def __init__(self, graph: UrbanGraph):
        self.graph = graph

    def snapshot(self) -> UrbanGraph:
        return self.graph


class M02InterventionManager:
    """M02: applies deltas to reality only; beliefs remain untouched."""

    def apply(self, graph: UrbanGraph, events: list[InterventionEvent], day: int) -> list[str]:
        affected: list[str] = []
        salience = graph.metadata.setdefault("intervention_salience", {})
        signage_support = graph.metadata.setdefault("intervention_signage_support", {})
        intervention_ids = graph.metadata.setdefault("intervention_ids", {})
        for event in events:
            if event.effective_day != day:
                continue
            if event.operation_type == "ROAD_OPEN":
                edge = graph.edges[event.target_id]
                self._validate_before(event, str(edge.status))
                edge.status = EdgeStatus.OPEN
                edge.is_intervention = True
            elif event.operation_type == "ROAD_CLOSE":
                edge = graph.edges[event.target_id]
                self._validate_before(event, str(edge.status))
                edge.status = EdgeStatus.CLOSED
                edge.is_intervention = True
            elif event.operation_type == "AOI_FUNCTION_CHANGE":
                aoi = graph.aois[event.target_id]
                self._validate_before(event, aoi.function)
                aoi.function = event.after_value
                aoi.is_intervention = True
            elif event.operation_type in {"ENTRANCE_OPEN", "ENTRANCE_CLOSE"}:
                entrance = graph.entrances[event.target_id]
                self._validate_before(event, str(entrance.status))
                entrance.status = EdgeStatus.OPEN if event.operation_type == "ENTRANCE_OPEN" else EdgeStatus.CLOSED
                entrance.is_intervention = True
            elif event.operation_type in {"SIGNAGE_ACTIVATE", "SIGNAGE_DEACTIVATE"}:
                sign = graph.signage[event.target_id]
                self._validate_before(event, str(sign.active).upper())
                sign.active = event.operation_type == "SIGNAGE_ACTIVATE"
            else:
                raise ValueError(f"Unsupported intervention: {event.operation_type}")
            salience[event.target_id] = event.effective_visual_salience
            signage_support[event.target_id] = max(0.0, min(1.0, event.signage_support))
            intervention_ids[event.target_id] = event.intervention_id
            affected.append(event.target_id)
        return affected

    @staticmethod
    def _validate_before(event: InterventionEvent, actual_value: str) -> None:
        expected = str(event.before_value).strip().upper()
        actual = str(actual_value).strip().upper()
        if expected and expected not in {"*", "ANY", actual}:
            raise ValueError(
                f"Intervention {event.intervention_id} expected {event.target_id}={event.before_value!r}, "
                f"but the scenario copy contains {actual_value!r}. The source SHP was not modified."
            )


class M03AgentStateManager:
    """M03: initializes daily state without mutating agent profiles."""

    def start_day(self, day: int, profile: AgentProfile, graph: UrbanGraph) -> AgentDailyState:
        return AgentDailyState(day, profile.agent_id, graph.aois[profile.home_aoi_id].node_id)


class M04CognitiveMapManager:
    """M04: builds and serves sparse per-agent belief overlays."""

    def initialize(self, graph: UrbanGraph, agents: list[AgentProfile], seed: int) -> dict[str, CognitiveBeliefGraph]:
        beliefs: dict[str, CognitiveBeliefGraph] = {}
        for profile in agents:
            rng = random.Random(seed + sum(ord(c) for c in profile.agent_id))
            home_node = graph.aois[profile.home_aoi_id].node_id
            anchor_ids = {
                profile.home_aoi_id,
                profile.routine_anchor_aoi_id,
                *profile.regular_meeting_aoi_ids,
            }
            anchor_nodes = [graph.aois[aoi_id].node_id for aoi_id in anchor_ids]
            familiar_routes: set[str] = set()
            for anchor_node in anchor_nodes:
                familiar_routes.update(graph.shortest_path(home_node, anchor_node) or [])
            known_edges = set(familiar_routes)
            for edge_id in list(familiar_routes):
                edge = graph.edges[edge_id]
                for node in (edge.from_node, edge.to_node):
                    for _, nearby_id in graph.adjacency[node]:
                        if rng.random() < 0.72:
                            known_edges.add(nearby_id)
            # A closed intervention-only link may not yet exist in a resident's map.
            # This is data-driven; no demo-specific edge id is hard coded.
            known_edges -= {
                edge_id for edge_id in known_edges
                if graph.edges[edge_id].is_intervention and graph.edges[edge_id].status == EdgeStatus.CLOSED
            }
            edge_beliefs = {
                eid: EdgeBelief(
                    edge_id=eid,
                    believed_status=graph.edges[eid].status,
                    perceived_travel_time=graph.edges[eid].base_travel_time * rng.uniform(0.92, 1.12),
                    perceived_safety=graph.edges[eid].safety,
                )
                for eid in known_edges
            }
            aoi_beliefs: dict[str, AOIBelief] = {}
            for aoi in graph.aois.values():
                anchor_distance = min(
                    graph.node_distance(aoi.node_id, anchor_node)
                    for anchor_node in anchor_nodes
                )
                known = aoi.aoi_id in anchor_ids or rng.random() < max(0.18, 0.82 - anchor_distance * 0.16)
                if known:
                    aoi_beliefs[aoi.aoi_id] = AOIBelief(
                        aoi_id=aoi.aoi_id, believed_function=aoi.function,
                        perceived_safety=aoi.safety, perceived_comfort=aoi.comfort,
                        perceived_attractiveness=aoi.attractiveness,
                    )
            entrance_beliefs = {
                entrance.entrance_id: EntranceBelief(
                    entrance_id=entrance.entrance_id,
                    believed_status=entrance.status,
                    conspicuity=entrance.conspicuity,
                )
                for entrance in graph.entrances.values()
                if entrance.aoi_id in aoi_beliefs
            }
            beliefs[profile.agent_id] = CognitiveBeliefGraph(
                profile.agent_id, edge_beliefs, aoi_beliefs, entrance_beliefs
            )
        return beliefs

    def make_omniscient(self, belief: CognitiveBeliefGraph, graph: UrbanGraph, day: int) -> None:
        for edge in graph.edges.values():
            current = belief.edges.get(edge.edge_id)
            if current is None:
                belief.edges[edge.edge_id] = EdgeBelief(
                    edge_id=edge.edge_id, believed_status=edge.status,
                    perceived_travel_time=edge.base_travel_time,
                    perceived_safety=edge.safety, information_source="OMNISCIENT",
                    first_known_day=day, last_update_day=day,
                )
            else:
                current.believed_status = edge.status
                current.information_source = "OMNISCIENT"
                current.last_update_day = day
        for aoi in graph.aois.values():
            current = belief.aois.get(aoi.aoi_id)
            if current is None:
                belief.aois[aoi.aoi_id] = AOIBelief(
                    aoi_id=aoi.aoi_id, believed_function=aoi.function,
                    perceived_safety=aoi.safety, perceived_comfort=aoi.comfort,
                    perceived_attractiveness=aoi.attractiveness,
                    information_source="OMNISCIENT", first_known_day=day,
                    last_update_day=day,
                )
            else:
                current.believed_function = aoi.function
                current.information_source = "OMNISCIENT"
                current.last_update_day = day
        for entrance in graph.entrances.values():
            current_entrance = belief.entrances.get(entrance.entrance_id)
            if current_entrance is None:
                belief.entrances[entrance.entrance_id] = EntranceBelief(
                    entrance_id=entrance.entrance_id,
                    believed_status=entrance.status,
                    conspicuity=entrance.conspicuity,
                    information_source="OMNISCIENT",
                    first_known_day=day,
                    last_update_day=day,
                )
            else:
                current_entrance.believed_status = entrance.status
                current_entrance.information_source = "OMNISCIENT"
                current_entrance.last_update_day = day


class M05DailyActivityGenerator:
    """M05: creates profile-conditioned anchors and stochastic flexible demand."""

    def generate(self, day: int, profile: AgentProfile, graph: UrbanGraph, rng: random.Random, llm_plan: dict[str, Any] | None = None) -> list[ActivityEpisode]:
        home = graph.aois[profile.home_aoi_id].node_id
        day_of_week = ((day - 1) % 7) + 1
        chronotype_shift = {"morning": -0.35, "intermediate": 0.0, "evening": 0.45}.get(
            profile.chronotype, 0.0,
        )
        variability = 0.65 + 0.7 * profile.schedule_flexibility
        work_start = self._bounded_gauss(
            rng,
            profile.work_start_mean + chronotype_shift,
            profile.work_start_std * variability,
            5.5,
            13.0,
        )
        work_duration = self._bounded_gauss(
            rng,
            profile.work_duration_mean,
            profile.work_duration_std * variability,
            2.0,
            12.0,
        )
        works_today = day_of_week in profile.work_days and profile.employment_status != "retired"
        works_from_home = works_today and rng.random() < profile.work_from_home_probability
        episodes: list[ActivityEpisode] = []
        sequence = 1
        work_end = work_start + work_duration if works_today else 0.0
        if works_today:
            episodes.append(ActivityEpisode(
                day, profile.agent_id, f"D{day}_{profile.agent_id}_E{sequence}", sequence,
                "WORK", home, work_start, "FIXED", work_duration, 12.0,
                profile.home_aoi_id if works_from_home else profile.routine_anchor_aoi_id,
            ))
            sequence += 1

        last_flexible_end = max(8.0, work_start)
        routine_probability = min(
            1.0, max(0.0, profile.routine_anchor_frequency_weekly / 7.0),
        )
        if not works_today and rng.random() < routine_probability:
            routine_start = self._bounded_gauss(
                rng,
                profile.routine_anchor_start_mean + chronotype_shift,
                profile.routine_anchor_start_std * variability,
                7.0,
                14.0,
            )
            routine_duration = rng.uniform(
                float(profile.routine_anchor_duration_minutes[0]),
                float(profile.routine_anchor_duration_minutes[1]),
            ) / 60.0
            episodes.append(ActivityEpisode(
                day, profile.agent_id, f"D{day}_{profile.agent_id}_E{sequence}", sequence,
                "ROUTINE", home, routine_start, "SEMI_FIXED", routine_duration, 9.0,
                profile.routine_anchor_aoi_id,
            ))
            sequence += 1
            last_flexible_end = routine_start + routine_duration

        cafe_probability = min(1.0, max(0.0, profile.activity_frequency_weekly.get("CAFE", 0.0) / 7.0))
        if rng.random() < cafe_probability:
            cafe_duration = self._activity_duration_hours(profile, "CAFE", rng)
            if works_today:
                cafe_start = min(work_end - 0.35, work_start + max(2.8, work_duration * 0.52))
            else:
                cafe_start = self._bounded_gauss(
                    rng, 11.2 + chronotype_shift, 1.1 * variability, 8.0, 17.0,
                )
            cafe_start = max(work_start + 0.5 if works_today else 8.0, cafe_start)
            episodes.append(ActivityEpisode(
                day, profile.agent_id, f"D{day}_{profile.agent_id}_E{sequence}", sequence,
                "CAFE", "", cafe_start, "SEMI_FIXED", cafe_duration, 5.5,
            ))
            sequence += 1
            last_flexible_end = cafe_start + cafe_duration

        flexible_type = (llm_plan or {}).get("flexible_activity_type", "PARK")
        weekly_frequency = profile.activity_frequency_weekly.get(flexible_type, 0.0)
        participation_factor = (
            0.75 + 0.25 * profile.schedule_flexibility + 0.25 * profile.spontaneity
            - 0.5 * profile.stay_home_preference
        )
        flexible_probability = min(1.0, max(0.0, weekly_frequency / 7.0 * participation_factor))
        if flexible_type != "NONE" and rng.random() < flexible_probability:
            duration = self._activity_duration_hours(profile, flexible_type, rng)
            if works_today:
                flexible_start = max(work_end + rng.uniform(0.35, 1.25), last_flexible_end + 0.4)
            else:
                leisure_mean = {"morning": 11.0, "intermediate": 13.2, "evening": 15.4}.get(
                    profile.chronotype, 13.2,
                )
                flexible_start = max(
                    last_flexible_end + 0.5,
                    self._bounded_gauss(rng, leisure_mean, 1.6 * variability, 9.0, 20.5),
                )
            flexible_start = min(21.5, flexible_start)
            episodes.append(ActivityEpisode(
                day, profile.agent_id, f"D{day}_{profile.agent_id}_E{sequence}", sequence,
                flexible_type, "", flexible_start, "FLEXIBLE", duration, 7.0,
            ))
            sequence += 1
            last_flexible_end = flexible_start + duration

        meeting_probability = min(1.0, max(0.0, profile.meeting_frequency_weekly / 7.0))
        if profile.regular_meeting_aoi_ids and rng.random() < meeting_probability:
            meeting_start = self._bounded_gauss(
                rng, profile.meeting_start_mean + chronotype_shift * 0.25,
                profile.meeting_start_std, 7.0, 21.5,
            )
            meeting_duration = rng.uniform(
                float(profile.meeting_duration_minutes[0]),
                float(profile.meeting_duration_minutes[1]),
            ) / 60.0
            episodes.append(ActivityEpisode(
                day, profile.agent_id, f"D{day}_{profile.agent_id}_E{sequence}", sequence,
                "SOCIAL", "", meeting_start, "SEMI_FIXED", meeting_duration, 8.0,
                rng.choice(profile.regular_meeting_aoi_ids),
            ))
            sequence += 1
            last_flexible_end = max(last_flexible_end, meeting_start + meeting_duration)

        return_home_start = max(
            17.0 if not works_today else work_end + 0.6,
            last_flexible_end + 0.45,
            18.0 + chronotype_shift,
        )
        episodes.append(ActivityEpisode(
            day, profile.agent_id, f"D{day}_{profile.agent_id}_E{sequence}", sequence,
            "HOME", "", min(23.5, return_home_start), "FIXED", 10.0, 14.0,
            profile.home_aoi_id,
        ))
        episodes.sort(key=lambda item: (item.planned_start, item.mandatory_level != "FIXED"))
        for index, episode in enumerate(episodes, start=1):
            episode.sequence = index
        return episodes

    @staticmethod
    def _bounded_gauss(
        rng: random.Random, mean: float, standard_deviation: float,
        lower: float, upper: float,
    ) -> float:
        value = mean if standard_deviation <= 0 else rng.gauss(mean, standard_deviation)
        return max(lower, min(upper, value))

    @staticmethod
    def _activity_duration_hours(profile: AgentProfile, activity: str, rng: random.Random) -> float:
        lower, upper = profile.activity_duration_minutes.get(activity, [30.0, 90.0])
        return rng.uniform(float(lower), float(upper)) / 60.0


class M06DestinationDecisionMaker:
    """M06: chooses only among cognitively available AOIs."""

    def choose(self, episode: ActivityEpisode, profile: AgentProfile, belief: CognitiveBeliefGraph, graph: UrbanGraph, rng: random.Random, model_type: str = "cognitive", preferred_aoi_ids: list[str] | None = None) -> DestinationDecision:
        if episode.fixed_destination_aoi:
            return DestinationDecision(episode.day, episode.agent_id, episode.episode_id, [episode.fixed_destination_aoi], episode.fixed_destination_aoi, 1.0, 1.0, {"fixed": 1.0}, "SELECTED")
        candidates = [b for b in belief.aois.values() if b.believed_function == episode.activity_type]
        if not candidates:
            return DestinationDecision(episode.day, episode.agent_id, episode.episode_id, [], None, 0.0, 0.0, {}, "NO_CANDIDATE")
        utilities: list[tuple[AOIBelief, float, dict[str, float]]] = []
        for candidate in candidates:
            aoi = graph.aois[candidate.aoi_id]
            path = graph.shortest_path(
                episode.origin_node, aoi.node_id,
                allowed=lambda e: e.edge_id in belief.edges and belief.edges[e.edge_id].believed_status in (EdgeStatus.OPEN, EdgeStatus.POSSIBLY_OPEN),
                cost=lambda e: belief.edges[e.edge_id].perceived_travel_time,
            )
            if path is None:
                continue
            travel = sum(belief.edges[e].perceived_travel_time for e in path)
            safety = candidate.perceived_safety * profile.safety_sensitivity
            if model_type in {"accessibility", "objective_accessibility"}:
                # Conventional baseline: objective opportunity and travel impedance only.
                factors = {"attractiveness": candidate.perceived_attractiveness * 1.6, "travel_cost": -0.18 * travel}
            else:
                factors = {
                    "attractiveness": candidate.perceived_attractiveness * 1.6,
                    "safety": safety,
                    "travel_cost": -0.18 * travel,
                }
            utility = sum(factors.values()) + rng.gammavariate(1.2, 0.12)
            utilities.append((candidate, utility, factors))
        if not utilities:
            return DestinationDecision(episode.day, episode.agent_id, episode.episode_id, [b.aoi_id for b in candidates], None, 0.0, 0.0, {}, "NO_ROUTE")
        if preferred_aoi_ids:
            by_id = {candidate.aoi_id: (candidate, utility, factors) for candidate, utility, factors in utilities}
            for preferred_id in preferred_aoi_ids:
                if preferred_id in by_id:
                    chosen = by_id[preferred_id]
                    return DestinationDecision(episode.day, episode.agent_id, episode.episode_id, [b.aoi_id for b, _, _ in utilities], chosen[0].aoi_id, chosen[1], 1.0, {**chosen[2], "llm_selected": 1.0}, "LLM_SELECTED")
        max_u = max(u for _, u, _ in utilities)
        weights = [math.exp((u - max_u) / 0.55) for _, u, _ in utilities]
        chosen = rng.choices(utilities, weights=weights, k=1)[0]
        probability = weights[utilities.index(chosen)] / sum(weights)
        return DestinationDecision(episode.day, episode.agent_id, episode.episode_id, [b.aoi_id for b, _, _ in utilities], chosen[0].aoi_id, chosen[1], probability, chosen[2], "SELECTED")


class M07CognitiveRoutePlanner:
    """M07: routes on the sparse belief overlay, never the full graph."""

    def plan(self, origin: str, destination: str, belief: CognitiveBeliefGraph, graph: UrbanGraph) -> list[str] | None:
        return graph.shortest_path(
            origin, destination,
            allowed=lambda e: e.edge_id in belief.edges and belief.edges[e.edge_id].believed_status in (EdgeStatus.OPEN, EdgeStatus.POSSIBLY_OPEN),
            cost=lambda e: belief.edges[e.edge_id].perceived_travel_time,
        )


class M08ActualEnvironmentExecutor:
    """M08: executes planned edges against reality and performs physical rerouting."""

    def execute(
        self, planned: list[str] | None, origin: str, destination: str, start_time: float,
        graph: UrbanGraph, temporary_closed: set[str], belief: CognitiveBeliefGraph | None = None,
        profile: AgentProfile | None = None,
    ) -> ExecutionResult:
        if planned is None:
            return ExecutionResult(False, [], None, False, origin, 0.0, 0.0, False, 0.0, "NO_COGNITIVE_ROUTE")
        current, clock, distance = origin, start_time, 0.0
        speed_factor = 1.0 if profile is None else max(0.55, min(2.2, 75.0 / profile.walking_speed_m_per_min))
        steps: list[TrajectoryStep] = []
        failed_edge: str | None = None
        rerouted = False
        for edge_id in planned:
            edge = graph.edges[edge_id]
            if edge.status != EdgeStatus.OPEN:
                steps.append(TrajectoryStep(edge_id, len(steps), clock, clock, "BLOCKED"))
                failed_edge = edge_id
                temporary_closed.add(edge_id)
                rerouted = True
                reroute = graph.shortest_path(
                    current, destination,
                    allowed=lambda e: (
                        e.status == EdgeStatus.OPEN
                        and e.edge_id not in temporary_closed
                        and (
                            belief is None
                            or (
                                e.edge_id in belief.edges
                                and belief.edges[e.edge_id].believed_status in (EdgeStatus.OPEN, EdgeStatus.POSSIBLY_OPEN)
                            )
                        )
                    ),
                    cost=(lambda e: belief.edges[e.edge_id].perceived_travel_time) if belief is not None else None,
                )
                if reroute is None:
                    return ExecutionResult(False, steps, failed_edge, True, current, (clock - start_time) * 60.0, distance, False, 0.0, "ROUTE_BLOCKED")
                reroute_id = _uid("REROUTE")
                for reroute_edge_id in reroute:
                    re = graph.edges[reroute_edge_id]
                    enter = clock
                    clock += re.base_travel_time * speed_factor / 60.0
                    distance += re.length
                    steps.append(TrajectoryStep(reroute_edge_id, len(steps), enter, clock, "REROUTED", reroute_id))
                    current = graph.other_node(reroute_edge_id, current)
                break
            enter = clock
            clock += edge.base_travel_time * speed_factor / 60.0
            distance += edge.length
            steps.append(TrajectoryStep(edge_id, len(steps), enter, clock))
            current = graph.other_node(edge_id, current)
        success = current == destination
        satisfaction = max(0.0, min(1.0, 0.82 - 0.08 * (clock - start_time) - (0.22 if rerouted else 0))) if success else 0.0
        return ExecutionResult(success, steps, failed_edge, rerouted, current, (clock - start_time) * 60.0, distance, success, satisfaction, None if success else "UNREACHABLE")


class M09ObservationExperienceGenerator:
    """M09: direction-aware visual search plus explicit direct experience."""

    def __init__(self, vision_enabled: bool = True, direct_experience_enabled: bool = True):
        self.vision_enabled = vision_enabled
        self.direct_experience_enabled = direct_experience_enabled
        self.vision = DirectionalVisionEngine()
        self.last_diagnostics: list[dict[str, Any]] = []

    def generate(
        self,
        day: int,
        episode: ActivityEpisode,
        profile: AgentProfile,
        selected_aoi: str | None,
        execution: ExecutionResult,
        graph: UrbanGraph,
        belief: CognitiveBeliefGraph,
        rng: random.Random,
    ) -> list[ObservationEvent]:
        events: list[ObservationEvent] = []
        self.last_diagnostics = []
        if self.direct_experience_enabled and execution.failed_edge_id:
            failed_step = next(
                (step for step in execution.steps if step.edge_id == execution.failed_edge_id),
                None,
            )
            events.append(self._event(
                day, episode, execution.failed_edge_id, "ROAD_EDGE", SourceChannel.ROUTE_FAILURE,
                "status", "CLOSED", 1.0, 1.0,
                event_time=failed_step.enter_time if failed_step else episode.planned_start,
            ))
        traversed = {step.edge_id for step in execution.steps if step.status != "BLOCKED"}
        directly_observed: set[tuple[str, str]] = set()
        for edge_id in sorted(traversed) if self.direct_experience_enabled else []:
            if graph.edges[edge_id].is_intervention:
                step = next(
                    (item for item in execution.steps if item.edge_id == edge_id and item.status != "BLOCKED"),
                    None,
                )
                events.append(self._event(
                    day, episode, edge_id, "ROAD_EDGE", SourceChannel.DIRECT_EXPERIENCE,
                    "status", str(graph.edges[edge_id].status), 1.0, 1.0,
                    event_time=step.exit_time if step else episode.planned_start,
                ))
                directly_observed.add(("ROAD_EDGE", edge_id))
        if self.direct_experience_enabled and selected_aoi and execution.success and graph.aois[selected_aoi].is_intervention:
            events.append(self._event(
                day, episode, selected_aoi, "AOI", SourceChannel.DIRECT_EXPERIENCE,
                "function", graph.aois[selected_aoi].function, 1.0, 1.0,
                event_time=episode.planned_start + execution.travel_time / 60.0,
            ))
            directly_observed.add(("AOI", selected_aoi))

        if not self.vision_enabled:
            return events
        samples = self.vision.trajectory_samples(execution, episode.origin_node, graph, profile)
        salience_by_object = graph.metadata.get("intervention_salience", {})
        configured_signage = graph.metadata.get("intervention_signage_support", {})
        for object_type, object_id in graph.intervention_object_ids():
            if (object_type, object_id) in directly_observed:
                continue
            if object_type == "ROAD_EDGE":
                default_salience = graph.edges[object_id].visibility
                attribute, value = "status", str(graph.edges[object_id].status)
            elif object_type == "AOI":
                default_salience = 0.82
                attribute, value = "function", graph.aois[object_id].function
            else:
                default_salience = graph.entrances[object_id].conspicuity
                attribute, value = "status", str(graph.entrances[object_id].status)
            salience = float(salience_by_object.get(object_id, default_salience))
            active_sign_bonus = max(
                (sign.credibility for sign in graph.signage.values() if sign.active and sign.target_id == object_id),
                default=0.0,
            )
            salience = min(1.0, salience + 0.25 * max(active_sign_bonus, float(configured_signage.get(object_id, 0.0))))
            visibility = self.vision.evaluate(
                samples, self.vision.object_geometry(graph, object_type, object_id),
                graph, profile, salience, rng,
            )
            diagnostic = {
                "diagnostic_id": _uid("VIS"), "day": day, "agent_id": profile.agent_id,
                "episode_id": episode.episode_id, "object_id": object_id, "object_type": object_type,
                "detected": visibility.detected, "probability": visibility.probability,
                "draw": visibility.draw, "distance": visibility.distance,
                "relative_angle_deg": visibility.relative_angle_deg,
                "line_of_sight": visibility.line_of_sight,
                "observer_x": visibility.sample.x if visibility.sample else None,
                "observer_y": visibility.sample.y if visibility.sample else None,
                "heading_deg": visibility.sample.heading_deg if visibility.sample else None,
                "sample_time": visibility.sample.event_time if visibility.sample else None,
            }
            self.last_diagnostics.append(diagnostic)
            if visibility.detected:
                sample = visibility.sample
                events.append(self._event(
                    day, episode, object_id, object_type, SourceChannel.VISUAL_SEARCH,
                    attribute, value, 0.82, salience, visibility.distance,
                    observer_x=sample.x if sample else None,
                    observer_y=sample.y if sample else None,
                    heading_deg=sample.heading_deg if sample else None,
                    relative_angle_deg=visibility.relative_angle_deg,
                    line_of_sight=visibility.line_of_sight,
                    detection_probability=visibility.probability,
                    detection_draw=visibility.draw,
                    event_time=sample.event_time if sample else episode.planned_start,
                ))
        return events

    def _event(
        self, day: int, episode: ActivityEpisode, object_id: str, object_type: str,
        channel: SourceChannel, attribute: str, value: str, reliability: float,
        salience: float, distance: float = 0.0, **diagnostics: Any,
    ) -> ObservationEvent:
        event_time = float(diagnostics.pop("event_time", episode.planned_start))
        return ObservationEvent(
            _uid("OBS"), day, event_time, episode.agent_id, episode.episode_id,
            object_id, object_type, channel, attribute, value, reliability, salience, distance,
            **diagnostics,
        )


class M10InformationArrivalUpdater:
    """M10: write valid information arrivals directly into the cognitive map.

    The simulation has daily resolution.  Information acquired during day t is
    visible in the day-t cognitive snapshot and can affect choices from the
    next simulated morning. Information is represented as a binary known state;
    no latent belief score or replacement threshold is used.
    """

    def __init__(self) -> None:
        self.ledger = InformationLedger()
        self._detected: set[tuple[str, str, str]] = set()

    def process(
        self,
        day: int,
        profile: AgentProfile,
        belief: CognitiveBeliefGraph,
        events: list[ObservationEvent],
        graph: UrbanGraph,
    ) -> tuple[list[CognitiveDetectionRecord], list[InformationItem], list[CognitiveUpdateRecord]]:
        detections: list[CognitiveDetectionRecord] = []
        updates: list[CognitiveUpdateRecord] = []
        information = self.ledger.ingest(day, profile.agent_id, events)
        for event in sorted(events, key=lambda item: item.event_time):
            prior = self._believed_value(belief, event.object_type, event.object_id)
            if self._states_match(prior, event.observed_value):
                continue
            key = (profile.agent_id, event.object_type, event.object_id)
            first = key not in self._detected
            self._detected.add(key)
            detections.append(CognitiveDetectionRecord(
                detection_id=_uid("DET"), day=day, event_time=event.event_time,
                agent_id=profile.agent_id, object_id=event.object_id,
                object_type=event.object_type,
                prior_value="UNKNOWN" if prior is None else str(prior),
                observed_value=str(event.observed_value),
                source_channel=event.source_channel,
                source_event_id=event.event_id,
                first_detection=first,
            ))
            update = self._write_arrival(day, profile.agent_id, belief, event, graph)
            if update is not None:
                updates.append(update)
        return detections, information, updates

    def _write_arrival(
        self,
        day: int,
        agent_id: str,
        belief: CognitiveBeliefGraph,
        event: ObservationEvent,
        graph: UrbanGraph,
    ) -> CognitiveUpdateRecord | None:
        source = str(event.source_channel)
        attribute = event.observed_attribute
        value = str(event.observed_value)
        old = "UNKNOWN"
        new = value
        if event.object_type == "ROAD_EDGE" and event.object_id in graph.edges:
            current = belief.edges.get(event.object_id)
            if current is None:
                actual = graph.edges[event.object_id]
                current = EdgeBelief(
                    edge_id=event.object_id,
                    believed_status=EdgeStatus.UNKNOWN,
                    perceived_travel_time=actual.base_travel_time,
                    perceived_safety=actual.safety,
                    information_source=source,
                    first_known_day=day,
                    last_update_day=day,
                    adoption_state=AdoptionState.UNKNOWN,
                )
                belief.edges[event.object_id] = current
            old = str(current.believed_status)
            current.believed_status = EdgeStatus(value)
            current.information_source = source
            current.last_update_day = day
            if current.adoption_state == AdoptionState.UNKNOWN:
                current.adoption_state = AdoptionState.AWARE
            new = str(current.believed_status)
            attribute = "believed_status"
        elif event.object_type == "AOI" and event.object_id in graph.aois:
            current_aoi = belief.aois.get(event.object_id)
            if current_aoi is None:
                actual_aoi = graph.aois[event.object_id]
                current_aoi = AOIBelief(
                    aoi_id=event.object_id,
                    believed_function="UNKNOWN",
                    perceived_safety=actual_aoi.safety,
                    perceived_comfort=actual_aoi.comfort,
                    perceived_attractiveness=actual_aoi.attractiveness,
                    information_source=source,
                    first_known_day=day,
                    last_update_day=day,
                    adoption_state=AdoptionState.UNKNOWN,
                )
                belief.aois[event.object_id] = current_aoi
            old = current_aoi.believed_function
            current_aoi.believed_function = value
            current_aoi.information_source = source
            current_aoi.last_update_day = day
            if current_aoi.adoption_state == AdoptionState.UNKNOWN:
                current_aoi.adoption_state = AdoptionState.AWARE
            new = current_aoi.believed_function
            attribute = "believed_function"
        elif event.object_type == "ENTRANCE" and event.object_id in graph.entrances:
            entrance = graph.entrances[event.object_id]
            current_entrance = belief.entrances.get(event.object_id)
            if current_entrance is None:
                current_entrance = EntranceBelief(
                    entrance_id=event.object_id,
                    believed_status=EdgeStatus.UNKNOWN,
                    conspicuity=entrance.conspicuity,
                    information_source=source,
                    first_known_day=day,
                    last_update_day=day,
                )
                belief.entrances[event.object_id] = current_entrance
            old = str(current_entrance.believed_status)
            current_entrance.believed_status = EdgeStatus(value)
            current_entrance.information_source = source
            current_entrance.last_update_day = day
            new = str(current_entrance.believed_status)
            attribute = "believed_status"
        else:
            return None
        if old == new:
            return None
        return CognitiveUpdateRecord(
            update_id=_uid("UPD"), day=day, agent_id=agent_id,
            object_id=event.object_id, object_type=event.object_type,
            attribute_name=attribute, old_value=old, new_value=new,
            update_reason=source, source_event_id=event.event_id,
            event_time=event.event_time, source_agent_id=event.source_agent_id,
            first_hand_source=event.first_hand_source,
        )

    @staticmethod
    def _believed_value(
        belief: CognitiveBeliefGraph, object_type: str, object_id: str,
    ) -> str | None:
        if object_type == "ROAD_EDGE":
            item = belief.edges.get(object_id)
            return None if item is None else str(item.believed_status)
        if object_type == "AOI":
            item = belief.aois.get(object_id)
            return None if item is None else item.believed_function
        if object_type == "ENTRANCE":
            item = belief.entrances.get(object_id)
            return None if item is None else str(item.believed_status)
        return None

    @staticmethod
    def _states_match(left: str | None, right: str) -> bool:
        return left is not None and str(left).strip().upper() == str(right).strip().upper()


class M11DataRecorderAggregator:
    """M11: persistence facade; aggregation is performed transactionally by Storage."""

    def __init__(self, storage: Any):
        self.storage = storage

    def save(self, result: Any, beliefs: dict[str, CognitiveBeliefGraph], graph: UrbanGraph, groups: list[str]) -> None:
        self.storage.save_day(result, beliefs, graph, groups)


MODULE_REGISTRY = {
    "M00": M00ExperimentController,
    "M01": M01ActualEnvironmentManager,
    "M02": M02InterventionManager,
    "M03": M03AgentStateManager,
    "M04": M04CognitiveMapManager,
    "M05": M05DailyActivityGenerator,
    "M06": M06DestinationDecisionMaker,
    "M07": M07CognitiveRoutePlanner,
    "M08": M08ActualEnvironmentExecutor,
    "M09": M09ObservationExperienceGenerator,
    "M10": M10InformationArrivalUpdater,
    "M11": M11DataRecorderAggregator,
}
