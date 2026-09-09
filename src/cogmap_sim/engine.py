from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .graph import UrbanGraph
from .interaction import CoPresenceInteractionEngine
from .llm import AgentLLMDriver, create_llm_driver
from .models import ActivityStop, AdoptionState, DayResult, ScenarioConfig
from .modules import (
    M00ExperimentController,
    M01ActualEnvironmentManager,
    M02InterventionManager,
    M03AgentStateManager,
    M04CognitiveMapManager,
    M05DailyActivityGenerator,
    M06DestinationDecisionMaker,
    M07CognitiveRoutePlanner,
    M08ActualEnvironmentExecutor,
    M09ObservationExperienceGenerator,
    M10InformationArrivalUpdater,
    M11DataRecorderAggregator,
)
from .storage import Storage


class SimulationEngine:
    """Coordinates the documented t0 initialization and t1..tT daily pipeline."""

    def __init__(self, config: ScenarioConfig, graph: UrbanGraph, agents: list[Any], interventions: list[Any], output_path: str | Path, llm_driver: AgentLLMDriver | None = None):
        self.config = config
        self.pre_intervention_graph = graph.clone()
        self.environment = M01ActualEnvironmentManager(graph)
        self.agents = agents
        self.interventions = interventions
        self.storage = Storage(output_path)
        self.m00 = M00ExperimentController(config.random_seed)
        self.m02 = M02InterventionManager()
        self.m03 = M03AgentStateManager()
        self.m04 = M04CognitiveMapManager()
        self.m05 = M05DailyActivityGenerator()
        self.m06 = M06DestinationDecisionMaker()
        self.m07 = M07CognitiveRoutePlanner()
        self.m08 = M08ActualEnvironmentExecutor()
        self.m09 = M09ObservationExperienceGenerator(
            config.directional_vision_enabled,
            config.direct_experience_enabled,
        )
        self.m10 = M10InformationArrivalUpdater()
        self.m11 = M11DataRecorderAggregator(self.storage)
        self.llm = llm_driver or create_llm_driver(config.behavior_driver, config.llm_model)
        self.interaction = CoPresenceInteractionEngine(
            config.copresence_min_overlap_minutes,
            config.max_reports_per_conversation,
        )
        self.agent_memory: dict[str, list[dict[str, Any]]] = {agent.agent_id: [] for agent in agents}
        self.beliefs = self.m04.initialize(self.pre_intervention_graph, agents, config.random_seed)
        self.groups = sorted({a.group_id for a in agents})

    def run(self, progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        model_type = self.config.canonical_model_type
        self.storage.initialize(self.config, self.pre_intervention_graph, self.agents, self.interventions)
        self.storage.save_environment(-1, self.pre_intervention_graph)
        self.storage.save_cognitive_snapshot(0, self.beliefs)
        if model_type != "no_change":
            self.m02.apply(self.environment.graph, self.interventions, 0)
        self.storage.save_environment(0, self.environment.graph)
        self.storage.connection.commit()
        if model_type == "objective_accessibility":
            self.storage.save_objective_accessibility(0, self.environment.graph, self.agents)
            for day in range(1, self.config.end_day + 1):
                self.m02.apply(self.environment.graph, self.interventions, day)
                self.storage.save_environment(day, self.environment.graph)
                self.storage.save_objective_accessibility(day, self.environment.graph, self.agents)
                if progress:
                    progress(day, self.config.end_day)
            self.storage.finalize_research_outputs()
            summary = self.summary()
            self.storage.close()
            return summary
        for day in range(1, self.config.end_day + 1):
            if model_type != "no_change":
                self.m02.apply(self.environment.graph, self.interventions, day)
            if model_type == "omniscient":
                for belief in self.beliefs.values():
                    self.m04.make_omniscient(belief, self.environment.graph, day)
            result = self._run_day(day)
            self.m11.save(result, self.beliefs, self.environment.graph, self.groups)
            if progress:
                progress(day, self.config.end_day)
        self.storage.finalize_research_outputs()
        summary = self.summary()
        self.storage.close()
        return summary

    def _run_day(self, day: int) -> DayResult:
        graph = self.environment.snapshot()
        model_type = self.config.canonical_model_type
        result = DayResult(day)
        contexts: dict[str, dict[str, Any]] = {}
        for profile in self.agents:
            rng = self.m00.random_for(day, profile.agent_id)
            belief = self.beliefs[profile.agent_id]
            before_known = len(belief.edges) + len(belief.aois) + len(belief.entrances)
            known_aois = [{
                "aoi_id": item.aoi_id, "believed_function": item.believed_function,
                "perceived_attractiveness": item.perceived_attractiveness,
                "perceived_safety": item.perceived_safety,
            } for item in belief.aois.values()]
            llm_plan = self.llm.plan_day(profile, day, known_aois, self.agent_memory[profile.agent_id][-10:], rng)
            state = self.m03.start_day(day, profile, graph)
            episodes = self.m05.generate(day, profile, graph, rng, llm_plan)
            state.episode_count = len(episodes)
            current_node = state.start_location
            temporary_closed: set[str] = set()
            agent_events = []
            satisfaction_values: list[float] = []
            for episode in episodes:
                episode.origin_node = current_node
                preferences = [] if model_type == "objective_accessibility" else llm_plan.get(
                    "destination_preferences", {}
                ).get(episode.activity_type, [])
                decision = self.m06.choose(episode, profile, belief, graph, rng, model_type, preferences)
                result.decisions.append(decision)
                destination_node = graph.aois[decision.selected_aoi_id].node_id if decision.selected_aoi_id else current_node
                planned_edges = self.m07.plan(current_node, destination_node, belief, graph) if decision.selected_aoi_id else None
                if planned_edges is not None:
                    clock = episode.planned_start
                    for sequence, edge_id in enumerate(planned_edges):
                        # Edge costs are minutes; daily clocks and co-presence windows are hours.
                        duration = belief.edges[edge_id].perceived_travel_time / 60.0
                        result.planned.append({"day": day, "agent_id": profile.agent_id, "episode_id": episode.episode_id, "sequence": sequence, "edge_id": edge_id, "enter_time": clock, "exit_time": clock + duration})
                        clock += duration
                execution = self.m08.execute(
                    planned_edges, current_node, destination_node, episode.planned_start,
                    graph, temporary_closed, belief, profile,
                )
                for step in execution.steps:
                    result.executed.append({"day": day, "agent_id": profile.agent_id, "episode_id": episode.episode_id, "sequence": step.sequence, "edge_id": step.edge_id, "enter_time": step.enter_time, "exit_time": step.exit_time, "status": step.status, "reroute_id": step.reroute_id})
                    if step.status != "BLOCKED" and step.edge_id in belief.edges:
                        edge_belief = belief.edges[step.edge_id]
                        edge_belief.use_count += 1
                        edge_belief.adoption_state = (
                            AdoptionState.ADOPTED if edge_belief.use_count >= 2 else AdoptionState.TRIED
                        )
                if execution.failed_edge_id:
                    result.temporary_memories.append({"day": day, "agent_id": profile.agent_id, "episode_id": episode.episode_id, "object_id": execution.failed_edge_id, "attribute": "status", "value": "CLOSED", "time": episode.planned_start})
                events = self.m09.generate(day, episode, profile, decision.selected_aoi_id, execution, graph, belief, rng)
                agent_events.extend(events)
                result.observations.extend(events)
                result.perception_diagnostics.extend(self.m09.last_diagnostics)
                if execution.success:
                    current_node = execution.arrival_node
                    if decision.selected_aoi_id:
                        aoi_belief = belief.aois.get(decision.selected_aoi_id)
                        if aoi_belief is not None:
                            aoi_belief.visit_count += 1
                            aoi_belief.adoption_state = (
                                AdoptionState.ADOPTED if aoi_belief.visit_count >= 2 else AdoptionState.TRIED
                            )
                        stop_start = episode.planned_start + execution.travel_time / 60.0
                        result.activity_stops.append(ActivityStop(
                            day=day, agent_id=profile.agent_id, episode_id=episode.episode_id,
                            place_id=decision.selected_aoi_id, activity_type=episode.activity_type,
                            start_time=stop_start, end_time=stop_start + episode.expected_duration,
                        ))
                state.total_travel_time += execution.travel_time
                state.total_distance += execution.distance
                state.failed_route_count += int(execution.failed_edge_id is not None)
                state.reroute_count += int(execution.rerouted)
                satisfaction_values.append(execution.satisfaction)
                result.episodes.append({
                    **asdict(episode), "selected_destination_id": decision.selected_aoi_id,
                    "activity_success": execution.activity_success,
                    "failure_reason": execution.failure_reason,
                    "satisfaction": execution.satisfaction,
                })
            state.end_location = current_node
            state.daily_satisfaction = sum(satisfaction_values) / max(1, len(satisfaction_values))
            result.agent_states.append(state)
            contexts[profile.agent_id] = {"events": agent_events, "state": state, "before_known": before_known, "rng": rng}

        if model_type == "dynamic_cognitive" and self.config.face_to_face_enabled:
            copresence = self.interaction.find_copresence(
                day, self.agents, result.activity_stops,
                lambda key: self.m00.random_for(day, key),
            )
            result.copresence_events.extend(copresence)
            reports_by_listener = self.interaction.build_reports(
                copresence,
                {agent_id: context["events"] for agent_id, context in contexts.items()},
                {profile.agent_id: self.m10.ledger.items_for(profile.agent_id) for profile in self.agents},
            )
            for profile in self.agents:
                reports = reports_by_listener.get(profile.agent_id, [])
                if not reports:
                    continue
                parsed = self.llm.interpret_face_to_face(
                    profile, day, reports, contexts[profile.agent_id]["rng"],
                )
                parsed_by_id = {
                    str(item.get("report_id")): item
                    for item in parsed.get("parsed_reports", []) if isinstance(item, dict)
                }
                for report in reports:
                    parsed_report = parsed_by_id.get(str(report["report_id"]))
                    if parsed_report is None:
                        continue
                    exchange, observation = self.interaction.materialize_exchange(
                        day, report,
                    )
                    result.interpersonal_exchanges.append(exchange)
                    contexts[profile.agent_id]["events"].append(observation)
                    result.observations.append(observation)

        # Daily resolution: valid information arriving on day t is written into
        # the day-t snapshot and can affect choices from the next morning.
        for profile in self.agents:
            context = contexts[profile.agent_id]
            belief = self.beliefs[profile.agent_id]
            if model_type == "dynamic_cognitive":
                detections, information, updates = self.m10.process(
                    day, profile, belief, context["events"], graph,
                )
                result.detections.extend(detections)
                result.information_items.extend(information)
                result.updates.extend(updates)
                self.agent_memory[profile.agent_id].extend([asdict(update) for update in updates])
                self.agent_memory[profile.agent_id] = self.agent_memory[profile.agent_id][-30:]
            context["state"].new_object_count = max(
                0, len(belief.edges) + len(belief.aois) + len(belief.entrances) - context["before_known"]
            )

        result.llm_decisions.extend(self.llm.pop_logs())
        return result

    def summary(self) -> dict[str, Any]:
        c = self.storage.connection
        return {
            "experiment_id": self.config.experiment_id,
            "model_type": self.config.canonical_model_type,
            "days": self.config.end_day,
            "agents": len(self.agents),
            "episodes": c.execute("SELECT COUNT(*) FROM activity_episode_log").fetchone()[0],
            "observations": c.execute("SELECT COUNT(*) FROM observation_event_log").fetchone()[0],
            "cognitive_updates": c.execute("SELECT COUNT(*) FROM cognitive_update_log").fetchone()[0],
            "route_failures": c.execute("SELECT COALESCE(SUM(failed_route_count),0) FROM agent_daily_state").fetchone()[0],
            "intervention_road_uses": c.execute(
                "SELECT COUNT(DISTINCT agent_id || ':' || simulation_day) FROM executed_trajectory_log "
                "WHERE edge_id IN (SELECT target_id FROM intervention_log WHERE operation_type LIKE 'ROAD_%')"
            ).fetchone()[0],
            "new_link_uses": c.execute(
                "SELECT COUNT(DISTINCT agent_id || ':' || simulation_day) "
                "FROM executed_trajectory_log WHERE edge_id='NEW_LINK'"
            ).fetchone()[0],
            "information_arrivals": c.execute("SELECT COUNT(*) FROM information_item").fetchone()[0],
            "copresence_events": c.execute("SELECT COUNT(*) FROM copresence_event").fetchone()[0],
            "conversations": c.execute("SELECT COUNT(*) FROM copresence_event WHERE conversation_flag=1").fetchone()[0],
            "delivered_interpersonal_reports": c.execute("SELECT COUNT(*) FROM interpersonal_exchange WHERE delivered=1").fetchone()[0],
            "llm_calls": c.execute("SELECT COUNT(*) FROM llm_decision_log").fetchone()[0],
        }
