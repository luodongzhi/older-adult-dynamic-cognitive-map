from __future__ import annotations

import json
import csv
import math
import platform
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

from .models import CognitiveBeliefGraph, DayResult, ScenarioConfig


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS experiment_config (experiment_id TEXT PRIMARY KEY, scenario_id TEXT, model_type TEXT, random_seed INTEGER, start_day INTEGER, end_day INTEGER, parameter_set_id TEXT, software_version TEXT, behavior_driver TEXT, llm_model TEXT);
CREATE TABLE IF NOT EXISTS run_manifest (experiment_id TEXT PRIMARY KEY, created_at TEXT, config_json TEXT, map_metadata_json TEXT, source_hashes_json TEXT, dependency_versions_json TEXT);
CREATE TABLE IF NOT EXISTS intervention_log (intervention_id TEXT PRIMARY KEY, effective_day INTEGER, operation_type TEXT, target_type TEXT, target_id TEXT, before_value TEXT, after_value TEXT, visibility_level REAL, announcement_level REAL, visual_salience REAL, official_announcement REAL, signage_support REAL);
CREATE TABLE IF NOT EXISTS node (node_id TEXT PRIMARY KEY, x REAL, y REAL);
CREATE TABLE IF NOT EXISTS road_edge (edge_id TEXT PRIMARY KEY, from_node TEXT, to_node TEXT, length REAL, base_travel_time REAL, initial_status TEXT, visibility REAL, safety REAL, comfort REAL, is_intervention INTEGER, geometry_json TEXT, name TEXT, road_class TEXT);
CREATE TABLE IF NOT EXISTS aoi_object (aoi_id TEXT PRIMARY KEY, name TEXT, node_id TEXT, initial_function TEXT, attractiveness REAL, safety REAL, comfort REAL, capacity INTEGER, is_intervention INTEGER, geometry_json TEXT, feature_kind TEXT);
CREATE TABLE IF NOT EXISTS entrance_object (entrance_id TEXT PRIMARY KEY, aoi_id TEXT, node_id TEXT, x REAL, y REAL, initial_status TEXT, conspicuity REAL, is_intervention INTEGER);
CREATE TABLE IF NOT EXISTS occluder_object (object_id TEXT PRIMARY KEY, object_type TEXT, opacity REAL, geometry_json TEXT);
CREATE TABLE IF NOT EXISTS signage_object (sign_id TEXT PRIMARY KEY, target_id TEXT, x REAL, y REAL, facing_deg REAL, range_m REAL, credibility REAL, active INTEGER);
CREATE TABLE IF NOT EXISTS agent_profile (agent_id TEXT PRIMARY KEY, group_id TEXT, home_aoi_id TEXT, routine_anchor_aoi_id TEXT, profile_json TEXT);
CREATE TABLE IF NOT EXISTS environment_daily_state (simulation_day INTEGER, object_id TEXT, object_type TEXT, actual_value TEXT, PRIMARY KEY(simulation_day, object_id));
CREATE TABLE IF NOT EXISTS agent_daily_state (simulation_day INTEGER, agent_id TEXT, start_location TEXT, end_location TEXT, episode_count INTEGER, total_travel_time REAL, total_distance REAL, failed_route_count INTEGER, reroute_count INTEGER, new_object_count INTEGER, daily_satisfaction REAL, PRIMARY KEY(simulation_day, agent_id));
CREATE TABLE IF NOT EXISTS activity_episode_log (simulation_day INTEGER, agent_id TEXT, episode_id TEXT, sequence_number INTEGER, activity_type TEXT, origin_node TEXT, selected_destination_id TEXT, activity_success INTEGER, failure_reason TEXT, satisfaction REAL, PRIMARY KEY(simulation_day, agent_id, episode_id));
CREATE TABLE IF NOT EXISTS destination_decision_log (simulation_day INTEGER, agent_id TEXT, episode_id TEXT, candidates_json TEXT, selected_aoi_id TEXT, expected_utility REAL, choice_probability REAL, factors_json TEXT, decision_status TEXT);
CREATE TABLE IF NOT EXISTS planned_trajectory_log (simulation_day INTEGER, agent_id TEXT, episode_id TEXT, sequence_index INTEGER, edge_id TEXT, enter_time REAL, exit_time REAL);
CREATE TABLE IF NOT EXISTS executed_trajectory_log (simulation_day INTEGER, agent_id TEXT, episode_id TEXT, sequence_index INTEGER, edge_id TEXT, enter_time REAL, exit_time REAL, execution_status TEXT, reroute_id TEXT);
CREATE TABLE IF NOT EXISTS observation_event_log (event_id TEXT PRIMARY KEY, simulation_day INTEGER, event_time REAL, agent_id TEXT, episode_id TEXT, object_id TEXT, object_type TEXT, source_channel TEXT, observed_attribute TEXT, observed_value TEXT, source_reliability REAL, salience REAL, distance_to_object REAL, observer_x REAL, observer_y REAL, heading_deg REAL, relative_angle_deg REAL, line_of_sight INTEGER, detection_probability REAL, detection_draw REAL, source_event_id TEXT, independence_group TEXT, source_agent_id TEXT, copresence_event_id TEXT, first_hand_source INTEGER);
CREATE TABLE IF NOT EXISTS perception_diagnostic_log (diagnostic_id TEXT PRIMARY KEY, simulation_day INTEGER, agent_id TEXT, episode_id TEXT, object_id TEXT, object_type TEXT, detected INTEGER, probability REAL, draw REAL, distance REAL, relative_angle_deg REAL, line_of_sight INTEGER, observer_x REAL, observer_y REAL, heading_deg REAL, sample_time REAL);
CREATE TABLE IF NOT EXISTS temporary_memory_log (simulation_day INTEGER, agent_id TEXT, episode_id TEXT, object_id TEXT, temporary_attribute TEXT, temporary_value TEXT, created_time REAL);
CREATE TABLE IF NOT EXISTS information_item (information_id TEXT PRIMARY KEY, simulation_day INTEGER, agent_id TEXT, object_id TEXT, object_type TEXT, attribute_name TEXT, observed_value TEXT, source_channel TEXT, source_event_id TEXT, reliability REAL, salience REAL, independence_group TEXT, source_agent_id TEXT, copresence_event_id TEXT, first_hand_source INTEGER);
CREATE TABLE IF NOT EXISTS cognitive_detection_log (detection_id TEXT PRIMARY KEY, simulation_day INTEGER, event_time REAL, agent_id TEXT, object_id TEXT, object_type TEXT, prior_value TEXT, observed_value TEXT, source_channel TEXT, source_event_id TEXT, first_detection INTEGER);
CREATE TABLE IF NOT EXISTS cognitive_update_log (update_id TEXT PRIMARY KEY, simulation_day INTEGER, event_time REAL, agent_id TEXT, object_id TEXT, object_type TEXT, attribute_name TEXT, old_value TEXT, new_value TEXT, update_reason TEXT, source_event_id TEXT, source_agent_id TEXT, first_hand_source INTEGER);
CREATE TABLE IF NOT EXISTS llm_decision_log (call_id TEXT PRIMARY KEY, simulation_day INTEGER, agent_id TEXT, phase TEXT, provider TEXT, model TEXT, request_json TEXT, response_json TEXT, latency_ms REAL, status TEXT);
CREATE TABLE IF NOT EXISTS activity_stop_log (simulation_day INTEGER, agent_id TEXT, episode_id TEXT, place_id TEXT, activity_type TEXT, start_time REAL, end_time REAL, PRIMARY KEY(simulation_day, agent_id, episode_id));
CREATE TABLE IF NOT EXISTS copresence_event (event_id TEXT PRIMARY KEY, simulation_day INTEGER, agent_a TEXT, agent_b TEXT, place_id TEXT, overlap_start REAL, overlap_end REAL, overlap_minutes REAL, relationship_strength REAL, conversation_probability REAL, conversation_draw REAL, conversation_flag INTEGER);
CREATE TABLE IF NOT EXISTS interpersonal_exchange (exchange_id TEXT PRIMARY KEY, simulation_day INTEGER, copresence_event_id TEXT, speaker_agent_id TEXT, listener_agent_id TEXT, place_id TEXT, report_id TEXT, object_id TEXT, object_type TEXT, observed_attribute TEXT, observed_value TEXT, source_event_id TEXT, independence_group TEXT, first_hand_source INTEGER, delivered INTEGER);
CREATE TABLE IF NOT EXISTS cognitive_edge_daily_state (simulation_day INTEGER, agent_id TEXT, edge_id TEXT, believed_status TEXT, information_source TEXT, first_known_day INTEGER, last_update_day INTEGER, adoption_state TEXT, use_count INTEGER, PRIMARY KEY(simulation_day, agent_id, edge_id));
CREATE TABLE IF NOT EXISTS cognitive_aoi_daily_state (simulation_day INTEGER, agent_id TEXT, aoi_id TEXT, believed_function TEXT, perceived_attractiveness REAL, information_source TEXT, first_known_day INTEGER, last_update_day INTEGER, adoption_state TEXT, visit_count INTEGER, PRIMARY KEY(simulation_day, agent_id, aoi_id));
CREATE TABLE IF NOT EXISTS cognitive_entrance_daily_state (simulation_day INTEGER, agent_id TEXT, entrance_id TEXT, believed_status TEXT, information_source TEXT, first_known_day INTEGER, last_update_day INTEGER, PRIMARY KEY(simulation_day, agent_id, entrance_id));
CREATE TABLE IF NOT EXISTS knowledge_update_event (simulation_day INTEGER, event_time REAL, agent_id TEXT, object_id TEXT, object_type TEXT, update_id TEXT, source_channel TEXT, PRIMARY KEY(agent_id, object_id));
CREATE TABLE IF NOT EXISTS mismatch_daily (simulation_day INTEGER, object_id TEXT, object_type TEXT, objective_improvement REAL, awareness_rate REAL, realized_use_rate REAL, mismatch_index REAL, PRIMARY KEY(simulation_day, object_id));
CREATE TABLE IF NOT EXISTS road_daily_aggregate (simulation_day INTEGER, edge_id TEXT, planned_agent_count INTEGER, actual_agent_count INTEGER, failed_attempt_count INTEGER, reroute_count INTEGER, first_time_user_count INTEGER, repeat_user_count INTEGER, PRIMARY KEY(simulation_day, edge_id));
CREATE TABLE IF NOT EXISTS aoi_daily_aggregate (simulation_day INTEGER, aoi_id TEXT, visit_count INTEGER, unique_agent_count INTEGER, activity_success_count INTEGER, function_mismatch_count INTEGER, first_visit_count INTEGER, repeat_visit_count INTEGER, average_satisfaction REAL, PRIMARY KEY(simulation_day, aoi_id));
CREATE TABLE IF NOT EXISTS group_daily_aggregate (simulation_day INTEGER, group_id TEXT, awareness_rate REAL, trial_rate REAL, adoption_rate REAL, rejection_rate REAL, average_travel_time REAL, average_route_failure_rate REAL, average_cognitive_accuracy REAL, PRIMARY KEY(simulation_day, group_id));
CREATE TABLE IF NOT EXISTS cognitive_lag_daily (simulation_day INTEGER, agent_id TEXT, intervention_id TEXT, object_id TEXT, object_type TEXT, intervention_day INTEGER, actual_state TEXT, believed_state TEXT, mismatch_class TEXT, baseline_state TEXT, baseline_known INTEGER, lag_indicator REAL, PRIMARY KEY(simulation_day, agent_id, intervention_id));
CREATE TABLE IF NOT EXISTS cognitive_lag_population_daily (simulation_day INTEGER, intervention_id TEXT, object_id TEXT, object_type TEXT, correct_update_rate REAL, unknown_rate REAL, obsolete_rate REAL, incorrect_rate REAL, mean_lag_index REAL, t50 INTEGER, t80 INTEGER, PRIMARY KEY(simulation_day, intervention_id));
CREATE TABLE IF NOT EXISTS update_timing (intervention_id TEXT, agent_id TEXT, object_id TEXT, object_type TEXT, intervention_day INTEGER, first_visual_day INTEGER, first_experience_day INTEGER, first_interpersonal_day INTEGER, first_information_day INTEGER, first_correct_update_day INTEGER, first_use_day INTEGER, information_delay INTEGER, behavior_delay INTEGER, obsolete_duration INTEGER, right_censored INTEGER, first_information_channel TEXT, baseline_state TEXT, route_habit REAL, visual_attention REAL, social_participation REAL, age INTEGER, PRIMARY KEY(intervention_id, agent_id));
CREATE TABLE IF NOT EXISTS information_process_daily (simulation_day INTEGER, agent_id TEXT, visual_count INTEGER, experience_count INTEGER, interpersonal_count INTEGER, copresence_count INTEGER, conversation_count INTEGER, delivered_report_count INTEGER, firsthand_report_count INTEGER, unique_target_count INTEGER, information_coverage REAL, PRIMARY KEY(simulation_day, agent_id));
CREATE TABLE IF NOT EXISTS route_outcome_daily (simulation_day INTEGER, agent_id TEXT, planned_edge_count INTEGER, executed_edge_count INTEGER, route_difference_ratio REAL, planned_distance REAL, executed_distance REAL, detour_distance REAL, route_failures INTEGER, reroute_count INTEGER, activity_success_rate REAL, intervention_use_count INTEGER, PRIMARY KEY(simulation_day, agent_id));
CREATE TABLE IF NOT EXISTS activity_distribution_daily (simulation_day INTEGER, activity_type TEXT, aoi_id TEXT, visit_count INTEGER, success_count INTEGER, destination_share REAL, PRIMARY KEY(simulation_day, activity_type, aoi_id));
CREATE TABLE IF NOT EXISTS objective_accessibility_daily (simulation_day INTEGER, origin_aoi_id TEXT, activity_type TEXT, reachable_places INTEGER, accessibility_score REAL, normalized_accessibility REAL, PRIMARY KEY(simulation_day, origin_aoi_id, activity_type));
CREATE TABLE IF NOT EXISTS daily_metric (simulation_day INTEGER, research_question TEXT, level_type TEXT, level_id TEXT, metric_name TEXT, value REAL, PRIMARY KEY(simulation_day, research_question, level_type, level_id, metric_name));
CREATE TABLE IF NOT EXISTS model_comparison (simulation_day INTEGER, research_question TEXT, model_a TEXT, model_b TEXT, metric_name TEXT, value_a REAL, value_b REAL, difference REAL, notes TEXT, PRIMARY KEY(simulation_day, model_a, model_b, metric_name));
CREATE INDEX IF NOT EXISTS idx_obs_agent_day ON observation_event_log(agent_id, simulation_day);
CREATE INDEX IF NOT EXISTS idx_update_agent_day ON cognitive_update_log(agent_id, simulation_day);
CREATE INDEX IF NOT EXISTS idx_exec_edge_day ON executed_trajectory_log(edge_id, simulation_day);
CREATE INDEX IF NOT EXISTS idx_copresence_day ON copresence_event(simulation_day, place_id);
CREATE INDEX IF NOT EXISTS idx_exchange_listener_day ON interpersonal_exchange(listener_agent_id, simulation_day);
CREATE INDEX IF NOT EXISTS idx_lag_agent_day ON cognitive_lag_daily(agent_id, simulation_day);
"""


class Storage:
    def __init__(self, path: str | Path, reset: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.path.exists():
            self.path.unlink()
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def initialize(self, config: ScenarioConfig, graph: Any, agents: list[Any], interventions: list[Any]) -> None:
        c = self.connection
        c.execute("INSERT INTO experiment_config VALUES (?,?,?,?,?,?,?,?,?,?)", (
            config.experiment_id, config.scenario_id, config.canonical_model_type,
            config.random_seed, config.start_day, config.end_day, config.parameter_set_id,
            config.code_version, config.behavior_driver, config.llm_model,
        ))
        dependencies = {"python": platform.python_version()}
        for package in ("shapely", "pyproj", "pyshp", "networkx"):
            try:
                dependencies[package] = version(package)
            except PackageNotFoundError:
                dependencies[package] = None
        c.execute("INSERT INTO run_manifest VALUES (?,?,?,?,?,?)", (
            config.experiment_id, datetime.now(timezone.utc).isoformat(),
            json.dumps(asdict(config), ensure_ascii=False),
            json.dumps(graph.metadata, ensure_ascii=False),
            json.dumps(graph.metadata.get("source_hashes", {}), ensure_ascii=False),
            json.dumps(dependencies, ensure_ascii=False),
        ))
        manifest = {
            "experiment_id": config.experiment_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "map_metadata": graph.metadata,
            "source_hashes": graph.metadata.get("source_hashes", {}),
            "dependencies": dependencies,
            "database": self.path.name,
        }
        self.path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        c.executemany("INSERT INTO intervention_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [(
            item.intervention_id, item.effective_day, item.operation_type, item.target_type,
            item.target_id, item.before_value, item.after_value, item.visibility_level,
            item.announcement_level, item.visual_salience, item.official_announcement,
            item.signage_support,
        ) for item in interventions])
        intervention_targets = {item.target_id for item in interventions}
        c.executemany("INSERT INTO node VALUES (?,?,?)", [(n.node_id, n.x, n.y) for n in graph.nodes.values()])
        c.executemany("INSERT INTO road_edge VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [(
            e.edge_id, e.from_node, e.to_node, e.length, e.base_travel_time,
            str(e.status), e.visibility, e.safety, e.comfort,
            int(e.is_intervention or e.edge_id in intervention_targets),
            json.dumps(e.geometry), e.name, e.road_class,
        ) for e in graph.edges.values()])
        c.executemany("INSERT INTO aoi_object VALUES (?,?,?,?,?,?,?,?,?,?,?)", [(a.aoi_id, a.name, a.node_id, a.function, a.attractiveness, a.safety, a.comfort, a.capacity, int(a.is_intervention or a.aoi_id in intervention_targets), json.dumps(a.geometry), a.feature_kind) for a in graph.aois.values()])
        c.executemany("INSERT INTO entrance_object VALUES (?,?,?,?,?,?,?,?)", [(
            item.entrance_id, item.aoi_id, item.node_id, item.x, item.y,
            str(item.status), item.conspicuity, int(item.is_intervention or item.entrance_id in intervention_targets),
        ) for item in graph.entrances.values()])
        c.executemany("INSERT INTO occluder_object VALUES (?,?,?,?)", [(
            item.object_id, item.object_type, item.opacity, json.dumps(item.geometry),
        ) for item in graph.occluders.values()])
        c.executemany("INSERT INTO signage_object VALUES (?,?,?,?,?,?,?,?)", [(
            item.sign_id, item.target_id, item.x, item.y, item.facing_deg,
            item.range_m, item.credibility, int(item.active),
        ) for item in graph.signage.values()])
        c.executemany("INSERT INTO agent_profile VALUES (?,?,?,?,?)", [(
            a.agent_id, a.group_id, a.home_aoi_id, a.routine_anchor_aoi_id,
            json.dumps(asdict(a), ensure_ascii=False),
        ) for a in agents])
        c.commit()

    def save_environment(self, day: int, graph: Any) -> None:
        rows = [(day, e.edge_id, "ROAD_EDGE", str(e.status)) for e in graph.edges.values()]
        rows += [(day, a.aoi_id, "AOI", a.function) for a in graph.aois.values()]
        rows += [(day, item.entrance_id, "ENTRANCE", str(item.status)) for item in graph.entrances.values()]
        rows += [(day, item.sign_id, "SIGNAGE", str(item.active).upper()) for item in graph.signage.values()]
        self.connection.executemany("INSERT OR REPLACE INTO environment_daily_state VALUES (?,?,?,?)", rows)

    def save_objective_accessibility(self, day: int, graph: Any, agents: list[Any]) -> None:
        origins = sorted({agent.home_aoi_id for agent in agents})
        rows = []
        metric_rows = []
        for origin_id in origins:
            origin = graph.aois[origin_id]
            for activity_type in ("CAFE", "PARK", "SHOP"):
                destinations = [
                    item for item in graph.aois.values()
                    if item.function == activity_type and item.aoi_id != origin_id
                ]
                total_attractiveness = sum(max(0.01, item.attractiveness) for item in destinations)
                reachable = 0
                score = 0.0
                for destination in destinations:
                    path = graph.shortest_path(origin.node_id, destination.node_id)
                    if path is None:
                        continue
                    reachable += 1
                    travel_time = sum(graph.edges[edge_id].base_travel_time for edge_id in path)
                    score += max(0.01, destination.attractiveness) * math.exp(-travel_time / 20.0)
                normalized = score / max(1e-9, total_attractiveness)
                rows.append((day, origin_id, activity_type, reachable, score, normalized))
                metric_rows.append((
                    day, "RQ3", "ORIGIN_ACTIVITY", f"{origin_id}:{activity_type}",
                    "objective_accessibility", normalized,
                ))
        self.connection.executemany(
            "INSERT OR REPLACE INTO objective_accessibility_daily VALUES (?,?,?,?,?,?)", rows,
        )
        self.connection.executemany(
            "INSERT OR REPLACE INTO daily_metric VALUES (?,?,?,?,?,?)", metric_rows,
        )
        self.connection.commit()

    def save_cognitive_snapshot(self, day: int, beliefs: dict[str, CognitiveBeliefGraph]) -> None:
        edge_rows, aoi_rows, entrance_rows = [], [], []
        for agent_id, graph in beliefs.items():
            for b in graph.edges.values():
                edge_rows.append((
                    day, agent_id, b.edge_id, str(b.believed_status),
                    b.information_source, b.first_known_day, b.last_update_day,
                    str(b.adoption_state), b.use_count,
                ))
            for b in graph.aois.values():
                aoi_rows.append((
                    day, agent_id, b.aoi_id, b.believed_function,
                    b.perceived_attractiveness, b.information_source,
                    b.first_known_day, b.last_update_day,
                    str(b.adoption_state), b.visit_count,
                ))
            for b in graph.entrances.values():
                entrance_rows.append((
                    day, agent_id, b.entrance_id, str(b.believed_status),
                    b.information_source, b.first_known_day, b.last_update_day,
                ))
        self.connection.executemany("INSERT OR REPLACE INTO cognitive_edge_daily_state VALUES (?,?,?,?,?,?,?,?,?)", edge_rows)
        self.connection.executemany("INSERT OR REPLACE INTO cognitive_aoi_daily_state VALUES (?,?,?,?,?,?,?,?,?,?)", aoi_rows)
        self.connection.executemany("INSERT OR REPLACE INTO cognitive_entrance_daily_state VALUES (?,?,?,?,?,?,?)", entrance_rows)

    def save_day(self, result: DayResult, beliefs: dict[str, CognitiveBeliefGraph], graph: Any, groups: list[str]) -> None:
        c = self.connection
        self.save_environment(result.day, graph)
        c.executemany("INSERT INTO agent_daily_state VALUES (?,?,?,?,?,?,?,?,?,?,?)", [(s.day, s.agent_id, s.start_location, s.end_location, s.episode_count, s.total_travel_time, s.total_distance, s.failed_route_count, s.reroute_count, s.new_object_count, s.daily_satisfaction) for s in result.agent_states])
        c.executemany("INSERT INTO activity_episode_log VALUES (?,?,?,?,?,?,?,?,?,?)", [(x["day"], x["agent_id"], x["episode_id"], x["sequence"], x["activity_type"], x["origin_node"], x.get("selected_destination_id"), int(x.get("activity_success", False)), x.get("failure_reason"), x.get("satisfaction", 0.0)) for x in result.episodes])
        c.executemany("INSERT INTO destination_decision_log VALUES (?,?,?,?,?,?,?,?,?)", [(d.day, d.agent_id, d.episode_id, json.dumps(d.candidate_aoi_ids), d.selected_aoi_id, d.expected_utility, d.choice_probability, json.dumps(d.decision_factors), d.status) for d in result.decisions])
        c.executemany("INSERT INTO planned_trajectory_log VALUES (?,?,?,?,?,?,?)", [(x["day"], x["agent_id"], x["episode_id"], x["sequence"], x["edge_id"], x["enter_time"], x["exit_time"]) for x in result.planned])
        c.executemany("INSERT INTO executed_trajectory_log VALUES (?,?,?,?,?,?,?,?,?)", [(x["day"], x["agent_id"], x["episode_id"], x["sequence"], x["edge_id"], x["enter_time"], x["exit_time"], x["status"], x.get("reroute_id")) for x in result.executed])
        c.executemany("INSERT INTO observation_event_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [(
            o.event_id, o.day, o.event_time, o.agent_id, o.episode_id, o.object_id,
            o.object_type, str(o.source_channel), o.observed_attribute, o.observed_value,
            o.source_reliability, o.salience, o.distance_to_object, o.observer_x, o.observer_y,
            o.heading_deg, o.relative_angle_deg, None if o.line_of_sight is None else int(o.line_of_sight),
            o.detection_probability, o.detection_draw, o.source_event_id,
            o.independence_group, o.source_agent_id,
            o.copresence_event_id,
            None if o.first_hand_source is None else int(o.first_hand_source),
        ) for o in result.observations])
        c.executemany("INSERT INTO perception_diagnostic_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [(
            x["diagnostic_id"], x["day"], x["agent_id"], x["episode_id"], x["object_id"],
            x["object_type"], int(x["detected"]), x["probability"], x["draw"], x["distance"],
            x["relative_angle_deg"], int(x["line_of_sight"]), x["observer_x"], x["observer_y"],
            x["heading_deg"], x["sample_time"],
        ) for x in result.perception_diagnostics])
        c.executemany("INSERT INTO temporary_memory_log VALUES (?,?,?,?,?,?,?)", [(x["day"], x["agent_id"], x["episode_id"], x["object_id"], x["attribute"], x["value"], x["time"]) for x in result.temporary_memories])
        c.executemany("INSERT INTO information_item VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [(
            item.information_id, item.day, item.agent_id, item.object_id,
            item.object_type, item.attribute_name, item.observed_value, str(item.source_channel),
            item.source_event_id, item.reliability, item.salience,
            item.independence_group, item.source_agent_id,
            item.copresence_event_id,
            None if item.first_hand_source is None else int(item.first_hand_source),
        ) for item in result.information_items])
        c.executemany("INSERT INTO cognitive_detection_log VALUES (?,?,?,?,?,?,?,?,?,?,?)", [(
            item.detection_id, item.day, item.event_time, item.agent_id, item.object_id,
            item.object_type, item.prior_value, item.observed_value,
            str(item.source_channel), item.source_event_id, int(item.first_detection),
        ) for item in result.detections])
        c.executemany("INSERT INTO cognitive_update_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [(
            u.update_id, u.day, u.event_time, u.agent_id, u.object_id, u.object_type,
            u.attribute_name, u.old_value, u.new_value, u.update_reason,
            u.source_event_id, u.source_agent_id,
            None if u.first_hand_source is None else int(u.first_hand_source),
        ) for u in result.updates])
        c.executemany("INSERT OR IGNORE INTO knowledge_update_event VALUES (?,?,?,?,?,?,?)", [(
            u.day, u.event_time, u.agent_id, u.object_id, u.object_type,
            u.update_id, u.update_reason,
        ) for u in result.updates])
        c.executemany("INSERT INTO llm_decision_log VALUES (?,?,?,?,?,?,?,?,?,?)", [(x["call_id"], x["simulation_day"], x["agent_id"], x["phase"], x["provider"], x["model"], json.dumps(x["request_json"], ensure_ascii=False), json.dumps(x["response_json"], ensure_ascii=False), x["latency_ms"], x["status"]) for x in result.llm_decisions])
        c.executemany("INSERT INTO activity_stop_log VALUES (?,?,?,?,?,?,?)", [(
            item.day, item.agent_id, item.episode_id, item.place_id,
            item.activity_type, item.start_time, item.end_time,
        ) for item in result.activity_stops])
        c.executemany("INSERT INTO copresence_event VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [(
            item.event_id, item.day, item.agent_a, item.agent_b, item.place_id,
            item.overlap_start, item.overlap_end, item.overlap_minutes,
            item.relationship_strength, item.conversation_probability,
            item.conversation_draw, int(item.conversation_flag),
        ) for item in result.copresence_events])
        c.executemany("INSERT INTO interpersonal_exchange VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [(
            item.exchange_id, item.day, item.copresence_event_id,
            item.speaker_agent_id, item.listener_agent_id, item.place_id,
            item.report_id, item.object_id, item.object_type,
            item.observed_attribute, item.observed_value, item.source_event_id,
            item.independence_group, int(item.first_hand_source),
            int(item.delivered),
        ) for item in result.interpersonal_exchanges])
        self.save_cognitive_snapshot(result.day, beliefs)
        self._aggregate(result, beliefs, graph, groups)
        self._research_metrics(result, beliefs, graph)
        c.commit()

    def _aggregate(self, result: DayResult, beliefs: dict[str, CognitiveBeliefGraph], graph: Any, groups: list[str]) -> None:
        c = self.connection
        planned: dict[str, set[str]] = {e: set() for e in graph.edges}
        actual: dict[str, set[str]] = {e: set() for e in graph.edges}
        failures = {e: 0 for e in graph.edges}
        reroutes = {e: 0 for e in graph.edges}
        for x in result.planned:
            planned[x["edge_id"]].add(x["agent_id"])
        for x in result.executed:
            actual[x["edge_id"]].add(x["agent_id"])
            failures[x["edge_id"]] += int(x["status"] == "BLOCKED")
            reroutes[x["edge_id"]] += int(bool(x.get("reroute_id")))
        road_rows = []
        for eid in graph.edges:
            previous = c.execute("SELECT COUNT(DISTINCT agent_id) FROM executed_trajectory_log WHERE simulation_day < ? AND edge_id=?", (result.day, eid)).fetchone()[0]
            road_rows.append((result.day, eid, len(planned[eid]), len(actual[eid]), failures[eid], reroutes[eid], len(actual[eid]) if previous == 0 else 0, len(actual[eid]) if previous else 0))
        c.executemany("INSERT INTO road_daily_aggregate VALUES (?,?,?,?,?,?,?,?)", road_rows)

        episode_by_aoi: dict[str, list[dict[str, Any]]] = {a: [] for a in graph.aois}
        for episode in result.episodes:
            if episode.get("selected_destination_id") in episode_by_aoi:
                episode_by_aoi[episode["selected_destination_id"]].append(episode)
        aoi_rows = []
        for aid, visits in episode_by_aoi.items():
            agents = {v["agent_id"] for v in visits}
            successes = sum(bool(v.get("activity_success")) for v in visits)
            mismatch = sum(v.get("failure_reason") == "FUNCTION_MISMATCH" for v in visits)
            first = sum(beliefs[v["agent_id"]].aois.get(aid) is not None and beliefs[v["agent_id"]].aois[aid].visit_count == 1 for v in visits)
            avg = sum(v.get("satisfaction", 0) for v in visits) / len(visits) if visits else 0
            aoi_rows.append((result.day, aid, len(visits), len(agents), successes, mismatch, first, len(visits) - first, avg))
        c.executemany("INSERT INTO aoi_daily_aggregate VALUES (?,?,?,?,?,?,?,?,?)", aoi_rows)

        profile_groups = {r["agent_id"]: r["group_id"] for r in c.execute("SELECT agent_id, group_id FROM agent_profile")}
        states = {s.agent_id: s for s in result.agent_states}
        group_rows = []
        targets = (
            [("ROAD_EDGE", eid) for eid, e in graph.edges.items() if e.is_intervention]
            + [("AOI", aid) for aid, a in graph.aois.items() if a.is_intervention]
            + [("ENTRANCE", item.entrance_id) for item in graph.entrances.values() if item.is_intervention]
        )
        for group in groups:
            members = [aid for aid, g in profile_groups.items() if g == group]
            statuses: list[str] = []
            correct, total = 0, 0
            for aid in members:
                belief = beliefs[aid]
                for object_type, target in targets:
                    b = (
                        belief.edges.get(target) if object_type == "ROAD_EDGE"
                        else belief.aois.get(target) if object_type == "AOI"
                        else belief.entrances.get(target)
                    )
                    if b:
                        statuses.append(str(getattr(b, "adoption_state", "AWARE")))
                for eid, b in belief.edges.items():
                    total += 1
                    correct += int(str(b.believed_status) == str(graph.edges[eid].status))
            n = max(1, len(members))
            awareness = sum(x != "UNKNOWN" for x in statuses) / max(1, len(targets) * n)
            trial = sum(x == "TRIED" for x in statuses) / max(1, len(targets) * n)
            adoption = sum(x == "ADOPTED" for x in statuses) / max(1, len(targets) * n)
            rejection = sum(x == "REJECTED" for x in statuses) / max(1, len(targets) * n)
            travel = sum(states[a].total_travel_time for a in members) / n
            failure = sum(states[a].failed_route_count for a in members) / n
            group_rows.append((result.day, group, awareness, trial, adoption, rejection, travel, failure, correct / max(1, total)))
        c.executemany("INSERT INTO group_daily_aggregate VALUES (?,?,?,?,?,?,?,?,?)", group_rows)

        population = max(1, c.execute("SELECT COUNT(*) FROM agent_profile").fetchone()[0])
        operation_by_target = {
            row["target_id"]: row["operation_type"]
            for row in c.execute("SELECT target_id, operation_type FROM intervention_log")
        }
        mismatch_rows = []
        for object_type, object_id in targets:
            aware = 0
            if object_type == "ROAD_EDGE":
                actual_value = str(graph.edges[object_id].status)
                for belief in beliefs.values():
                    item = belief.edges.get(object_id)
                    aware += int(item is not None and str(item.believed_status) in {
                        actual_value,
                        "POSSIBLY_OPEN" if actual_value == "OPEN" else "POSSIBLY_CLOSED",
                    })
                realized = len(actual.get(object_id, set())) / population
            elif object_type == "AOI":
                actual_value = graph.aois[object_id].function
                for belief in beliefs.values():
                    item = belief.aois.get(object_id)
                    aware += int(item is not None and item.believed_function == actual_value)
                realized = len({
                    item["agent_id"] for item in episode_by_aoi.get(object_id, [])
                    if item.get("activity_success")
                }) / population
            else:
                actual_value = str(graph.entrances[object_id].status)
                for belief in beliefs.values():
                    item = belief.entrances.get(object_id)
                    aware += int(item is not None and str(item.believed_status) in {
                        actual_value,
                        "POSSIBLY_OPEN" if actual_value == "OPEN" else "POSSIBLY_CLOSED",
                    })
                realized = 0.0
            objective = 1.0 if operation_by_target.get(object_id) in {
                "ROAD_OPEN", "AOI_FUNCTION_CHANGE", "ENTRANCE_OPEN",
            } else 0.0
            awareness_rate = aware / population
            mismatch_rows.append((
                result.day, object_id, object_type, objective, awareness_rate,
                min(1.0, realized), max(0.0, objective - min(1.0, realized)),
            ))
        c.executemany("INSERT INTO mismatch_daily VALUES (?,?,?,?,?,?,?)", mismatch_rows)

    def _research_metrics(
        self, result: DayResult, beliefs: dict[str, CognitiveBeliefGraph], graph: Any,
    ) -> None:
        c = self.connection
        active = [dict(row) for row in c.execute(
            "SELECT * FROM intervention_log WHERE effective_day<=? ORDER BY effective_day, intervention_id",
            (result.day,),
        )]
        if not active:
            self.save_objective_accessibility(result.day, graph, self._profiles_from_database())
            self._save_process_and_route_metrics(result, graph, set())
            return

        profile_json = {
            row["agent_id"]: json.loads(row["profile_json"])
            for row in c.execute("SELECT agent_id, profile_json FROM agent_profile")
        }
        lag_rows = []
        for intervention in active:
            object_type = intervention["target_type"]
            object_id = intervention["target_id"]
            actual = self._actual_value(graph, object_type, object_id)
            for agent_id, belief in beliefs.items():
                believed = self._belief_value(belief, object_type, object_id)
                if believed is None or believed == "UNKNOWN":
                    mismatch_class = "UNKNOWN"
                elif believed == actual:
                    mismatch_class = "CORRECT"
                elif believed == str(intervention["before_value"]):
                    mismatch_class = "OBSOLETE"
                else:
                    mismatch_class = "INCORRECT"
                baseline_state = self._belief_before_intervention(
                    agent_id, object_type, object_id, int(intervention["effective_day"]),
                )
                baseline_known = int(baseline_state not in {None, "UNKNOWN"})
                lag_indicator = 0.0 if mismatch_class == "CORRECT" else 1.0
                lag_rows.append((
                    result.day, agent_id, intervention["intervention_id"], object_id,
                    object_type, intervention["effective_day"], actual,
                    believed or "UNKNOWN", mismatch_class,
                    baseline_state or "UNKNOWN", baseline_known, lag_indicator,
                ))
        c.executemany(
            "INSERT OR REPLACE INTO cognitive_lag_daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            lag_rows,
        )

        population_rows = []
        metric_rows = []
        for intervention in active:
            rows = [
                row for row in lag_rows
                if row[2] == intervention["intervention_id"]
            ]
            n = max(1, len(rows))
            rates = {
                state: sum(row[8] == state for row in rows) / n
                for state in ("CORRECT", "UNKNOWN", "OBSOLETE", "INCORRECT")
            }
            mean_lag = sum(row[11] for row in rows) / n
            t50 = self._threshold_day(intervention["intervention_id"], 0.5, result.day, rates["CORRECT"])
            t80 = self._threshold_day(intervention["intervention_id"], 0.8, result.day, rates["CORRECT"])
            population_rows.append((
                result.day, intervention["intervention_id"], intervention["target_id"],
                intervention["target_type"], rates["CORRECT"], rates["UNKNOWN"],
                rates["OBSOLETE"], rates["INCORRECT"], mean_lag, t50, t80,
            ))
            for metric_name, value in (
                ("correct_update_rate", rates["CORRECT"]),
                ("unknown_rate", rates["UNKNOWN"]),
                ("obsolete_rate", rates["OBSOLETE"]),
                ("incorrect_rate", rates["INCORRECT"]),
                ("cognitive_lag_index", mean_lag),
            ):
                metric_rows.append((
                    result.day, "RQ1", "INTERVENTION",
                    intervention["intervention_id"], metric_name, value,
                ))
        c.executemany(
            "INSERT OR REPLACE INTO cognitive_lag_population_daily VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            population_rows,
        )

        for agent_id in beliefs:
            rows = [row for row in lag_rows if row[1] == agent_id]
            lag = sum(row[11] for row in rows) / max(1, len(rows))
            metric_rows.append((result.day, "RQ1", "AGENT", agent_id, "cognitive_lag_index", lag))
            metric_rows.append((
                result.day, "RQ1", "AGENT", agent_id, "correct_share",
                sum(row[8] == "CORRECT" for row in rows) / max(1, len(rows)),
            ))
        groups = sorted({value["group_id"] for value in profile_json.values()})
        for group in groups:
            members = {agent_id for agent_id, value in profile_json.items() if value["group_id"] == group}
            rows = [row for row in lag_rows if row[1] in members]
            lag = sum(row[11] for row in rows) / max(1, len(rows))
            metric_rows.append((result.day, "RQ1", "GROUP", group, "cognitive_lag_index", lag))
        c.executemany("INSERT OR REPLACE INTO daily_metric VALUES (?,?,?,?,?,?)", metric_rows)

        self._update_timing(active, beliefs, graph, profile_json, result.day)
        self._save_process_and_route_metrics(result, graph, {row["target_id"] for row in active})
        self.save_objective_accessibility(result.day, graph, self._profiles_from_database())

    def _save_process_and_route_metrics(
        self, result: DayResult, graph: Any, active_targets: set[str],
    ) -> None:
        c = self.connection
        agent_ids = [row[0] for row in c.execute("SELECT agent_id FROM agent_profile ORDER BY agent_id")]
        process_rows = []
        route_rows = []
        metric_rows = []
        for agent_id in agent_ids:
            observations = [item for item in result.observations if item.agent_id == agent_id]
            visual = sum(str(item.source_channel) == "VISUAL_SEARCH" for item in observations)
            experience = sum(str(item.source_channel) in {"DIRECT_EXPERIENCE", "ROUTE_FAILURE"} for item in observations)
            interpersonal = sum(str(item.source_channel) == "INTERPERSONAL" for item in observations)
            copresence = [
                item for item in result.copresence_events
                if agent_id in {item.agent_a, item.agent_b}
            ]
            conversations = sum(item.conversation_flag for item in copresence)
            delivered = sum(
                item.delivered for item in result.interpersonal_exchanges
                if item.listener_agent_id == agent_id
            )
            firsthand_reports = sum(
                item.delivered and item.first_hand_source
                for item in result.interpersonal_exchanges
                if item.listener_agent_id == agent_id
            )
            reached_targets = {
                item.object_id for item in observations if item.object_id in active_targets
            }
            unique_targets = len(reached_targets)
            coverage = 0.0 if not active_targets else unique_targets / len(active_targets)
            process_rows.append((
                result.day, agent_id, visual, experience, interpersonal,
                len(copresence), conversations, delivered, firsthand_reports,
                unique_targets, coverage,
            ))
            for name, value in (
                ("visual_information_count", visual),
                ("direct_experience_count", experience),
                ("interpersonal_information_count", interpersonal),
                ("copresence_count", len(copresence)),
                ("conversation_count", conversations),
                ("delivered_report_count", delivered),
                ("firsthand_report_count", firsthand_reports),
                ("unique_changed_target_count", unique_targets),
                ("information_coverage", coverage),
            ):
                metric_rows.append((result.day, "RQ2", "AGENT", agent_id, name, float(value)))

            planned = [item for item in result.planned if item["agent_id"] == agent_id]
            executed = [item for item in result.executed if item["agent_id"] == agent_id]
            planned_set = {item["edge_id"] for item in planned}
            executed_set = {item["edge_id"] for item in executed if item["status"] != "BLOCKED"}
            union = planned_set | executed_set
            difference = 0.0 if not union else 1.0 - len(planned_set & executed_set) / len(union)
            planned_distance = sum(graph.edges[item["edge_id"]].length for item in planned)
            executed_distance = sum(
                graph.edges[item["edge_id"]].length for item in executed
                if item["status"] != "BLOCKED"
            )
            state = next((item for item in result.agent_states if item.agent_id == agent_id), None)
            episodes = [item for item in result.episodes if item["agent_id"] == agent_id]
            success_rate = sum(bool(item.get("activity_success")) for item in episodes) / max(1, len(episodes))
            intervention_uses = sum(
                item["edge_id"] in active_targets and item["status"] != "BLOCKED"
                for item in executed
            ) + sum(
                item.get("selected_destination_id") in active_targets and item.get("activity_success")
                for item in episodes
            )
            route_rows.append((
                result.day, agent_id, len(planned), len(executed), difference,
                planned_distance, executed_distance, max(0.0, executed_distance - planned_distance),
                0 if state is None else state.failed_route_count,
                0 if state is None else state.reroute_count,
                success_rate, intervention_uses,
            ))
            for name, value in (
                ("route_difference_ratio", difference),
                ("detour_distance", max(0.0, executed_distance - planned_distance)),
                ("route_failures", 0 if state is None else state.failed_route_count),
                ("activity_success_rate", success_rate),
                ("intervention_use_count", intervention_uses),
            ):
                metric_rows.append((result.day, "RQ3", "AGENT", agent_id, name, float(value)))
        c.executemany(
            "INSERT OR REPLACE INTO information_process_daily VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            process_rows,
        )
        c.executemany(
            "INSERT OR REPLACE INTO route_outcome_daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            route_rows,
        )

        by_activity: dict[str, list[dict[str, Any]]] = {}
        for episode in result.episodes:
            if episode.get("selected_destination_id"):
                by_activity.setdefault(episode["activity_type"], []).append(episode)
        activity_rows = []
        for activity_type, episodes in by_activity.items():
            total = max(1, len(episodes))
            destination_ids = sorted({item["selected_destination_id"] for item in episodes})
            for destination_id in destination_ids:
                items = [item for item in episodes if item["selected_destination_id"] == destination_id]
                activity_rows.append((
                    result.day, activity_type, destination_id, len(items),
                    sum(bool(item.get("activity_success")) for item in items), len(items) / total,
                ))
        c.executemany(
            "INSERT OR REPLACE INTO activity_distribution_daily VALUES (?,?,?,?,?,?)",
            activity_rows,
        )
        c.executemany("INSERT OR REPLACE INTO daily_metric VALUES (?,?,?,?,?,?)", metric_rows)

    def _update_timing(
        self,
        active: list[dict[str, Any]],
        beliefs: dict[str, CognitiveBeliefGraph],
        graph: Any,
        profile_json: dict[str, dict[str, Any]],
        day: int,
    ) -> None:
        c = self.connection
        rows = []
        for intervention in active:
            object_id = intervention["target_id"]
            object_type = intervention["target_type"]
            for agent_id in beliefs:
                def first_channel(channels: tuple[str, ...]) -> int | None:
                    placeholders = ",".join("?" for _ in channels)
                    row = c.execute(
                        f"SELECT MIN(simulation_day) FROM observation_event_log "
                        f"WHERE agent_id=? AND object_id=? AND simulation_day>=? "
                        f"AND observed_value=? "
                        f"AND source_channel IN ({placeholders})",
                        (
                            agent_id, object_id, intervention["effective_day"],
                            str(intervention["after_value"]), *channels,
                        ),
                    ).fetchone()
                    return None if row is None else row[0]

                first_visual = first_channel(("VISUAL_SEARCH",))
                first_experience = first_channel(("DIRECT_EXPERIENCE", "ROUTE_FAILURE"))
                first_interpersonal = first_channel(("INTERPERSONAL",))
                first_information_row = c.execute(
                    "SELECT simulation_day, source_channel FROM observation_event_log "
                    "WHERE agent_id=? AND object_id=? AND simulation_day>=? AND observed_value=? "
                    "ORDER BY simulation_day, event_time LIMIT 1",
                    (
                        agent_id, object_id, intervention["effective_day"],
                        str(intervention["after_value"]),
                    ),
                ).fetchone()
                first_information = None if first_information_row is None else first_information_row[0]
                first_channel_name = None if first_information_row is None else first_information_row[1]
                first_correct = self._first_correct_day(
                    agent_id, object_type, object_id, str(intervention["after_value"]),
                    int(intervention["effective_day"]),
                )
                first_use = self._first_use_day(
                    agent_id, object_type, object_id, int(intervention["effective_day"]),
                )
                profile = profile_json[agent_id]
                intervention_day = int(intervention["effective_day"])
                information_delay = (
                    None if first_information is None else first_information - intervention_day
                )
                behavior_delay = (
                    first_use - first_correct
                    if first_use is not None and first_correct is not None and first_use >= first_correct
                    else None
                )
                obsolete_duration = (
                    first_correct - intervention_day
                    if first_correct is not None
                    else day - intervention_day + 1
                )
                baseline_state = self._belief_before_intervention(
                    agent_id, object_type, object_id, intervention_day,
                ) or "UNKNOWN"
                rows.append((
                    intervention["intervention_id"], agent_id, object_id, object_type,
                    intervention["effective_day"], first_visual, first_experience,
                    first_interpersonal, first_information, first_correct, first_use,
                    information_delay, behavior_delay, obsolete_duration,
                    int(first_correct is None), first_channel_name, baseline_state,
                    profile["route_habit"], profile["visual_attention"],
                    profile["social_participation"], profile["age"],
                ))
        c.executemany(
            "INSERT OR REPLACE INTO update_timing VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )

    def _first_correct_day(
        self, agent_id: str, object_type: str, object_id: str, correct_value: str, start_day: int,
    ) -> int | None:
        table, id_column, value_column = {
            "ROAD_EDGE": ("cognitive_edge_daily_state", "edge_id", "believed_status"),
            "AOI": ("cognitive_aoi_daily_state", "aoi_id", "believed_function"),
            "ENTRANCE": ("cognitive_entrance_daily_state", "entrance_id", "believed_status"),
        }[object_type]
        row = self.connection.execute(
            f"SELECT MIN(simulation_day) FROM {table} WHERE agent_id=? AND {id_column}=? "
            f"AND simulation_day>=? AND {value_column}=?",
            (agent_id, object_id, start_day, correct_value),
        ).fetchone()
        return None if row is None else row[0]

    def _first_use_day(
        self, agent_id: str, object_type: str, object_id: str, start_day: int,
    ) -> int | None:
        if object_type == "ROAD_EDGE":
            row = self.connection.execute(
                "SELECT MIN(simulation_day) FROM executed_trajectory_log "
                "WHERE agent_id=? AND edge_id=? AND simulation_day>=? AND execution_status!='BLOCKED'",
                (agent_id, object_id, start_day),
            ).fetchone()
        elif object_type == "AOI":
            row = self.connection.execute(
                "SELECT MIN(simulation_day) FROM activity_episode_log "
                "WHERE agent_id=? AND selected_destination_id=? AND simulation_day>=? AND activity_success=1",
                (agent_id, object_id, start_day),
            ).fetchone()
        else:
            return None
        return None if row is None else row[0]

    def _belief_before_intervention(
        self, agent_id: str, object_type: str, object_id: str, intervention_day: int,
    ) -> str | None:
        table, id_column, value_column = {
            "ROAD_EDGE": ("cognitive_edge_daily_state", "edge_id", "believed_status"),
            "AOI": ("cognitive_aoi_daily_state", "aoi_id", "believed_function"),
            "ENTRANCE": ("cognitive_entrance_daily_state", "entrance_id", "believed_status"),
        }[object_type]
        snapshot_day = 0 if intervention_day <= 0 else intervention_day - 1
        row = self.connection.execute(
            f"SELECT {value_column} FROM {table} WHERE agent_id=? AND {id_column}=? "
            f"AND simulation_day<=? ORDER BY simulation_day DESC LIMIT 1",
            (agent_id, object_id, snapshot_day),
        ).fetchone()
        return None if row is None or row[0] is None else str(row[0])

    def _threshold_day(
        self, intervention_id: str, threshold: float, day: int, current_rate: float,
    ) -> int | None:
        row = self.connection.execute(
            "SELECT MIN(simulation_day) FROM cognitive_lag_population_daily "
            "WHERE intervention_id=? AND correct_update_rate>=?",
            (intervention_id, threshold),
        ).fetchone()
        if row and row[0] is not None:
            return int(row[0])
        return day if current_rate >= threshold else None

    @staticmethod
    def _actual_value(graph: Any, object_type: str, object_id: str) -> str:
        if object_type == "ROAD_EDGE":
            return str(graph.edges[object_id].status)
        if object_type == "AOI":
            return str(graph.aois[object_id].function)
        if object_type == "ENTRANCE":
            return str(graph.entrances[object_id].status)
        raise ValueError(f"Unsupported cognitive object: {object_type}/{object_id}")

    @staticmethod
    def _belief_value(
        belief: CognitiveBeliefGraph, object_type: str, object_id: str,
    ) -> str | None:
        if object_type == "ROAD_EDGE":
            item = belief.edges.get(object_id)
            return None if item is None else str(item.believed_status)
        if object_type == "AOI":
            item = belief.aois.get(object_id)
            return None if item is None else item.believed_function
        item = belief.entrances.get(object_id)
        return None if item is None else str(item.believed_status)

    def _profiles_from_database(self) -> list[Any]:
        from types import SimpleNamespace
        return [
            SimpleNamespace(home_aoi_id=row[0])
            for row in self.connection.execute("SELECT DISTINCT home_aoi_id FROM agent_profile")
        ]

    def finalize_research_outputs(self) -> Path | None:
        config_row = self.connection.execute("SELECT config_json FROM run_manifest LIMIT 1").fetchone()
        if config_row is None or not json.loads(config_row[0]).get("export_research_csv", True):
            self.connection.commit()
            return None
        output_dir = self.path.parent / "research_outputs" / self.path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        exports = {
            "rq1_cognitive_lag.csv": "SELECT * FROM cognitive_lag_daily ORDER BY simulation_day, agent_id, intervention_id",
            "rq1_population_update_curve.csv": "SELECT * FROM cognitive_lag_population_daily ORDER BY simulation_day, intervention_id",
            "rq2_update_timing.csv": "SELECT * FROM update_timing ORDER BY intervention_id, agent_id",
            "rq2_information_process.csv": "SELECT * FROM information_process_daily ORDER BY simulation_day, agent_id",
            "rq2_copresence_events.csv": "SELECT * FROM copresence_event ORDER BY simulation_day, overlap_start",
            "rq2_interpersonal_exchange.csv": "SELECT * FROM interpersonal_exchange ORDER BY simulation_day, listener_agent_id",
            "rq3_route_outcomes.csv": "SELECT * FROM route_outcome_daily ORDER BY simulation_day, agent_id",
            "rq3_activity_distribution.csv": "SELECT * FROM activity_distribution_daily ORDER BY simulation_day, activity_type, aoi_id",
            "rq3_objective_accessibility.csv": "SELECT * FROM objective_accessibility_daily ORDER BY simulation_day, origin_aoi_id, activity_type",
            "daily_metrics_long.csv": "SELECT * FROM daily_metric ORDER BY research_question, simulation_day, level_type, level_id, metric_name",
        }
        for filename, query in exports.items():
            cursor = self.connection.execute(query)
            with (output_dir / filename).open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow([item[0] for item in cursor.description])
                writer.writerows(cursor)
        dictionary = {
            "RQ1": "真实地图改变后，有限信息会造成多长的认知滞后？",
            "RQ2": "日常路线与线下相遇如何决定新信息何时到达？",
            "RQ3": "这种信息延迟会在多大程度上改变活动和路线预测？",
            "cognition_rule": "视觉、亲历或面对面信息到达后，当日写入二元认知地图，次日参与行为决策。",
            "source_channels": ["VISUAL_SEARCH", "DIRECT_EXPERIENCE", "ROUTE_FAILURE", "INTERPERSONAL"],
            "database": self.path.name,
        }
        (output_dir / "README.json").write_text(
            json.dumps(dictionary, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        self.connection.commit()
        return output_dir


def query_database(path: str | Path, query: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(query, tuple(params))]
    finally:
        connection.close()
