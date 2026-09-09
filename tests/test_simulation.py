from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from cogmap_sim import SimulationEngine, build_demo_scenario
from cogmap_sim.llm import DeterministicLLMDriver
from cogmap_sim.models import InterventionEvent
from cogmap_sim.analysis import compare_model_databases
from cogmap_sim.web import (
    _json_safe,
    cognitive_payload,
    collective_payload,
    dashboard_payload,
    export_dashboard_gallery,
    list_dashboard_databases,
)


class SimulationTests(unittest.TestCase):
    def test_dashboard_json_replaces_non_finite_values(self) -> None:
        self.assertEqual(
            _json_safe({"distance": float("inf"), "values": [1.0, float("nan")]}),
            {"distance": None, "values": [1.0, None]},
        )

    def run_small(self, model: str = "cognitive") -> Path:
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / "experiment.db"
        config, graph, agents, interventions = build_demo_scenario(days=4, agent_count=12, seed=7, model_type=model)
        config.behavior_driver = "deterministic"
        SimulationEngine(config, graph, agents, interventions, path, DeterministicLLMDriver()).run()
        return path

    def tearDown(self) -> None:
        if hasattr(self, "temp"):
            self.temp.cleanup()

    def test_information_chain_and_same_day_updates(self) -> None:
        path = self.run_small()
        with sqlite3.connect(path) as c:
            counts = {
                table: c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "activity_episode_log", "planned_trajectory_log", "executed_trajectory_log",
                    "observation_event_log", "cognitive_update_log", "information_item",
                )
            }
        self.assertGreater(counts["activity_episode_log"], 100)
        for table in counts:
            self.assertGreater(counts[table], 0, table)
        with sqlite3.connect(path) as c:
            delayed_writes = c.execute(
                "SELECT COUNT(*) FROM cognitive_update_log u "
                "JOIN observation_event_log o ON u.source_event_id=o.event_id "
                "WHERE u.simulation_day<>o.simulation_day"
            ).fetchone()[0]
            edge_columns = {row[1] for row in c.execute("PRAGMA table_info(cognitive_edge_daily_state)")}
        self.assertEqual(delayed_writes, 0)
        self.assertTrue({"confidence", "uncertainty", "familiarity"}.isdisjoint(edge_columns))

    def test_intervention_does_not_retroactively_change_t0_belief(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / "non_retroactive.db"
        config, graph, agents, _ = build_demo_scenario(days=3, agent_count=3, seed=7)
        config.behavior_driver = "deterministic"
        intervention = InterventionEvent(
            "INT_LATE_CLOSE", 2, "ROAD_CLOSE", "ROAD_EDGE", "H00", "OPEN", "CLOSED", 0.8, 0.0,
        )
        SimulationEngine(config, graph, agents, [intervention], path, DeterministicLLMDriver()).run()
        with sqlite3.connect(path) as c:
            actual_day_2 = c.execute(
                "SELECT actual_value FROM environment_daily_state WHERE simulation_day=2 AND object_id='H00'"
            ).fetchone()[0]
            day_1_beliefs = c.execute(
                "SELECT DISTINCT believed_status FROM cognitive_edge_daily_state WHERE simulation_day=1 AND edge_id='H00'"
            ).fetchall()
        self.assertEqual(actual_day_2, "CLOSED")
        self.assertIn(("OPEN",), day_1_beliefs)

    def test_aoi_function_change_applies_on_its_effective_day(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / "day_two.db"
        config, graph, agents, interventions = build_demo_scenario(days=3, agent_count=3, seed=7)
        config.behavior_driver = "deterministic"
        aoi_change = next(event for event in interventions if event.operation_type == "AOI_FUNCTION_CHANGE")
        aoi_change.effective_day = 2
        SimulationEngine(config, graph, agents, [aoi_change], path, DeterministicLLMDriver()).run()
        c = sqlite3.connect(path)
        day_1 = c.execute("SELECT actual_value FROM environment_daily_state WHERE simulation_day=1 AND object_id='PLAZA_NEW'").fetchone()[0]
        day_2 = c.execute("SELECT actual_value FROM environment_daily_state WHERE simulation_day=2 AND object_id='PLAZA_NEW'").fetchone()[0]
        self.assertEqual(day_1, "INDUSTRIAL")
        self.assertEqual(day_2, "PARK")
        c.close()

    def test_dashboard_payload(self) -> None:
        path = self.run_small()
        data = dashboard_payload(path)
        self.assertEqual(data["metrics"]["agents"], 12)
        self.assertEqual(len(data["groups"]), 4 * 3)
        self.assertTrue(any(e["edge_id"] == "NEW_LINK" for e in data["edges"]))
        self.assertTrue(data["database_path"].endswith("experiment.db"))
        cognitive = cognitive_payload(path, "A001", 2)
        self.assertEqual(cognitive["agent_id"], "A001")
        self.assertEqual(cognitive["day"], 2)
        self.assertEqual(len(cognitive["edges"]), len(data["edges"]))
        self.assertGreaterEqual(len(cognitive["llm"]), 1)
        self.assertIn("cognitive_lag", cognitive)
        self.assertIn("information", cognitive)
        self.assertIn("copresence", cognitive)
        collective = collective_payload(path, 2)
        self.assertEqual(collective["day"], 2)
        self.assertEqual(len(collective["agents"]), 12)
        self.assertGreater(len(collective["executed"]), 0)
        self.assertTrue(all("home_node" in agent for agent in collective["agents"]))

    def test_standalone_dashboard_gallery(self) -> None:
        path = self.run_small()
        databases = list_dashboard_databases(path.parent)
        self.assertEqual([item["name"] for item in databases], ["experiment.db"])
        selector, reports = export_dashboard_gallery(path.parent, path)
        self.assertTrue(selector.exists())
        self.assertEqual(len(reports), 1)
        selector_text = selector.read_text(encoding="utf-8")
        report_text = reports[0].read_text(encoding="utf-8")
        self.assertIn("experiment.db", selector_text)
        self.assertIn("experiment_dashboard.html", selector_text)
        self.assertIn("window.COGMAP_EMBEDDED", report_text)
        self.assertIn("function drawWalker", report_text)
        self.assertIn("function drawVisionFan", report_text)
        self.assertIn("showVisionCone", report_text)
        self.assertIn("visual_field_of_view_deg", report_text)
        self.assertIn("requestAnimationFrame(animationLoop)", report_text)
        self.assertNotIn('id="collectiveTimeSlider"', report_text)
        self.assertIn("function restartCollectiveAnimation", report_text)
        self.assertIn("requestAnimationFrame(collectiveAnimationLoop)", report_text)
        self.assertIn("function drawCollective", report_text)
        self.assertIn("function buildCollectiveSegments", report_text)
        self.assertIn('"collective":', report_text)
        self.assertNotIn("function arrowHead", report_text)
        self.assertIn("mode==='cognitive'?a.believed_function", report_text)
        self.assertIn("known,false,'cognitive'", report_text)
        self.assertIn("handleMapClick('actual'", report_text)
        self.assertIn('id="actualFeature"', report_text)
        self.assertIn("function isPOI", report_text)
        self.assertIn("function drawPOI", report_text)
        self.assertIn("base.aois.filter(isPOI)", report_text)
        self.assertIn("function functionLegendHTML", report_text)
        self.assertIn('id="actualFunctionLegend"', report_text)
        self.assertIn('id="cognitiveFunctionLegend"', report_text)
        self.assertIn("functionStrokes", report_text)
        self.assertIn("AOI 现实功能", report_text)
        self.assertIn("AOI 认知功能", report_text)
        self.assertIn("三项研究问题的数据闭环", report_text)
        self.assertIn("function renderResearchQuestions", report_text)
        self.assertIn("function drawMeetingEvents", report_text)
        self.assertIn("INTERPERSONAL", report_text)
        self.assertNotIn("SOCIAL_MEDIA", report_text)

    def test_llm_and_face_to_face_information(self) -> None:
        path = self.run_small()
        c = sqlite3.connect(path)
        self.assertGreaterEqual(c.execute("SELECT COUNT(*) FROM llm_decision_log").fetchone()[0], 4 * 12)
        self.assertGreater(c.execute("SELECT COUNT(*) FROM copresence_event").fetchone()[0], 0)
        self.assertGreater(c.execute("SELECT COUNT(*) FROM interpersonal_exchange").fetchone()[0], 0)
        self.assertIsNone(c.execute("SELECT name FROM sqlite_master WHERE name='social_media_post'").fetchone())
        channels = {row[0] for row in c.execute("SELECT DISTINCT source_channel FROM observation_event_log")}
        self.assertTrue(channels <= {"VISUAL_SEARCH", "DIRECT_EXPERIENCE", "ROUTE_FAILURE", "INTERPERSONAL"})
        self.assertGreater(c.execute("SELECT COUNT(*) FROM information_item").fetchone()[0], 0)
        self.assertEqual(c.execute("SELECT COUNT(*) FROM interpersonal_exchange WHERE delivered<>1").fetchone()[0], 0)
        c.close()

    def test_research_question_outputs_are_materialized(self) -> None:
        path = self.run_small()
        c = sqlite3.connect(path)
        expected = (
            "cognitive_lag_daily", "cognitive_lag_population_daily", "update_timing",
            "information_process_daily", "route_outcome_daily",
            "activity_distribution_daily", "objective_accessibility_daily", "daily_metric",
        )
        for table in expected:
            self.assertGreater(c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0, table)
        questions = {row[0] for row in c.execute("SELECT DISTINCT research_question FROM daily_metric")}
        self.assertEqual(questions, {"RQ1", "RQ2", "RQ3"})
        c.close()

    def test_b0_has_no_agent_behavior_or_llm_calls(self) -> None:
        path = self.run_small("objective_accessibility")
        c = sqlite3.connect(path)
        self.assertEqual(c.execute("SELECT COUNT(*) FROM activity_episode_log").fetchone()[0], 0)
        self.assertEqual(c.execute("SELECT COUNT(*) FROM llm_decision_log").fetchone()[0], 0)
        self.assertGreater(c.execute("SELECT COUNT(*) FROM objective_accessibility_daily").fetchone()[0], 0)
        c.close()

    def test_rq2_information_channels_can_be_ablated_independently(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / "ablation.db"
        config, graph, agents, interventions = build_demo_scenario(
            days=2, agent_count=3, seed=9, model_type="dynamic_cognitive",
        )
        config.behavior_driver = "deterministic"
        config.directional_vision_enabled = False
        config.direct_experience_enabled = False
        config.face_to_face_enabled = False
        SimulationEngine(config, graph, agents, interventions, path, DeterministicLLMDriver()).run()
        with sqlite3.connect(path) as c:
            observations = c.execute("SELECT COUNT(*) FROM observation_event_log").fetchone()[0]
            meetings = c.execute("SELECT COUNT(*) FROM copresence_event").fetchone()[0]
            process_rows = c.execute("SELECT COUNT(*) FROM information_process_daily").fetchone()[0]
        self.assertEqual(observations, 0)
        self.assertEqual(meetings, 0)
        self.assertEqual(process_rows, 2 * 3)

    def test_three_model_pairing_outputs_rq3_comparison(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        paths = {}
        for label, model in (("B0", "objective_accessibility"), ("B1", "omniscient"), ("M", "dynamic_cognitive")):
            path = Path(self.temp.name) / f"suite_{label}.db"
            config, graph, agents, interventions = build_demo_scenario(days=2, agent_count=3, seed=11, model_type=model)
            config.behavior_driver = "deterministic"
            SimulationEngine(config, graph, agents, interventions, path, DeterministicLLMDriver()).run()
            paths[label] = path
        output = compare_model_databases(paths, Path(self.temp.name) / "research")
        self.assertTrue(Path(output["comparison_csv"]).exists())
        c = sqlite3.connect(paths["M"])
        pairs = {row for row in c.execute("SELECT DISTINCT model_a, model_b FROM model_comparison")}
        self.assertIn(("B0", "B1"), pairs)
        self.assertIn(("B1", "M"), pairs)
        c.close()

    def test_omniscient_has_no_cognitive_update_log(self) -> None:
        path = self.run_small("omniscient")
        c = sqlite3.connect(path)
        self.assertEqual(c.execute("SELECT COUNT(*) FROM cognitive_update_log").fetchone()[0], 0)
        c.close()


if __name__ == "__main__":
    unittest.main()
