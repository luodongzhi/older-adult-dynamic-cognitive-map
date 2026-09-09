from __future__ import annotations

import json
import unittest
from pathlib import Path

from cogmap_sim.gis import load_graph_json
from cogmap_sim.models import EdgeStatus, InterventionEvent
from cogmap_sim.modules import M02InterventionManager


PROGRAM_ROOT = Path(__file__).resolve().parents[1]


class InterventionFileTests(unittest.TestCase):
    def test_generated_older_adult_bindings_match_current_map(self) -> None:
        graph = load_graph_json(PROGRAM_ROOT / "osmmap" / "processed" / "imported_map.json")
        bindings = json.loads(
            (PROGRAM_ROOT / "osmmap" / "processed" / "agent_bindings.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(bindings), 3)
        self.assertEqual(len({item["home_aoi_id"] for item in bindings}), 3)
        meeting_sets = {tuple(item["regular_meeting_aoi_ids"]) for item in bindings}
        self.assertEqual(len(meeting_sets), 1)
        for item in bindings:
            self.assertIn(item["home_aoi_id"], graph.aois)
            self.assertIn(item["routine_anchor_aoi_id"], graph.aois)
            self.assertTrue(all(value in graph.aois for value in item["regular_meeting_aoi_ids"]))
            multiplier = float(item["binding_radius_multiplier"])
            limit = float(item["activity_radius_m"]) * multiplier
            self.assertLessEqual(float(item["home_to_anchor_distance_m"]), limit)
            self.assertLessEqual(float(item["home_to_meeting_distance_m"]), limit)

    def test_two_stage_real_map_interventions(self) -> None:
        graph = load_graph_json(PROGRAM_ROOT / "osmmap" / "processed" / "imported_map.json")
        data = json.loads(
            (PROGRAM_ROOT / "experiments" / "current" / "interventions.json").read_text(encoding="utf-8")
        )
        events = [InterventionEvent(**item) for item in data]
        self.assertEqual([event.effective_day for event in events], [3, 6])
        self.assertEqual(len({event.intervention_id for event in events}), 2)
        aoi_event = next(event for event in events if event.target_type == "AOI")
        road_event = next(event for event in events if event.target_type == "ROAD_EDGE")
        self.assertIn(aoi_event.target_id, graph.aois)
        self.assertIn(road_event.target_id, graph.edges)
        self.assertEqual(graph.aois[aoi_event.target_id].function, aoi_event.before_value)
        self.assertEqual(str(graph.edges[road_event.target_id].status), road_event.before_value)
        source_function = graph.aois[aoi_event.target_id].function
        source_road_status = graph.edges[road_event.target_id].status

        experiment_graph = graph.clone()
        manager = M02InterventionManager()
        manager.apply(experiment_graph, events, 3)
        self.assertEqual(experiment_graph.aois[aoi_event.target_id].function, "PARK")
        self.assertEqual(experiment_graph.edges[road_event.target_id].status, EdgeStatus.OPEN)
        manager.apply(experiment_graph, events, 6)
        self.assertEqual(experiment_graph.edges[road_event.target_id].status, EdgeStatus.CLOSED)

        self.assertEqual(graph.aois[aoi_event.target_id].function, source_function)
        self.assertEqual(graph.edges[road_event.target_id].status, source_road_status)


if __name__ == "__main__":
    unittest.main()
