from __future__ import annotations

import unittest

from shapely.geometry import Point

from cogmap_sim.information import InformationLedger
from cogmap_sim.graph import UrbanGraph
from cogmap_sim.models import (
    AOI,
    AgentProfile,
    Node,
    ObservationEvent,
    Occluder,
    SourceChannel,
)
from cogmap_sim.spatial import DirectionalVisionEngine, TrajectorySample


class FixedRandom:
    def __init__(self, value: float):
        self.value = value

    def random(self) -> float:
        return self.value


def profile() -> AgentProfile:
    return AgentProfile(
        agent_id="A", group_id="G", home_aoi_id="HOME",
        routine_anchor_aoi_id="WORK", exploration_tendency=0.5,
        map_search_tendency=0.5, social_participation=0.5,
        safety_sensitivity=0.5, route_habit=0.5, visual_attention=0.8,
        social_contact_frequency=0.5, visual_search_radius=20.0,
        visual_detection_skill=1.0, visual_field_of_view_deg=90.0,
    )


def graph(occluded: bool = False) -> UrbanGraph:
    nodes = {"N": Node("N", 0.0, 0.0)}
    aois = {
        "HOME": AOI("HOME", "Home", "N", "HOME", 0.5),
        "WORK": AOI("WORK", "Work", "N", "WORK", 0.5),
    }
    occluders = {
        "B": Occluder("B", [(4, -2), (6, -2), (6, 2), (4, 2), (4, -2)])
    } if occluded else {}
    return UrbanGraph(nodes, {}, aois, {"distance_scale": 1.0}, occluders=occluders)


class DirectionalVisionTests(unittest.TestCase):
    def test_front_target_is_visible_but_target_behind_is_not(self) -> None:
        engine = DirectionalVisionEngine()
        sample = [TrajectorySample(8.0, 0.0, 0.0, 0.0, "E", 0.5)]
        visible = engine.evaluate(sample, Point(10, 0), graph(), profile(), 1.0, FixedRandom(0.0))
        behind = engine.evaluate(sample, Point(-10, 0), graph(), profile(), 1.0, FixedRandom(0.0))
        self.assertTrue(visible.detected)
        self.assertTrue(visible.line_of_sight)
        self.assertFalse(behind.detected)
        self.assertEqual(behind.probability, 0.0)

    def test_building_blocks_line_of_sight(self) -> None:
        engine = DirectionalVisionEngine()
        sample = [TrajectorySample(8.0, 0.0, 0.0, 0.0, "E", 0.5)]
        result = engine.evaluate(sample, Point(10, 0), graph(True), profile(), 1.0, FixedRandom(0.0))
        self.assertFalse(result.detected)
        self.assertFalse(result.line_of_sight)


class InformationLedgerTests(unittest.TestCase):
    def test_information_is_deduplicated_and_available_on_arrival_day(self) -> None:
        ledger = InformationLedger()
        event = ObservationEvent(
            "OBS1", 2, 9.0, "A", "E1", "PARK", "AOI",
            SourceChannel.VISUAL_SEARCH, "function", "PARK", 0.9, 0.8,
        )
        created = ledger.ingest(2, "A", [event, event])
        self.assertEqual(len(created), 1)
        self.assertEqual(ledger.items_for("A"), created)
        self.assertTrue(created[0].first_hand_source)


if __name__ == "__main__":
    unittest.main()
