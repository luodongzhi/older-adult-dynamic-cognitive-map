from __future__ import annotations

import random
import tempfile
import unittest

from cogmap_sim.agent_repository import AgentRepository
from cogmap_sim.models import AgentProfile
from cogmap_sim.modules import M05DailyActivityGenerator
from cogmap_sim.scenario import build_demo_scenario


class AgentRepositoryTests(unittest.TestCase):
    def test_add_list_remove(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = AgentRepository(directory)
            profile = AgentProfile(
                agent_id="TEST_AGENT", group_id="test", home_aoi_id="HOME",
                routine_anchor_aoi_id="WORK", exploration_tendency=0.5,
                map_search_tendency=0.5, social_participation=0.5,
                safety_sensitivity=0.5, route_habit=0.5,
                visual_attention=0.5, social_contact_frequency=0.8,
            )
            repository.add(profile)
            self.assertEqual(repository.get("TEST_AGENT").social_contact_frequency, 0.8)
            self.assertEqual(len(repository.list()), 1)
            repository.remove("TEST_AGENT")
            self.assertEqual(repository.list(), [])

    def test_society_citysim_profile_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = AgentRepository(directory)
            profile = AgentProfile(
                agent_id="SCHEDULE_AGENT", group_id="schedule_test",
                home_aoi_id="HOME_W", routine_anchor_aoi_id="OFFICE_E",
                exploration_tendency=0.5, map_search_tendency=0.5,
                social_participation=0.5, safety_sensitivity=0.5,
                route_habit=0.5, visual_attention=0.5,
                age=68,
                chronotype="morning",
                work_days=[1, 3, 5],
                work_from_home_probability=0.25,
                schedule_flexibility=0.35,
                activity_frequency_weekly={"CAFE": 1.0, "PARK": 2.0, "SHOP": 1.5},
                initial_needs={"energy": 0.7, "social": 0.3, "leisure": 0.6, "errand": 0.2},
            )
            repository.add(profile)

            restored = repository.get("SCHEDULE_AGENT")
            self.assertEqual(restored.profile_version, "older-adult-information-arrival-v2")
            self.assertEqual(restored.age, 68)
            self.assertEqual(restored.work_days, [1, 3, 5])
            self.assertEqual(restored.activity_frequency_weekly["PARK"], 2.0)

    def test_schedule_respects_work_days_and_work_from_home(self) -> None:
        _, graph, _, _ = build_demo_scenario(agent_count=1)
        profile = AgentProfile(
            agent_id="SCHEDULE_AGENT", group_id="schedule_test",
            home_aoi_id="HOME_W", routine_anchor_aoi_id="OFFICE_E",
            exploration_tendency=0.5, map_search_tendency=0.5,
            social_participation=0.5, safety_sensitivity=0.5,
            route_habit=0.5, visual_attention=0.5,
            work_days=[1],
            occupation="part_time",
            employment_status="employed",
            chronotype="intermediate",
            work_start_mean=9.0,
            work_start_std=0.0,
            work_duration_mean=6.0,
            work_duration_std=0.0,
            work_from_home_probability=1.0,
            routine_anchor_frequency_weekly=0.0,
            activity_frequency_weekly={"CAFE": 0.0, "PARK": 0.0, "SHOP": 0.0},
            stay_home_preference=1.0,
        )
        generator = M05DailyActivityGenerator()

        monday = generator.generate(1, profile, graph, random.Random(7), {"flexible_activity_type": "NONE"})
        tuesday = generator.generate(2, profile, graph, random.Random(7), {"flexible_activity_type": "NONE"})

        monday_work = next(episode for episode in monday if episode.activity_type == "WORK")
        self.assertEqual(monday_work.fixed_destination_aoi, "HOME_W")
        self.assertAlmostEqual(monday_work.planned_start, 9.0)
        self.assertFalse(any(episode.activity_type == "WORK" for episode in tuesday))
        self.assertEqual([episode.activity_type for episode in tuesday], ["HOME"])

    def test_repository_rejects_under_age_profile_for_older_adult_experiment(self) -> None:
        _, graph, _, _ = build_demo_scenario(agent_count=1)
        with tempfile.TemporaryDirectory() as directory:
            repository = AgentRepository(directory)
            profile = AgentProfile(
                agent_id="YOUNG_AGENT", group_id="out_of_scope",
                home_aoi_id="HOME_W", routine_anchor_aoi_id="OFFICE_E",
                exploration_tendency=0.5, map_search_tendency=0.5,
                social_participation=0.5, safety_sensitivity=0.5,
                route_habit=0.5, visual_attention=0.5,
                age=59,
            )
            with self.assertRaisesRegex(ValueError, "age >= 60"):
                repository.validate_for_graph([profile], graph, minimum_age=60)

    def test_repository_rejects_invalid_schedule_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = AgentRepository(directory)
            profile = AgentProfile(
                agent_id="INVALID_AGENT", group_id="schedule_test",
                home_aoi_id="HOME", routine_anchor_aoi_id="WORK",
                exploration_tendency=0.5, map_search_tendency=0.5,
                social_participation=0.5, safety_sensitivity=0.5,
                route_habit=0.5, visual_attention=0.5,
                chronotype="night_owl",
            )
            with self.assertRaisesRegex(ValueError, "chronotype"):
                repository.add(profile)


if __name__ == "__main__":
    unittest.main()
