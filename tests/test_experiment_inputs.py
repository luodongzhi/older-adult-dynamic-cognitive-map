from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROGRAM_ROOT = Path(__file__).resolve().parents[1]
SRC = PROGRAM_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cogmap_sim.experiment_inputs import ExperimentInputError, load_experiment_settings


class ExperimentInputTests(unittest.TestCase):
    def test_current_bundle_resolves_independent_inputs(self) -> None:
        settings = load_experiment_settings(PROGRAM_ROOT / "pycharm_run.json", PROGRAM_ROOT)
        self.assertEqual(settings["_experiment_format"], "bundle_v1")
        self.assertEqual(settings["days"], 10)
        self.assertEqual(
            [item["agent_id"] for item in settings["agents"]],
            ["A001_active_senior", "A002_routine_senior", "A003_limited_senior"],
        )
        self.assertEqual(settings["agent_binding_mode"], "generated")
        self.assertEqual(
            Path(settings["interventions"]),
            PROGRAM_ROOT / "experiments" / "current" / "interventions.json",
        )
        self.assertEqual(len(settings["_input_hashes"]), 5)

    def test_invalid_days_are_rejected_before_map_or_llm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            experiment.mkdir()
            (root / "pycharm_run.json").write_text(
                json.dumps({"active_experiment": "experiment"}), encoding="utf-8"
            )
            (experiment / "map.json").write_text("{}", encoding="utf-8")
            (experiment / "agents.json").write_text(
                json.dumps({
                    "agent_repository": "agents",
                    "location_binding": {"mode": "strict"},
                    "selected_agents": ["A001"],
                }),
                encoding="utf-8",
            )
            (experiment / "run.json").write_text(
                json.dumps({
                    "days": 0,
                    "seed": 42,
                    "experiment_id": "bad",
                    "output_database": "output/bad.db",
                }),
                encoding="utf-8",
            )
            (experiment / "interventions.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ExperimentInputError, "days"):
                load_experiment_settings(root / "pycharm_run.json", root)

    def test_duplicate_agent_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            experiment.mkdir()
            (root / "pycharm_run.json").write_text(
                json.dumps({"active_experiment": "experiment"}), encoding="utf-8"
            )
            (experiment / "map.json").write_text("{}", encoding="utf-8")
            (experiment / "agents.json").write_text(
                json.dumps({"selected_agents": ["A001", "A001"]}), encoding="utf-8"
            )
            (experiment / "run.json").write_text(
                json.dumps({
                    "days": 1,
                    "seed": 42,
                    "experiment_id": "bad",
                    "output_database": "output/bad.db",
                }),
                encoding="utf-8",
            )
            (experiment / "interventions.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ExperimentInputError, "Agent ID"):
                load_experiment_settings(root / "pycharm_run.json", root)


if __name__ == "__main__":
    unittest.main()
