from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ExperimentInputError(ValueError):
    """Raised when an experiment bundle is incomplete or internally inconsistent."""


def _read_json(path: Path, expected: type) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"实验输入文件不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentInputError(
            f"实验输入不是合法 JSON：{path}（第 {exc.lineno} 行，第 {exc.colno} 列）"
        ) from exc
    if not isinstance(value, expected):
        expected_name = "JSON 对象" if expected is dict else "JSON 数组"
        raise ExperimentInputError(f"{path.name} 的最外层必须是{expected_name}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _merge_unique(target: dict[str, Any], source: dict[str, Any], source_name: str) -> None:
    duplicates = sorted(set(target).intersection(source))
    if duplicates:
        raise ExperimentInputError(
            f"{source_name} 与其他输入文件重复定义字段：{duplicates}。"
            "每个参数只能在一个文件中定义。"
        )
    target.update(source)


def _normalise_agents(document: dict[str, Any]) -> dict[str, Any]:
    allowed = {"agent_repository", "location_binding", "selected_agents"}
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ExperimentInputError(f"agents.json 包含未知字段：{unknown}")

    selected = document.get("selected_agents")
    if not isinstance(selected, list) or not selected:
        raise ExperimentInputError("agents.json 的 selected_agents 必须是非空数组")
    agents: list[dict[str, Any]] = []
    for index, item in enumerate(selected, start=1):
        if isinstance(item, str):
            agents.append({"agent_id": item})
        elif isinstance(item, dict) and isinstance(item.get("agent_id"), str):
            agents.append(dict(item))
        else:
            raise ExperimentInputError(
                f"agents.json 的 selected_agents 第 {index} 项必须是 Agent ID 字符串或对象"
            )
    identifiers = [item["agent_id"] for item in agents]
    if len(identifiers) != len(set(identifiers)):
        raise ExperimentInputError("agents.json 的 Agent ID 不能重复")

    binding = document.get("location_binding", {})
    if not isinstance(binding, dict):
        raise ExperimentInputError("agents.json 的 location_binding 必须是对象")
    mode = str(binding.get("mode", "strict")).lower()
    if mode not in {"strict", "generated"}:
        raise ExperimentInputError("location_binding.mode 只支持 strict 或 generated")
    if mode == "generated" and not binding.get("file"):
        raise ExperimentInputError("generated 模式必须设置 location_binding.file")

    return {
        "agent_repository": str(document.get("agent_repository", "agents")),
        "agent_binding_mode": mode,
        "agent_bindings_file": binding.get("file"),
        "agents": agents,
    }


def _validate_run_settings(settings: dict[str, Any]) -> None:
    days = settings.get("days")
    if not isinstance(days, int) or isinstance(days, bool) or days < 1:
        raise ExperimentInputError("run.json 的 days 必须是大于等于 1 的整数")
    seed = settings.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ExperimentInputError("run.json 的 seed 必须是整数")
    experiment_id = settings.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ExperimentInputError("run.json 必须设置非空 experiment_id")
    output = settings.get("output_database")
    if not isinstance(output, str) or not output.lower().endswith(".db"):
        raise ExperimentInputError("run.json 的 output_database 必须指向 .db 文件")
    models = settings.get("research_models", ["B0", "B1", "M"])
    if not isinstance(models, list) or not models:
        raise ExperimentInputError("run.json 的 research_models 必须是非空数组")
    invalid = [str(value) for value in models if str(value).upper() not in {"B0", "B1", "M"}]
    if invalid:
        raise ExperimentInputError(f"research_models 只支持 B0、B1、M：{invalid}")


def _validate_intervention_document(items: list[Any]) -> None:
    required = {
        "intervention_id", "effective_day", "operation_type", "target_type",
        "target_id", "before_value", "after_value",
    }
    identifiers: list[str] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ExperimentInputError(f"interventions.json 第 {index} 项必须是对象")
        missing = sorted(required - set(item))
        if missing:
            raise ExperimentInputError(f"interventions.json 第 {index} 项缺少字段：{missing}")
        if not isinstance(item["intervention_id"], str) or not item["intervention_id"].strip():
            raise ExperimentInputError(f"interventions.json 第 {index} 项的 intervention_id 必须是非空字符串")
        day = item["effective_day"]
        if not isinstance(day, int) or isinstance(day, bool) or day < 0:
            raise ExperimentInputError(f"interventions.json 第 {index} 项的 effective_day 必须是非负整数")
        for field in ("operation_type", "target_type", "target_id", "before_value", "after_value"):
            if not isinstance(item[field], str) or not item[field].strip():
                raise ExperimentInputError(f"interventions.json 第 {index} 项的 {field} 必须是非空字符串")
        for field in (
            "visibility_level", "announcement_level", "visual_salience",
            "official_announcement", "signage_support",
        ):
            if field not in item or item[field] is None:
                continue
            value = item[field]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
                raise ExperimentInputError(
                    f"interventions.json 第 {index} 项的 {field} 必须位于 0..1"
                )
        identifiers.append(str(item["intervention_id"]))
    if len(identifiers) != len(set(identifiers)):
        raise ExperimentInputError("interventions.json 的 intervention_id 不能重复")


def load_experiment_settings(selector_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    """Load either the clean experiment bundle or a legacy flat settings file.

    New format: ``pycharm_run.json`` contains only ``active_experiment`` and the
    selected directory contains map.json, agents.json, run.json and
    interventions.json.  Legacy flat JSON remains readable for old CLI projects.
    """
    selector_path = Path(selector_path).resolve()
    project_root = Path(project_root).resolve()
    selector = _read_json(selector_path, dict)
    active = selector.get("active_experiment")
    if active is None:
        settings = dict(selector)
        settings["_experiment_format"] = "legacy_flat"
        settings["_experiment_directory"] = str(selector_path.parent)
        settings["_input_hashes"] = {str(selector_path): file_sha256(selector_path)}
        return settings
    if set(selector) != {"active_experiment"}:
        raise ExperimentInputError(
            "新的 pycharm_run.json 只能包含 active_experiment；具体参数请放入实验目录"
        )

    experiment_dir = Path(str(active))
    if not experiment_dir.is_absolute():
        experiment_dir = project_root / experiment_dir
    experiment_dir = experiment_dir.resolve()
    try:
        experiment_dir.relative_to(project_root)
    except ValueError as exc:
        raise ExperimentInputError("active_experiment 必须位于项目目录内") from exc
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"当前实验目录不存在：{experiment_dir}")

    paths = {
        "selector": selector_path,
        "map": experiment_dir / "map.json",
        "agents": experiment_dir / "agents.json",
        "run": experiment_dir / "run.json",
        "interventions": experiment_dir / "interventions.json",
    }
    map_settings = _read_json(paths["map"], dict)
    agent_document = _read_json(paths["agents"], dict)
    run_settings = _read_json(paths["run"], dict)
    interventions = _read_json(paths["interventions"], list)
    agent_settings = _normalise_agents(agent_document)
    _validate_run_settings(run_settings)
    _validate_intervention_document(interventions)

    settings: dict[str, Any] = {}
    _merge_unique(settings, map_settings, "map.json")
    _merge_unique(settings, agent_settings, "agents.json")
    _merge_unique(settings, run_settings, "run.json")
    settings["interventions"] = str(paths["interventions"])
    settings["_experiment_format"] = "bundle_v1"
    settings["_experiment_directory"] = str(experiment_dir)
    settings["_input_files"] = {name: str(path) for name, path in paths.items()}
    settings["_input_hashes"] = {
        str(path): file_sha256(path) for path in paths.values()
    }
    return settings
