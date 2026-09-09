"""PyCharm 一键运行入口：老年人动态认知地图 B0-B1-M 实验。

在 PyCharm 中打开本文件，点击绿色 Run 三角即可。所有相对路径均以
本文件所在目录为基准，不依赖 PyCharm 的 Working directory。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import webbrowser
from dataclasses import asdict, replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cogmap_sim.agent_repository import AgentRepository
from cogmap_sim.analysis import compare_model_databases
from cogmap_sim.engine import SimulationEngine
from cogmap_sim.experiment_inputs import file_sha256, load_experiment_settings
from cogmap_sim.gis import import_shapefiles, load_graph_json, save_graph_json, shapefile_bundle_hash
from cogmap_sim.llm import create_llm_driver
from cogmap_sim.models import InterventionEvent, ScenarioConfig
from cogmap_sim.web import export_dashboard_gallery, serve_dashboard


MODEL_ALIASES = {
    "B0": "objective_accessibility",
    "B1": "omniscient",
    "M": "dynamic_cognitive",
}


def project_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings() -> dict:
    path = PROJECT_ROOT / "pycharm_run.json"
    if not path.exists():
        raise FileNotFoundError(f"PyCharm 配置文件不存在：{path}")
    settings = load_experiment_settings(path, PROJECT_ROOT)
    if settings.get("_experiment_format") == "bundle_v1":
        print(f"[实验] 当前实验目录：{settings['_experiment_directory']}")
    else:
        print("[实验] 正在读取旧版单文件配置；建议迁移到 experiments/ 实验包")
    return settings


def load_api_key(api_file: Path) -> None:
    if not api_file.exists():
        raise FileNotFoundError(f"豆包 API 文件不存在：{api_file}")
    raw = api_file.read_text(encoding="utf-8-sig").strip()
    if not raw:
        raise ValueError("api.txt 是空文件")
    if "=" in raw:
        name, raw = raw.split("=", 1)
        if name.strip() not in {"ARK_API_KEY", "API_KEY"}:
            raise ValueError("api.txt 使用 name=value 格式时，name 必须是 ARK_API_KEY")
    key = raw.strip().strip('"').strip("'")
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    if len(key) < 10 or any(character.isspace() for character in key):
        raise ValueError("api.txt 中的 API Key 格式不正确")
    os.environ["ARK_API_KEY"] = key
    print(f"[配置] 已从 {api_file.name} 安全读取豆包 API Key（不会打印或写入数据库）")


def recorded_source_hash(source_hashes: dict[str, str], source_path: Path) -> str | None:
    """Find a source hash even when a repository was cloned to a new directory."""
    exact = source_hashes.get(str(source_path.resolve()))
    if exact is not None:
        return exact
    same_name = {
        value
        for stored_path, value in source_hashes.items()
        if Path(stored_path).name.casefold() == source_path.name.casefold()
    }
    return next(iter(same_name)) if len(same_name) == 1 else None


def map_cache_is_stale(cache: Path, source_paths: list[Path]) -> bool:
    if not cache.exists():
        return True
    cache_time = cache.stat().st_mtime
    if not any(path.stat().st_mtime > cache_time for path in source_paths):
        return False
    try:
        source_hashes = load_graph_json(cache).metadata.get("source_hashes", {})
    except (OSError, ValueError, TypeError, KeyError):
        return True
    return any(
        recorded_source_hash(source_hashes, path) != shapefile_bundle_hash(path)
        for path in source_paths
    )


def load_or_import_map(settings: dict):
    roads = project_path(settings["roads_shp"])
    aois = project_path(settings["aois_shp"])
    nodes = project_path(settings.get("nodes_shp"))
    entrances = project_path(settings.get("entrances_shp"))
    occluders = project_path(settings.get("occluders_shp"))
    signage = project_path(settings.get("signage_shp"))
    cache = project_path(settings["map_cache"])
    assert roads is not None and aois is not None and cache is not None
    for label, path in (("道路 SHP", roads), ("AOI SHP", aois)):
        if not path.exists():
            raise FileNotFoundError(f"{label} 不存在：{path}")
    optional_layers = [item for item in (nodes, entrances, occluders, signage) if item is not None]
    for path in optional_layers:
        if not path.exists():
            raise FileNotFoundError(f"配置中的可选 SHP 不存在：{path}")
    sources = [roads, aois, *optional_layers]
    graph = None
    rebuild_cache = map_cache_is_stale(cache, sources)
    if not rebuild_cache:
        graph = load_graph_json(cache)
        rebuild_cache = graph.metadata.get("schema_version") != "1.0.1"
    if rebuild_cache:
        print("[地图] 首次运行或 SHP 已修改，正在构建道路拓扑……")
        graph = import_shapefiles(
            roads, aois, nodes,
            float(settings.get("snap_tolerance", 0.5)),
            settings.get("field_mapping"),
            entrances_path=entrances,
            occluders_path=occluders,
            signage_path=signage,
        )
        save_graph_json(graph, cache)
        print(f"[地图] 缓存已生成：{cache}")
    else:
        print(f"[地图] 使用现有缓存：{cache}")
    assert graph is not None
    osm_source = project_path(settings.get("osm_source"))
    if osm_source is not None:
        if not osm_source.exists():
            raise FileNotFoundError(f"配置的原始 OSM 文件不存在：{osm_source}")
        source_hashes = graph.metadata.get("source_hashes", {})
        expected_hash = recorded_source_hash(source_hashes, osm_source)
        actual_hash = file_sha256(osm_source)
        if expected_hash is None or expected_hash != actual_hash:
            raise RuntimeError(
                "原始 map.osm 与 processed/imported_map.json 不属于同一地图。"
                "请先运行 tools/process_osm_map.py，再开始实验。"
            )
    print(
        f"[地图] 节点={len(graph.nodes)}, 道路={len(graph.edges)}, AOI={len(graph.aois)}, "
        f"入口={len(graph.entrances)}, 遮挡物={len(graph.occluders)}, 标识={len(graph.signage)}"
    )
    return graph


def load_agents(settings: dict, graph):
    repository_path = project_path(settings.get("agent_repository", "agents"))
    assert repository_path is not None
    repository = AgentRepository(repository_path)
    configured_agents = list(settings["agents"])
    binding_mode = str(settings.get("agent_binding_mode", "strict")).lower()
    if binding_mode == "generated":
        bindings_path = project_path(
            settings.get("agent_bindings_file", "osmmap/processed/agent_bindings.json")
        )
        assert bindings_path is not None
        if not bindings_path.exists():
            raise FileNotFoundError(
                f"地图专属 Agent 绑定文件不存在：{bindings_path}；请先运行 tools/process_osm_map.py"
            )
        generated = json.loads(bindings_path.read_text(encoding="utf-8"))
        by_agent = {str(item["agent_id"]): item for item in generated}
        rebound = []
        for item in configured_agents:
            agent_id = str(item["agent_id"])
            if agent_id not in by_agent:
                raise ValueError(f"地图绑定文件中没有 Agent：{agent_id}")
            binding = by_agent[agent_id]
            rebound.append({
                **item,
                "home_aoi_id": binding["home_aoi_id"],
                "routine_anchor_aoi_id": binding["routine_anchor_aoi_id"],
                "regular_meeting_aoi_ids": binding["regular_meeting_aoi_ids"],
            })
        configured_agents = rebound
        print(f"[Agent绑定] 已加载当前地图专属配置：{bindings_path}")
    elif binding_mode != "strict":
        raise ValueError("agent_binding_mode 只支持 strict 或 generated")

    profiles = []
    for item in configured_agents:
        profile = repository.get(item["agent_id"])
        anchor = str(item.get("routine_anchor_aoi_id", item.get("work_aoi_id", profile.routine_anchor_aoi_id)))
        meetings = [str(value) for value in item.get("regular_meeting_aoi_ids", profile.regular_meeting_aoi_ids)]
        profiles.append(replace(
            profile,
            home_aoi_id=str(item.get("home_aoi_id", profile.home_aoi_id)),
            routine_anchor_aoi_id=anchor,
            regular_meeting_aoi_ids=meetings,
            llm_model=settings["llm_model"],
        ))
    repository.validate_for_graph(profiles, graph, int(settings.get("minimum_agent_age", 60)))
    for profile in profiles:
        print(
            f"[老年 Agent] {profile.agent_id}: 年龄={profile.age}, HOME={profile.home_aoi_id}, "
            f"日常锚点={profile.routine_anchor_aoi_id}, 社交参与={profile.social_participation:.2f}, "
            f"视觉注意={profile.visual_attention:.2f}, 路线习惯={profile.route_habit:.2f}"
        )
    return profiles


def load_interventions(settings: dict, graph) -> list[InterventionEvent]:
    path = project_path(settings.get("interventions"))
    if path is None or not path.exists():
        return []
    events = [InterventionEvent(**item) for item in json.loads(path.read_text(encoding="utf-8"))]
    operation_targets = {
        "ROAD_OPEN": "ROAD_EDGE", "ROAD_CLOSE": "ROAD_EDGE",
        "AOI_FUNCTION_CHANGE": "AOI",
        "ENTRANCE_OPEN": "ENTRANCE", "ENTRANCE_CLOSE": "ENTRANCE",
        "SIGNAGE_ACTIVATE": "SIGNAGE", "SIGNAGE_DEACTIVATE": "SIGNAGE",
    }
    object_ids = {
        "ROAD_EDGE": set(graph.edges), "AOI": set(graph.aois),
        "ENTRANCE": set(graph.entrances), "SIGNAGE": set(graph.signage),
    }
    seen_ids: set[str] = set()
    days = int(settings.get("days", 1))
    for event in events:
        if event.intervention_id in seen_ids:
            raise ValueError(f"干预 ID 重复：{event.intervention_id}")
        seen_ids.add(event.intervention_id)
        expected_target = operation_targets.get(event.operation_type)
        if expected_target is None or event.target_type != expected_target:
            raise ValueError(f"干预 {event.intervention_id} 的操作和 target_type 不匹配")
        if event.target_id not in object_ids[event.target_type]:
            raise ValueError(f"干预对象不存在：{event.target_type}/{event.target_id}")
        if not 0 <= event.effective_day <= days:
            raise ValueError(f"干预 {event.intervention_id} 超出模拟天数 0..{days}")
    events.sort(key=lambda event: (event.effective_day, event.intervention_id))

    expected_after = {
        "ROAD_OPEN": "OPEN", "ROAD_CLOSE": "CLOSED",
        "ENTRANCE_OPEN": "OPEN", "ENTRANCE_CLOSE": "CLOSED",
        "SIGNAGE_ACTIVATE": "TRUE", "SIGNAGE_DEACTIVATE": "FALSE",
    }
    planned_states: dict[tuple[str, str], str] = {}
    for event in events:
        key = (event.target_type, event.target_id)
        if key not in planned_states:
            if event.target_type == "ROAD_EDGE":
                planned_states[key] = str(graph.edges[event.target_id].status)
            elif event.target_type == "AOI":
                planned_states[key] = str(graph.aois[event.target_id].function)
            elif event.target_type == "ENTRANCE":
                planned_states[key] = str(graph.entrances[event.target_id].status)
            else:
                planned_states[key] = str(graph.signage[event.target_id].active).upper()
        current_value = planned_states[key]
        if event.before_value != current_value:
            raise ValueError(
                f"干预 {event.intervention_id} 的 before_value={event.before_value}，"
                f"但该对象在 Day {event.effective_day} 前应为 {current_value}"
            )
        required_after = expected_after.get(event.operation_type)
        if required_after is not None and event.after_value != required_after:
            raise ValueError(
                f"干预 {event.intervention_id} 的 {event.operation_type} 要求 after_value={required_after}"
            )
        planned_states[key] = event.after_value

    print(f"[干预] 已加载 {len(events)} 个空间干预：{path}")
    for event in events:
        print(
            f"[干预] Day {event.effective_day}: {event.operation_type} "
            f"{event.target_id} ({event.before_value} -> {event.after_value})"
        )
    return events


def attach_input_provenance(settings: dict, graph, agents) -> None:
    """Attach all non-secret experiment inputs to every database manifest."""
    hashes = dict(settings.get("_input_hashes", {}))
    repository = project_path(settings.get("agent_repository", "agents"))
    if repository is not None:
        for agent in agents:
            profile_path = repository / f"{agent.agent_id}.json"
            if profile_path.exists():
                hashes[str(profile_path.resolve())] = file_sha256(profile_path)
    bindings = project_path(settings.get("agent_bindings_file"))
    if bindings is not None and bindings.exists():
        hashes[str(bindings.resolve())] = file_sha256(bindings)

    graph.metadata.setdefault("source_hashes", {}).update(hashes)
    graph.metadata["experiment_inputs"] = {
        "format": settings.get("_experiment_format", "legacy_flat"),
        "directory": settings.get("_experiment_directory"),
        "selected_agent_ids": [agent.agent_id for agent in agents],
        "input_files": settings.get("_input_files", {}),
    }


def export_resolved_input_snapshot(
    settings: dict,
    agents,
    interventions: list[InterventionEvent],
    base_output: Path,
) -> Path:
    """Write the exact, resolved input of this run without exposing the API key."""
    snapshot_path = base_output.with_name(f"{base_output.stem}_inputs.resolved.json")
    public_settings = {
        key: value for key, value in settings.items()
        if not key.startswith("_")
    }
    payload = {
        "schema_version": "1.0",
        "experiment_format": settings.get("_experiment_format", "legacy_flat"),
        "experiment_directory": settings.get("_experiment_directory"),
        "input_hashes": settings.get("_input_hashes", {}),
        "resolved_settings": public_settings,
        "resolved_agents": [asdict(agent) for agent in agents],
        "resolved_interventions": [asdict(event) for event in interventions],
    }
    snapshot_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return snapshot_path


def build_config(settings: dict, model_type: str, experiment_id: str, agent_count: int) -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id=settings.get("scenario_id", "osm_older_adult_neighbourhood"),
        experiment_id=experiment_id,
        end_day=int(settings.get("days", 60)),
        random_seed=int(settings.get("seed", 42)),
        model_type=model_type,
        agent_count=agent_count,
        behavior_driver=settings.get("llm_provider", "doubao"),
        llm_model=settings["llm_model"],
        paired_seed=int(settings.get("paired_seed", settings.get("seed", 42))),
        directional_vision_enabled=bool(settings.get("directional_vision_enabled", True)),
        direct_experience_enabled=bool(settings.get("direct_experience_enabled", True)),
        face_to_face_enabled=bool(settings.get("face_to_face_enabled", True)),
        copresence_min_overlap_minutes=float(settings.get("copresence_min_overlap_minutes", 8.0)),
        max_reports_per_conversation=int(settings.get("max_reports_per_conversation", 2)),
        minimum_agent_age=int(settings.get("minimum_agent_age", 60)),
        export_research_csv=bool(settings.get("export_research_csv", True)),
    )


def model_output(base_output: Path, model_label: str, suite_enabled: bool) -> Path:
    if not suite_enabled:
        return base_output
    return base_output.with_name(f"{base_output.stem}_{model_label}{base_output.suffix}")


def run() -> None:
    settings = load_settings()
    suite_enabled = bool(settings.get("run_research_suite", True))
    labels = [str(value).upper() for value in settings.get("research_models", ["B0", "B1", "M"])] if suite_enabled else [str(settings.get("model_label", "M")).upper()]
    invalid = [label for label in labels if label not in MODEL_ALIASES]
    if invalid:
        raise ValueError(f"research_models 只支持 B0、B1、M；当前无效值：{invalid}")
    graph = load_or_import_map(settings)
    agents = load_agents(settings, graph)
    interventions = load_interventions(settings, graph)
    attach_input_provenance(settings, graph, agents)
    base_output = project_path(settings["output_database"])
    assert base_output is not None
    base_output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = export_resolved_input_snapshot(settings, agents, interventions, base_output)
    print(
        f"[预检] 通过：天数={settings['days']}, Agent={len(agents)}, "
        f"干预={len(interventions)}, 模型={','.join(labels)}"
    )
    print(f"[实验输入] 已保存本次解析后的完整快照：{snapshot}")

    provider = os.environ.get("COGMAP_LLM_PROVIDER", settings.get("llm_provider", "doubao"))
    if provider == "doubao" and any(label != "B0" for label in labels):
        api_file = project_path(settings.get("api_file", "api.txt"))
        assert api_file is not None
        load_api_key(api_file)

    outputs: dict[str, Path] = {}
    summaries: dict[str, dict] = {}
    for label in labels:
        model_type = MODEL_ALIASES[label]
        output = model_output(base_output, label, suite_enabled)
        experiment_id = f"{settings.get('experiment_id', base_output.stem)}_{label}"
        config = build_config(settings, model_type, experiment_id, len(agents))
        driver_provider = "rule_based" if label == "B0" else provider
        driver = create_llm_driver(driver_provider, settings["llm_model"])
        print(
            f"[模拟:{label}] 开始运行：天数={config.end_day}, 老年 Agent={len(agents)}, "
            f"模型={model_type}, 行为驱动={driver_provider}"
        )
        summary = SimulationEngine(
            config, graph.clone(), agents, interventions, output, driver,
        ).run(progress=lambda day, total, label=label: print(f"[模拟:{label}] 完成第 {day}/{total} 天"))
        outputs[label] = output
        summaries[label] = summary
        print(f"[模拟:{label}] 数据库：{output}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    suite_summary = None
    if {"B0", "B1", "M"}.issubset(outputs):
        comparison_dir = base_output.parent / "research_outputs" / f"{base_output.stem}_suite"
        suite_summary = compare_model_databases(outputs, comparison_dir)
        print(f"[研究输出] RQ3 配对比较：{suite_summary['comparison_csv']}")

    selected_output = outputs.get("M") or outputs.get("B1") or outputs[labels[0]]
    selector, reports = export_dashboard_gallery(selected_output.parent, selected_output)
    print(f"[网页文件] 数据库选择入口：{selector}")
    print(f"[网页文件] 已生成 {len(reports)} 个独立实验报告")

    dashboard_override = os.environ.get("COGMAP_START_DASHBOARD")
    start_dashboard = settings.get("start_dashboard", True) if dashboard_override is None else dashboard_override.lower() not in {"0", "false", "no"}
    if start_dashboard:
        host = settings.get("dashboard_host", "127.0.0.1")
        port = int(settings.get("dashboard_port", 8765))
        url = f"http://{host}:{port}/"
        if settings.get("open_browser", True):
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        print(f"[网页] 正在启动数据库选择页 {url}；在 PyCharm 中点击红色停止按钮即可关闭")
        serve_dashboard(selected_output, host, port)


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        print("\n[运行失败]", error)
        print(
            "请检查 experiments/current 中的 agents.json、run.json、"
            "interventions.json、map.json，以及 osmmap、api.txt 和 PyCharm Python 解释器。"
        )
        raise
