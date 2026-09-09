from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

from .agent_repository import AgentRepository
from .analysis import compare_model_databases
from .engine import SimulationEngine
from .gis import import_shapefiles, load_graph_json, save_graph_json
from .llm import create_llm_driver
from .models import InterventionEvent, ScenarioConfig
from .scenario import build_demo_scenario
from .web import serve_dashboard


def _run(args: argparse.Namespace) -> None:
    repository = AgentRepository(args.agent_repository)
    available = repository.list()
    if args.agent_ids:
        agents = [repository.get(agent_id.strip()) for agent_id in args.agent_ids.split(",")]
    else:
        agents = available[:args.agents]
    if not agents:
        raise SystemExit(f"No agents found in {args.agent_repository}. Add one with: cogmap-sim agent-add --file profile.json")
    if args.map_json:
        graph = load_graph_json(args.map_json)
        repository.validate_for_graph(agents, graph)
        interventions = _load_interventions(args.interventions)
        config = ScenarioConfig(end_day=args.days, random_seed=args.seed, model_type=args.model, agent_count=len(agents))
    else:
        config, graph, agents, interventions = build_demo_scenario(args.days, len(agents), args.seed, args.model, agents)
    config.behavior_driver = args.llm_provider
    config.llm_model = args.llm_model
    config.directional_vision_enabled = not args.no_vision
    config.direct_experience_enabled = not args.no_direct_experience
    config.face_to_face_enabled = not args.no_face_to_face
    config.experiment_id = args.experiment_id or f"demo_{args.model}"
    driver = create_llm_driver(args.llm_provider, args.llm_model)
    summary = SimulationEngine(config, graph, agents, interventions, args.output, driver).run(
        progress=(lambda day, total: print(f"\rSimulating day {day}/{total}", end="", flush=True)) if not args.quiet else None
    )
    if not args.quiet:
        print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _compare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    repository = AgentRepository(args.agent_repository)
    profiles = repository.list()[:args.agents]
    databases = {}
    for label, model in (("B0", "objective_accessibility"), ("B1", "omniscient"), ("M", "dynamic_cognitive")):
        output = output_dir / f"{label}_{model}.db"
        if args.map_json:
            graph = load_graph_json(args.map_json)
            agents = profiles
            repository.validate_for_graph(agents, graph)
            interventions = _load_interventions(args.interventions)
            config = ScenarioConfig(
                end_day=args.days, random_seed=args.seed, model_type=model,
                agent_count=len(agents), paired_seed=args.seed,
            )
        else:
            config, graph, agents, interventions = build_demo_scenario(args.days, len(profiles), args.seed, model, profiles)
            config.paired_seed = args.seed
        config.behavior_driver = args.llm_provider
        config.llm_model = args.llm_model
        config.experiment_id = f"comparison_{model}"
        driver_provider = "rule_based" if label == "B0" else args.llm_provider
        summaries.append(SimulationEngine(config, graph, agents, interventions, output, create_llm_driver(driver_provider, args.llm_model)).run())
        summaries[-1]["database"] = str(output)
        databases[label] = output
        print(f"Completed {label}/{model}: {output}")
    comparison = compare_model_databases(databases, output_dir / "research_outputs")
    (output_dir / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


def _import_shp(args: argparse.Namespace) -> None:
    field_mapping = json.loads(Path(args.field_map).read_text(encoding="utf-8")) if args.field_map else None
    graph = import_shapefiles(
        args.roads, args.aois, args.nodes, args.snap_tolerance, field_mapping,
        entrances_path=args.entrances, occluders_path=args.occluders,
        signage_path=args.signage,
    )
    save_graph_json(graph, args.output)
    print(json.dumps({"output": args.output, "nodes": len(graph.nodes), "edges": len(graph.edges), "aois": len(graph.aois), "crs_available": bool(graph.metadata.get("crs_wkt"))}, ensure_ascii=False, indent=2))


def _agent_list(args: argparse.Namespace) -> None:
    profiles = AgentRepository(args.repository).list()
    for profile in profiles:
        print(
            f"{profile.agent_id:24} age={profile.age:3d} {profile.group_id:28} "
            f"contact={profile.social_contact_frequency:.2f} visual={profile.visual_detection_skill:.2f}"
        )


def _agent_add(args: argparse.Namespace) -> None:
    profile = AgentRepository(args.repository).add_from_file(args.file, args.overwrite)
    print(f"Added agent: {profile.agent_id}")


def _agent_remove(args: argparse.Namespace) -> None:
    AgentRepository(args.repository).remove(args.agent_id)
    print(f"Removed agent: {args.agent_id}")


def _load_interventions(path: str | None) -> list[InterventionEvent]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [InterventionEvent(**item) for item in data]


def _export(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.database)
    tables = [r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    selected = tables if args.tables == "all" else [x.strip() for x in args.tables.split(",")]
    for table in selected:
        if table not in tables:
            raise SystemExit(f"Unknown table: {table}")
        cursor = connection.execute(f'SELECT * FROM "{table}"')
        with (output_dir / f"{table}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow([d[0] for d in cursor.description])
            writer.writerows(cursor)
    connection.close()
    print(f"Exported {len(selected)} tables to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cogmap-sim", description="Older-adult dynamic cognitive-map simulator")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a reproducible experiment")
    run.add_argument("--output", default="output/demo.db")
    run.add_argument("--days", type=int, default=30)
    run.add_argument("--agents", type=int, default=3)
    run.add_argument("--agent-ids", help="comma-separated IDs from the agent repository")
    run.add_argument("--agent-repository", default="agents")
    run.add_argument("--llm-provider", choices=["doubao", "openai", "rule_based", "deterministic"], default="doubao")
    run.add_argument("--llm-model", default="doubao-seed-2-1-pro-260628")
    run.add_argument("--map-json", help="map cache created by import-shp; omit for built-in demo")
    run.add_argument("--interventions", help="JSON intervention list used with --map-json")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--model", choices=["dynamic_cognitive", "static_partial", "omniscient", "objective_accessibility", "no_change", "cognitive", "accessibility"], default="dynamic_cognitive")
    run.add_argument("--no-vision", action="store_true", help="mechanism ablation: disable directional visual discovery")
    run.add_argument("--no-direct-experience", action="store_true", help="mechanism ablation: disable information from direct encounters")
    run.add_argument(
        "--no-face-to-face", "--no-social", dest="no_face_to_face", action="store_true",
        help="mechanism ablation: disable co-presence-based face-to-face information",
    )
    run.add_argument("--experiment-id")
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(func=_run)

    compare = sub.add_parser("compare", help="run the documented B0-B1-M research suite")
    compare.add_argument("--output-dir", default="output/comparison")
    compare.add_argument("--days", type=int, default=30)
    compare.add_argument("--agents", type=int, default=3)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--agent-repository", default="agents")
    compare.add_argument("--llm-provider", choices=["doubao", "openai", "rule_based", "deterministic"], default="rule_based")
    compare.add_argument("--llm-model", default="doubao-seed-2-1-pro-260628")
    compare.add_argument("--map-json", help="run B0, B1 and M on the same imported map")
    compare.add_argument("--interventions", help="shared intervention JSON used with --map-json")
    compare.set_defaults(func=_compare)

    shp = sub.add_parser("import-shp", help="import roads/AOIs from SHP and build graph topology")
    shp.add_argument("--roads", required=True, help="road polyline SHP")
    shp.add_argument("--aois", required=True, help="AOI point or polygon SHP")
    shp.add_argument("--nodes", help="optional road-node point SHP")
    shp.add_argument("--entrances", help="optional AOI entrance point SHP")
    shp.add_argument("--occluders", help="optional building/occluder polygon SHP")
    shp.add_argument("--signage", help="optional signage point SHP")
    shp.add_argument("--output", default="config/imported_map.json")
    shp.add_argument("--snap-tolerance", type=float, default=0.001)
    shp.add_argument("--field-map", help="optional JSON mapping from logical to SHP field names")
    shp.set_defaults(func=_import_shp)

    agent_list = sub.add_parser("agent-list", help="list agents in the repository")
    agent_list.add_argument("--repository", default="agents")
    agent_list.set_defaults(func=_agent_list)
    agent_add = sub.add_parser("agent-add", help="add a JSON agent profile")
    agent_add.add_argument("--repository", default="agents")
    agent_add.add_argument("--file", required=True)
    agent_add.add_argument("--overwrite", action="store_true")
    agent_add.set_defaults(func=_agent_add)
    agent_remove = sub.add_parser("agent-remove", help="remove an agent by ID")
    agent_remove.add_argument("--repository", default="agents")
    agent_remove.add_argument("agent_id")
    agent_remove.set_defaults(func=_agent_remove)

    serve = sub.add_parser("serve", help="serve the local experiment dashboard")
    serve.add_argument("--database", default="output/demo.db")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=lambda a: serve_dashboard(a.database, a.host, a.port))

    export = sub.add_parser("export", help="export SQLite research tables to CSV")
    export.add_argument("--database", default="output/demo.db")
    export.add_argument("--output-dir", default="output/csv")
    export.add_argument("--tables", default="all", help="all or comma-separated table names")
    export.set_defaults(func=_export)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
