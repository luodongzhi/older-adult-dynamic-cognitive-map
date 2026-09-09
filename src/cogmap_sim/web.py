from __future__ import annotations

import json
import html as html_module
import math
import mimetypes
import sqlite3
import socket
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any


class ExclusiveThreadingHTTPServer(ThreadingHTTPServer):
    """Prevent two stale dashboards from silently sharing the same port."""

    allow_reuse_address = False
    allow_reuse_port = False

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def _rows(connection: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(sql, params)]


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _agent_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = _rows(connection, "SELECT agent_id, group_id, profile_json FROM agent_profile ORDER BY agent_id")
    for row in rows:
        profile = _json_value(row.pop("profile_json", None), {})
        row.update({
            "age": profile.get("age"),
            "mobility_aid": profile.get("mobility_aid"),
            "years_residence": profile.get("years_residence"),
            "route_habit": profile.get("route_habit"),
            "social_participation": profile.get("social_participation"),
        })
    return rows


def dashboard_payload(database: str | Path) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    config = _rows(connection, "SELECT * FROM experiment_config")[0]
    completed_day = connection.execute(
        "SELECT COALESCE(MAX(simulation_day), 0) FROM environment_daily_state"
    ).fetchone()[0]
    metrics = dict(connection.execute("""
        SELECT
          (SELECT COUNT(*) FROM agent_profile) agents,
          (SELECT COUNT(*) FROM activity_episode_log) episodes,
          (SELECT COUNT(*) FROM observation_event_log) observations,
          (SELECT COUNT(*) FROM cognitive_update_log) updates,
          (SELECT COALESCE(SUM(failed_route_count),0) FROM agent_daily_state) route_failures,
          (SELECT COUNT(DISTINCT agent_id || ':' || simulation_day) FROM executed_trajectory_log WHERE edge_id='NEW_LINK') new_link_uses,
          (SELECT COUNT(*) FROM llm_decision_log) llm_calls
    """).fetchone())
    metrics["copresence_events"] = connection.execute("SELECT COUNT(*) FROM copresence_event").fetchone()[0] if _table_exists(connection, "copresence_event") else 0
    metrics["conversations"] = connection.execute("SELECT COUNT(*) FROM copresence_event WHERE conversation_flag=1").fetchone()[0] if _table_exists(connection, "copresence_event") else 0
    metrics["delivered_interpersonal_reports"] = connection.execute("SELECT COUNT(*) FROM interpersonal_exchange WHERE delivered=1").fetchone()[0] if _table_exists(connection, "interpersonal_exchange") else 0
    metrics["intervention_road_uses"] = connection.execute(
        "SELECT COUNT(DISTINCT agent_id || ':' || simulation_day) FROM executed_trajectory_log "
        "WHERE edge_id IN (SELECT target_id FROM intervention_log WHERE operation_type LIKE 'ROAD_%')"
    ).fetchone()[0]
    metrics["information_arrivals"] = connection.execute("SELECT COUNT(*) FROM information_item").fetchone()[0] if _table_exists(connection, "information_item") else 0
    nodes = _rows(connection, "SELECT * FROM node")
    node_lookup = {node["node_id"]: node for node in nodes}
    edges = _rows(connection, "SELECT * FROM road_edge")
    for edge in edges:
        geometry = _json_value(edge.pop("geometry_json", None), [])
        if not geometry:
            start, end = node_lookup[edge["from_node"]], node_lookup[edge["to_node"]]
            geometry = [[start["x"], start["y"]], [end["x"], end["y"]]]
        edge["geometry"] = geometry
    aois = _rows(connection, "SELECT * FROM aoi_object")
    for aoi in aois:
        geometry = _json_value(aoi.pop("geometry_json", None), [])
        if not geometry:
            node = node_lookup[aoi["node_id"]]
            geometry = [[node["x"], node["y"]]]
        aoi["geometry"] = geometry
    payload = {
        "database_path": str(Path(database).resolve()),
        "config": config,
        "available_day_max": int(completed_day or 0),
        "run_complete": int(completed_day or 0) >= int(config["end_day"]),
        "metrics": metrics,
        "agents": _agent_rows(connection),
        "nodes": nodes,
        "edges": edges,
        "aois": aois,
        "entrances": _rows(connection, "SELECT * FROM entrance_object") if _table_exists(connection, "entrance_object") else [],
        "occluders": _rows(connection, "SELECT * FROM occluder_object") if _table_exists(connection, "occluder_object") else [],
        "signage": _rows(connection, "SELECT * FROM signage_object") if _table_exists(connection, "signage_object") else [],
        "interventions": _rows(connection, "SELECT * FROM intervention_log"),
        "road": _rows(connection, "SELECT * FROM road_daily_aggregate ORDER BY simulation_day, edge_id"),
        "aoi_activity": _rows(connection, "SELECT a.*, o.name FROM aoi_daily_aggregate a JOIN aoi_object o USING(aoi_id) ORDER BY simulation_day, aoi_id"),
        "groups": _rows(connection, "SELECT * FROM group_daily_aggregate ORDER BY simulation_day, group_id"),
        "channels": _rows(connection, "SELECT source_channel, COUNT(*) count FROM observation_event_log GROUP BY source_channel ORDER BY count DESC"),
        "updates_by_day": _rows(connection, "SELECT simulation_day, COUNT(*) count FROM cognitive_update_log GROUP BY simulation_day ORDER BY simulation_day"),
        "llm_phases": _rows(connection, "SELECT phase, provider, model, COUNT(*) count, AVG(latency_ms) average_latency_ms FROM llm_decision_log GROUP BY phase, provider, model ORDER BY phase"),
        "mismatch": _rows(connection, "SELECT * FROM mismatch_daily ORDER BY simulation_day, object_id") if _table_exists(connection, "mismatch_daily") else [],
        "cognitive_lag_population": _rows(connection, "SELECT * FROM cognitive_lag_population_daily ORDER BY simulation_day, intervention_id") if _table_exists(connection, "cognitive_lag_population_daily") else [],
        "daily_metrics": _rows(connection, "SELECT * FROM daily_metric ORDER BY simulation_day, research_question, level_type, level_id, metric_name") if _table_exists(connection, "daily_metric") else [],
        "information_process": _rows(connection, "SELECT * FROM information_process_daily ORDER BY simulation_day, agent_id") if _table_exists(connection, "information_process_daily") else [],
        "route_outcomes": _rows(connection, "SELECT * FROM route_outcome_daily ORDER BY simulation_day, agent_id") if _table_exists(connection, "route_outcome_daily") else [],
        "activity_distribution": _rows(connection, "SELECT * FROM activity_distribution_daily ORDER BY simulation_day, activity_type, aoi_id") if _table_exists(connection, "activity_distribution_daily") else [],
        "objective_accessibility": _rows(connection, "SELECT * FROM objective_accessibility_daily ORDER BY simulation_day, origin_aoi_id, activity_type") if _table_exists(connection, "objective_accessibility_daily") else [],
        "model_comparison": _rows(connection, "SELECT * FROM model_comparison ORDER BY simulation_day, model_a, model_b, metric_name") if _table_exists(connection, "model_comparison") else [],
        "copresence": _rows(connection, "SELECT * FROM copresence_event ORDER BY simulation_day, overlap_start") if _table_exists(connection, "copresence_event") else [],
    }
    connection.close()
    return payload


def cognitive_payload(database: str | Path, agent_id: str, day: int) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    profile_rows = _rows(connection, "SELECT agent_id, group_id, profile_json FROM agent_profile WHERE agent_id=?", (agent_id,))
    if not profile_rows:
        connection.close()
        raise ValueError(f"Unknown agent: {agent_id}")
    config = _rows(connection, "SELECT end_day FROM experiment_config")[0]
    day = max(0, min(int(day), int(config["end_day"])))
    snapshot = connection.execute("SELECT MAX(simulation_day) FROM cognitive_edge_daily_state WHERE agent_id=? AND simulation_day<=?", (agent_id, day)).fetchone()[0]
    snapshot = day if snapshot is None else int(snapshot)
    edges = _rows(connection, """
        SELECT r.edge_id, r.from_node, r.to_node, r.is_intervention,
               e.actual_value AS actual_status, c.believed_status,
               c.information_source, c.first_known_day, c.last_update_day,
               c.adoption_state, c.use_count
        FROM road_edge r
        LEFT JOIN environment_daily_state e ON e.object_id=r.edge_id AND e.object_type='ROAD_EDGE' AND e.simulation_day=?
        LEFT JOIN cognitive_edge_daily_state c ON c.edge_id=r.edge_id AND c.agent_id=? AND c.simulation_day=?
        ORDER BY r.edge_id
    """, (day, agent_id, snapshot))
    aois = _rows(connection, """
        SELECT a.aoi_id, a.name, a.node_id, a.is_intervention,
               e.actual_value AS actual_function, c.believed_function,
               c.perceived_attractiveness, c.information_source,
               c.first_known_day, c.last_update_day,
               c.adoption_state, c.visit_count
        FROM aoi_object a
        LEFT JOIN environment_daily_state e ON e.object_id=a.aoi_id AND e.object_type='AOI' AND e.simulation_day=?
        LEFT JOIN cognitive_aoi_daily_state c ON c.aoi_id=a.aoi_id AND c.agent_id=? AND c.simulation_day=?
        ORDER BY a.aoi_id
    """, (day, agent_id, snapshot))
    entrances = _rows(connection, """
        SELECT e.entrance_id, e.aoi_id, e.node_id, e.x, e.y, e.is_intervention,
               s.actual_value AS actual_status, c.believed_status,
               c.information_source, c.first_known_day, c.last_update_day
        FROM entrance_object e
        LEFT JOIN environment_daily_state s ON s.object_id=e.entrance_id AND s.object_type='ENTRANCE' AND s.simulation_day=?
        LEFT JOIN cognitive_entrance_daily_state c ON c.entrance_id=e.entrance_id AND c.agent_id=? AND c.simulation_day=?
        ORDER BY e.entrance_id
    """, (day, agent_id, snapshot)) if _table_exists(connection, "entrance_object") else []
    llm = _rows(connection, "SELECT call_id, phase, provider, model, response_json, latency_ms, status FROM llm_decision_log WHERE agent_id=? AND simulation_day=? ORDER BY phase", (agent_id, day))
    for item in llm:
        item["response"] = _json_value(item.pop("response_json"), {})
    observation_columns = {row[1] for row in connection.execute("PRAGMA table_info(observation_event_log)")}
    diagnostic_select = ", observer_x, observer_y, heading_deg, relative_angle_deg, line_of_sight, detection_probability, detection_draw, source_event_id, independence_group, source_agent_id, copresence_event_id, first_hand_source" if "independence_group" in observation_columns else (", observer_x, observer_y, heading_deg, relative_angle_deg, line_of_sight, detection_probability, detection_draw, source_event_id" if "heading_deg" in observation_columns else "")
    observations = _rows(connection, f"SELECT event_time, episode_id, object_id, object_type, source_channel, observed_attribute, observed_value, source_reliability, salience{diagnostic_select} FROM observation_event_log WHERE agent_id=? AND simulation_day=? ORDER BY event_time", (agent_id, day))
    updates = _rows(connection, "SELECT event_time, object_id, object_type, attribute_name, old_value, new_value, update_reason, source_event_id, source_agent_id, first_hand_source FROM cognitive_update_log WHERE agent_id=? AND simulation_day=? ORDER BY event_time, object_id", (agent_id, day))
    information = _rows(connection, "SELECT information_id, object_id, object_type, attribute_name, observed_value, source_channel, source_event_id, reliability, salience, source_agent_id, first_hand_source FROM information_item WHERE agent_id=? AND simulation_day=? ORDER BY object_id", (agent_id, day)) if _table_exists(connection, "information_item") else []
    perception = _rows(connection, "SELECT * FROM perception_diagnostic_log WHERE agent_id=? AND simulation_day=? ORDER BY sample_time", (agent_id, day)) if _table_exists(connection, "perception_diagnostic_log") else []
    planned = _rows(connection, "SELECT episode_id, sequence_index, edge_id FROM planned_trajectory_log WHERE agent_id=? AND simulation_day=? ORDER BY episode_id, sequence_index", (agent_id, day))
    executed = _rows(connection, """
        SELECT x.episode_id, x.sequence_index, x.edge_id, x.execution_status,
               x.reroute_id, x.enter_time, x.exit_time,
               a.origin_node, a.sequence_number, a.activity_type
        FROM executed_trajectory_log x
        LEFT JOIN activity_episode_log a
          ON a.simulation_day=x.simulation_day AND a.agent_id=x.agent_id AND a.episode_id=x.episode_id
        WHERE x.agent_id=? AND x.simulation_day=?
        ORDER BY a.sequence_number, x.sequence_index
    """, (agent_id, day))
    comparisons = _rows(connection, """
        SELECT p.agent_id, p.group_id,
          (SELECT COUNT(*) FROM cognitive_edge_daily_state ce WHERE ce.agent_id=p.agent_id AND ce.simulation_day=?) known_edges,
          (SELECT COUNT(*) FROM cognitive_aoi_daily_state ca WHERE ca.agent_id=p.agent_id AND ca.simulation_day=?) known_aois,
          (SELECT COUNT(*) FROM cognitive_update_log cu WHERE cu.agent_id=p.agent_id AND cu.simulation_day<=?) cumulative_updates,
          (SELECT AVG(lag_indicator) FROM cognitive_lag_daily cl WHERE cl.agent_id=p.agent_id AND cl.simulation_day=?) cognitive_lag_index,
          (SELECT COUNT(*) FROM copresence_event cp WHERE cp.simulation_day<=? AND (cp.agent_a=p.agent_id OR cp.agent_b=p.agent_id)) copresence_count,
          (SELECT COUNT(*) FROM interpersonal_exchange ie WHERE ie.listener_agent_id=p.agent_id AND ie.simulation_day<=? AND ie.delivered=1) delivered_reports,
          p.profile_json
        FROM agent_profile p ORDER BY p.agent_id
    """, (snapshot, snapshot, day, day, day, day))
    for item in comparisons:
        profile = _json_value(item.pop("profile_json", None), {})
        item["age"] = profile.get("age")
        item["route_habit"] = profile.get("route_habit")
        item["social_participation"] = profile.get("social_participation")
    lag = _rows(connection, "SELECT * FROM cognitive_lag_daily WHERE agent_id=? AND simulation_day=? ORDER BY intervention_id", (agent_id, day)) if _table_exists(connection, "cognitive_lag_daily") else []
    timing = _rows(connection, "SELECT * FROM update_timing WHERE agent_id=? ORDER BY intervention_day, intervention_id", (agent_id,)) if _table_exists(connection, "update_timing") else []
    copresence = _rows(connection, "SELECT * FROM copresence_event WHERE simulation_day=? AND (agent_a=? OR agent_b=?) ORDER BY overlap_start", (day, agent_id, agent_id)) if _table_exists(connection, "copresence_event") else []
    exchanges = _rows(connection, "SELECT * FROM interpersonal_exchange WHERE simulation_day=? AND (speaker_agent_id=? OR listener_agent_id=?) ORDER BY listener_agent_id, object_id", (day, agent_id, agent_id)) if _table_exists(connection, "interpersonal_exchange") else []
    connection.close()
    return {
        "agent_id": agent_id, "day": day, "snapshot_day": snapshot,
        "profile": _json_value(profile_rows[0]["profile_json"], {}),
        "edges": edges, "aois": aois, "entrances": entrances, "llm": llm,
        "observations": observations, "updates": updates, "information": information,
        "perception_diagnostics": perception,
        "planned": planned, "executed": executed, "agent_comparison": comparisons,
        "cognitive_lag": lag, "update_timing": timing,
        "copresence": copresence, "interpersonal_exchanges": exchanges,
    }


def collective_payload(database: str | Path, day: int) -> dict[str, Any]:
    """Return every agent's timed route for one shared-map dashboard view."""
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    config = _rows(connection, "SELECT end_day FROM experiment_config")[0]
    completed_day = connection.execute(
        "SELECT COALESCE(MAX(simulation_day), 0) FROM environment_daily_state"
    ).fetchone()[0]
    day = max(0, min(int(day), int(config["end_day"]), int(completed_day or 0)))
    agents = _rows(connection, """
        SELECT p.agent_id, p.group_id, p.home_aoi_id, a.node_id AS home_node
        FROM agent_profile p
        LEFT JOIN aoi_object a ON a.aoi_id=p.home_aoi_id
        ORDER BY p.agent_id
    """)
    executed = _rows(connection, """
        SELECT x.agent_id, x.episode_id, x.sequence_index, x.edge_id,
               x.execution_status, x.reroute_id, x.enter_time, x.exit_time,
               a.origin_node, a.sequence_number, a.activity_type
        FROM executed_trajectory_log x
        LEFT JOIN activity_episode_log a
          ON a.simulation_day=x.simulation_day
         AND a.agent_id=x.agent_id
         AND a.episode_id=x.episode_id
        WHERE x.simulation_day=?
        ORDER BY x.agent_id, a.sequence_number, x.sequence_index
    """, (day,))
    stops = _rows(connection, "SELECT * FROM activity_stop_log WHERE simulation_day=? ORDER BY start_time", (day,)) if _table_exists(connection, "activity_stop_log") else []
    copresence = _rows(connection, "SELECT * FROM copresence_event WHERE simulation_day=? ORDER BY overlap_start", (day,)) if _table_exists(connection, "copresence_event") else []
    connection.close()
    return {"day": day, "agents": agents, "executed": executed, "stops": stops, "copresence": copresence}


def agent_payload(database: str | Path, agent_id: str) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    payload = {
        "profile": (_rows(connection, "SELECT * FROM agent_profile WHERE agent_id=?", (agent_id,)) or [{}])[0],
        "states": _rows(connection, "SELECT * FROM agent_daily_state WHERE agent_id=? ORDER BY simulation_day", (agent_id,)),
        "episodes": _rows(connection, "SELECT * FROM activity_episode_log WHERE agent_id=? ORDER BY simulation_day, sequence_number", (agent_id,)),
        "observations": _rows(connection, "SELECT * FROM observation_event_log WHERE agent_id=? ORDER BY simulation_day, event_time", (agent_id,)),
        "updates": _rows(connection, "SELECT * FROM cognitive_update_log WHERE agent_id=? ORDER BY simulation_day", (agent_id,)),
        "copresence": _rows(connection, "SELECT * FROM copresence_event WHERE agent_a=? OR agent_b=? ORDER BY simulation_day, overlap_start", (agent_id, agent_id)) if _table_exists(connection, "copresence_event") else [],
        "interpersonal_exchanges": _rows(connection, "SELECT * FROM interpersonal_exchange WHERE speaker_agent_id=? OR listener_agent_id=? ORDER BY simulation_day", (agent_id, agent_id)) if _table_exists(connection, "interpersonal_exchange") else [],
        "cognitive_lag": _rows(connection, "SELECT * FROM cognitive_lag_daily WHERE agent_id=? ORDER BY simulation_day, intervention_id", (agent_id,)) if _table_exists(connection, "cognitive_lag_daily") else [],
    }
    connection.close()
    return payload


def list_dashboard_databases(directory: str | Path) -> list[dict[str, Any]]:
    """List readable experiment databases without allowing paths outside directory."""
    directory = Path(directory).resolve()
    results: list[dict[str, Any]] = []
    for database in sorted(directory.glob("*.db"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            with sqlite3.connect(database) as connection:
                connection.row_factory = sqlite3.Row
                tables = {
                    row[0] for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                required = {
                    "copresence_event", "cognitive_lag_daily",
                    "information_process_daily", "information_item",
                }
                if not required.issubset(tables):
                    continue
                config = dict(connection.execute("SELECT * FROM experiment_config LIMIT 1").fetchone())
                agents = connection.execute("SELECT COUNT(*) FROM agent_profile").fetchone()[0]
                completed_days = connection.execute(
                    "SELECT COALESCE(MAX(simulation_day),0) FROM environment_daily_state"
                ).fetchone()[0]
            results.append({
                "name": database.name,
                "path": str(database),
                "scenario_id": config.get("scenario_id", ""),
                "experiment_id": config.get("experiment_id", ""),
                "model_type": config.get("model_type", ""),
                "days": config.get("end_day", 0),
                "completed_days": int(completed_days or 0),
                "agents": agents,
                "behavior_driver": config.get("behavior_driver", ""),
                "llm_model": config.get("llm_model", ""),
                "modified_time": database.stat().st_mtime,
                "size_bytes": database.stat().st_size,
            })
        except (sqlite3.Error, TypeError):
            continue
    return results


def export_dashboard_report(database: str | Path, destination: str | Path | None = None) -> Path:
    """Create a self-contained HTML report for every agent and simulated day."""
    database = Path(database).resolve()
    destination = Path(destination) if destination else database.with_name(f"{database.stem}_dashboard.html")
    base = dashboard_payload(database)
    cognitive: dict[str, dict[str, Any]] = {}
    end_day = int(base["available_day_max"])
    for agent in base["agents"]:
        agent_id = agent["agent_id"]
        cognitive[agent_id] = {
            str(day): cognitive_payload(database, agent_id, day)
            for day in range(0, end_day + 1)
        }
    collective = {
        str(day): collective_payload(database, day)
        for day in range(0, end_day + 1)
    }
    embedded = json.dumps(
        _json_safe({"dashboard": base, "cognitive": cognitive, "collective": collective}),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    embedded = embedded.replace("</", "<\\/")
    page = files("cogmap_sim").joinpath("static/index.html").read_text(encoding="utf-8")
    page = page.replace("</head>", f"<script>window.COGMAP_EMBEDDED={embedded};</script></head>", 1)
    destination.write_text(page, encoding="utf-8")
    return destination.resolve()


def _selector_html(databases: list[dict[str, Any]], current_name: str = "", standalone: bool = False) -> str:
    cards = []
    for item in databases:
        name = str(item["name"])
        report_name = f"{Path(name).stem}_dashboard.html"
        target = urllib.parse.quote(report_name if standalone else name)
        href = target if standalone else f"/viewer?db={target}"
        current = '<span class="current">本次实验</span>' if name == current_name else ""
        size_mb = float(item["size_bytes"]) / 1024 / 1024
        completed = int(item.get("completed_days", 0))
        configured = int(item.get("days", 0))
        day_fact = f"已完成 {completed} / 配置 {configured} 天"
        day_class = " incomplete" if completed < configured else ""
        cards.append(f"""
          <a class="card" href="{href}">
            <div class="top"><strong>{html_module.escape(name)}</strong>{current}</div>
            <div class="scenario">{html_module.escape(str(item['scenario_id']))}</div>
            <div class="facts"><span>{html_module.escape(str(item.get('model_type', '')))}</span><span class="{day_class.strip()}">{day_fact}</span><span>{item['agents']} Agent</span><span>{size_mb:.1f} MB</span></div>
            <div class="model">{html_module.escape(str(item['behavior_driver']))} · {html_module.escape(str(item['llm_model']))}</div>
            <div class="open">进入认知地图展示 →</div>
          </a>""")
    content = "".join(cards) or '<div class="empty">当前文件夹中没有可展示的实验数据库。</div>'
    mode_note = "这是可离线打开的数据库报告入口。" if standalone else "选择数据库后再进入正式展示页面。"
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>选择实验数据库</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f1f2ed;color:#18312b;font-family:'Segoe UI','Microsoft YaHei',sans-serif}}main{{max-width:1060px;margin:auto;padding:64px 24px}}.eyebrow{{font-size:12px;letter-spacing:2px;font-weight:800;color:#14705e}}h1{{font-family:Georgia,'Songti SC',serif;font-size:44px;margin:10px 0}}.lead{{color:#66736e;margin-bottom:32px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}.card{{display:block;text-decoration:none;color:inherit;background:#fffdf7;border:1px solid #dce0d8;border-radius:16px;padding:22px;box-shadow:0 8px 24px #193b3010;transition:.18s}}.card:hover{{transform:translateY(-3px);border-color:#5c9688}}.top{{display:flex;justify-content:space-between;gap:12px}}strong{{font-size:18px;word-break:break-all}}.current{{white-space:nowrap;background:#dcefe8;color:#11604f;padding:4px 8px;border-radius:999px;font-size:11px}}.scenario{{margin:13px 0;color:#36685c}}.facts{{display:flex;gap:8px;flex-wrap:wrap}}.facts span{{background:#f0f2ed;padding:5px 9px;border-radius:7px;font-size:12px}}.facts .incomplete{{background:#fff0e9;color:#b44e32;font-weight:700}}.model{{font-size:12px;color:#7a837f;margin-top:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.open{{margin-top:20px;font-weight:700;color:#176858}}.empty{{background:white;padding:30px;border-radius:14px}}</style></head><body><main><div class="eyebrow">COGNITIVE MAP EXPERIMENTS</div><h1>选择实验数据库</h1><div class="lead">{mode_note} 共找到 {len(databases)} 个可用的 .db 文件。</div><section class="grid">{content}</section></main></body></html>"""


def export_dashboard_gallery(directory: str | Path, current_database: str | Path | None = None) -> tuple[Path, list[Path]]:
    """Export a database chooser and one standalone report per valid database."""
    directory = Path(directory).resolve()
    current_name = Path(current_database).name if current_database else ""
    databases = list_dashboard_databases(directory)
    reports: list[Path] = []
    exported_databases: list[dict[str, Any]] = []
    for item in databases:
        try:
            reports.append(export_dashboard_report(directory / item["name"]))
            exported_databases.append(item)
        except (sqlite3.Error, KeyError, ValueError):
            continue
    selector = directory / "选择实验数据库.html"
    selector.write_text(_selector_html(exported_databases, current_name, standalone=True), encoding="utf-8")
    return selector, reports


def serve_dashboard(database: str | Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    database = Path(database).resolve()
    if not database.exists():
        raise SystemExit(f"Database does not exist: {database}. Run an experiment first.")
    html = files("cogmap_sim").joinpath("static/index.html").read_bytes()
    database_directory = database.parent

    def selected_database(parsed: urllib.parse.ParseResult) -> Path:
        query = urllib.parse.parse_qs(parsed.query)
        requested = query.get("db", [database.name])[0]
        if Path(requested).name != requested or not requested.lower().endswith(".db"):
            raise ValueError("Invalid database name")
        selected = (database_directory / requested).resolve()
        if selected.parent != database_directory or not selected.exists():
            raise ValueError(f"Database does not exist: {requested}")
        return selected

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            try:
                if parsed.path == "/api/dashboard":
                    self._json(dashboard_payload(selected_database(parsed)))
                elif parsed.path == "/api/databases":
                    self._json({"current": database.name, "databases": list_dashboard_databases(database_directory)})
                elif parsed.path == "/api/cognitive":
                    query = urllib.parse.parse_qs(parsed.query)
                    agent_id = query.get("agent", [""])[0]
                    day = int(query.get("day", ["1"])[0])
                    self._json(cognitive_payload(selected_database(parsed), agent_id, day))
                elif parsed.path == "/api/collective":
                    day = int(urllib.parse.parse_qs(parsed.query).get("day", ["1"])[0])
                    self._json(collective_payload(selected_database(parsed), day))
                elif parsed.path == "/api/agent":
                    agent_id = urllib.parse.parse_qs(parsed.query).get("id", ["A001"])[0]
                    self._json(agent_payload(selected_database(parsed), agent_id))
                elif parsed.path == "/viewer":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                elif parsed.path in ("/", "/index.html"):
                    chooser = _selector_html(list_dashboard_databases(database_directory), database.name).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(chooser)))
                    self.end_headers()
                    self.wfile.write(chooser)
                else:
                    self.send_error(404)
            except Exception as exc:
                self._json({"error": str(exc)}, status=500)

        def _json(self, value: Any, status: int = 200) -> None:
            body = json.dumps(
                _json_safe(value), ensure_ascii=False, allow_nan=False,
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[dashboard] {format % args}")

    try:
        server = ExclusiveThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        raise RuntimeError(f"Dashboard port {host}:{port} is already in use. Stop the previous PyCharm run before starting a new one.") from exc
    print(f"Dashboard: http://{host}:{port}")
    print(f"Database:  {database}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _json_value(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _json_safe(value: Any) -> Any:
    """Replace SQLite infinities/NaN with JSON null for browser compatibility."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
