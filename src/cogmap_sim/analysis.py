from __future__ import annotations

import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Any


def compare_model_databases(
    databases: dict[str, str | Path], output_directory: str | Path,
) -> dict[str, Any]:
    """Create the paired B0-B1-M outputs required by RQ3.

    The function never reruns behaviour. It compares already completed runs
    that share the same map, intervention file, population and seed.
    """

    paths = {key.upper(): Path(value).resolve() for key, value in databases.items()}
    required = {"B0", "B1", "M"}
    missing = required - paths.keys()
    if missing:
        raise ValueError(f"RQ3 comparison requires B0, B1 and M databases; missing {sorted(missing)}")
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{key} database does not exist: {path}")

    end_day = min(_end_day(path) for path in paths.values())
    rows: list[dict[str, Any]] = []
    for day in range(1, end_day + 1):
        b0_access = _scalar(paths["B0"], "SELECT AVG(normalized_accessibility) FROM objective_accessibility_daily WHERE simulation_day=?", day)
        b1_success = _scalar(paths["B1"], "SELECT AVG(activity_success_rate) FROM route_outcome_daily WHERE simulation_day=?", day)
        rows.append(_row(day, "B0", "B1", "opportunity_vs_realized_activity", b0_access, b1_success,
                         "B0 objective opportunity minus B1 realized activity success"))

        for metric, aggregate, note in (
            ("mean_travel_time", "AVG(total_travel_time)", "daily travel-time difference"),
            ("route_failures_per_agent", "AVG(route_failures)", "route failures per older adult"),
            ("activity_success_rate", "AVG(activity_success_rate)", "successful activity share"),
            ("intervention_use_per_agent", "AVG(intervention_use_count)", "use of changed roads or places"),
            ("route_difference_ratio", "AVG(route_difference_ratio)", "planned-versus-executed route divergence"),
        ):
            table = "agent_daily_state" if metric == "mean_travel_time" else "route_outcome_daily"
            b1 = _scalar(paths["B1"], f"SELECT {aggregate} FROM {table} WHERE simulation_day=?", day)
            dynamic = _scalar(paths["M"], f"SELECT {aggregate} FROM {table} WHERE simulation_day=?", day)
            rows.append(_row(day, "B1", "M", metric, b1, dynamic, note))

        b1_distribution = _destination_distribution(paths["B1"], day)
        dynamic_distribution = _destination_distribution(paths["M"], day)
        js = _jensen_shannon(b1_distribution, dynamic_distribution)
        rows.append(_row(day, "B1", "M", "destination_js_divergence", 0.0, js,
                         "0 means identical destination shares; larger values mean stronger redistribution difference"))

    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "rq3_model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "simulation_day", "research_question", "model_a", "model_b",
            "metric_name", "value_a", "value_b", "difference", "notes",
        ])
        writer.writeheader()
        writer.writerows(rows)

    stability = {
        model: _stability_day(path, end_day)
        for model, path in (("B1", paths["B1"]), ("M", paths["M"]))
    }
    summary = {
        "databases": {key: str(value) for key, value in paths.items()},
        "paired_days": end_day,
        "stability_day": stability,
        "comparison_csv": str(csv_path),
        "interpretation": {
            "B0-B1": "个体活动过程相对客观空间机会增加了多少解释。",
            "B1-M": "有限信息和逐日认知更新相对全知 Agent 改变了多少行为预测。",
        },
    }
    json_path = output_directory / "research_suite_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for database in paths.values():
        with sqlite3.connect(database) as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO model_comparison VALUES (?,?,?,?,?,?,?,?,?)",
                [tuple(row.values()) for row in rows],
            )
            connection.commit()
    return summary


def _row(
    day: int, model_a: str, model_b: str, metric: str,
    value_a: float, value_b: float, notes: str,
) -> dict[str, Any]:
    return {
        "simulation_day": day,
        "research_question": "RQ3",
        "model_a": model_a,
        "model_b": model_b,
        "metric_name": metric,
        "value_a": value_a,
        "value_b": value_b,
        "difference": value_b - value_a,
        "notes": notes,
    }


def _end_day(path: Path) -> int:
    with sqlite3.connect(path) as connection:
        return int(connection.execute("SELECT end_day FROM experiment_config LIMIT 1").fetchone()[0])


def _scalar(path: Path, query: str, day: int) -> float:
    with sqlite3.connect(path) as connection:
        value = connection.execute(query, (day,)).fetchone()[0]
    return 0.0 if value is None else float(value)


def _destination_distribution(path: Path, day: int) -> dict[str, float]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT aoi_id, SUM(visit_count) FROM activity_distribution_daily "
            "WHERE simulation_day=? GROUP BY aoi_id",
            (day,),
        ).fetchall()
    total = sum(float(row[1]) for row in rows)
    return {str(row[0]): float(row[1]) / total for row in rows} if total else {}


def _jensen_shannon(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    midpoint = {key: (left.get(key, 0.0) + right.get(key, 0.0)) / 2.0 for key in keys}

    def divergence(values: dict[str, float]) -> float:
        return sum(
            probability * math.log(probability / midpoint[key], 2)
            for key, probability in values.items()
            if probability > 0 and midpoint[key] > 0
        )

    return math.sqrt(max(0.0, 0.5 * divergence(left) + 0.5 * divergence(right)))


def _stability_day(path: Path, end_day: int, tolerance: float = 0.05, consecutive_days: int = 3) -> int | None:
    previous: dict[str, float] | None = None
    stable_run = 0
    for day in range(1, end_day + 1):
        current = _destination_distribution(path, day)
        if previous is None:
            previous = current
            continue
        distance = _jensen_shannon(previous, current)
        stable_run = stable_run + 1 if distance <= tolerance else 0
        if stable_run >= consecutive_days:
            return day - consecutive_days + 1
        previous = current
    return None
