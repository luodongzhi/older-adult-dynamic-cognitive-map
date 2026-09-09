from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from .graph import UrbanGraph
from .models import AgentProfile, ExecutionResult


@dataclass(slots=True)
class TrajectorySample:
    event_time: float
    x: float
    y: float
    heading_deg: float
    edge_id: str
    progress: float


@dataclass(slots=True)
class VisibilityResult:
    detected: bool
    probability: float
    draw: float
    distance: float
    relative_angle_deg: float
    line_of_sight: bool
    sample: TrajectorySample | None


def _angular_difference(target: float, heading: float) -> float:
    return (target - heading + 180.0) % 360.0 - 180.0


def _heading(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360.0


class DirectionalVisionEngine:
    """Two-dimensional field-of-view and line-of-sight perception.

    Shapely performs the geometric operations. LLMs never participate in
    visibility, object recognition candidates, or map writes.
    """

    def trajectory_samples(
        self,
        execution: ExecutionResult,
        origin_node: str,
        graph: UrbanGraph,
        profile: AgentProfile,
    ) -> list[TrajectorySample]:
        try:
            from shapely.geometry import LineString
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("Directional vision requires Shapely>=2.0. Install with: pip install -e .[gis]") from exc

        samples: list[TrajectorySample] = []
        current_node = origin_node
        scale = max(1e-6, float(graph.metadata.get("distance_scale", 1.0)))
        spacing = max(scale * profile.visual_sample_spacing_ratio, scale * 0.04)
        for step in execution.steps:
            if step.status == "BLOCKED" or step.edge_id not in graph.edges:
                continue
            edge = graph.edges[step.edge_id]
            coords = list(graph._edge_coordinates(edge))
            current = graph.nodes[current_node]
            first_distance = math.hypot(coords[0][0] - current.x, coords[0][1] - current.y)
            last_distance = math.hypot(coords[-1][0] - current.x, coords[-1][1] - current.y)
            if last_distance < first_distance:
                coords.reverse()
            line = LineString(coords)
            length = max(line.length, 1e-9)
            count = max(1, int(math.ceil(length / spacing)))
            for index in range(count + 1):
                progress = index / count
                distance = progress * length
                point = line.interpolate(distance)
                delta = min(max(length * 0.01, 0.05), max(length / 3, 0.05))
                before = line.interpolate(max(0.0, distance - delta))
                after = line.interpolate(min(length, distance + delta))
                heading = _heading(before.x, before.y, after.x, after.y)
                event_time = step.enter_time + progress * (step.exit_time - step.enter_time)
                sample = TrajectorySample(event_time, point.x, point.y, heading, step.edge_id, progress)
                if not samples or math.hypot(sample.x - samples[-1].x, sample.y - samples[-1].y) > 1e-7:
                    samples.append(sample)
            current_node = graph.other_node(step.edge_id, current_node)
        return samples

    def object_geometry(self, graph: UrbanGraph, object_type: str, object_id: str):
        from shapely.geometry import LineString, Point, Polygon

        if object_type == "ROAD_EDGE":
            return LineString(graph._edge_coordinates(graph.edges[object_id]))
        if object_type == "AOI":
            aoi = graph.aois[object_id]
            if len(aoi.geometry) >= 3:
                polygon = Polygon(aoi.geometry)
                return polygon if polygon.is_valid else polygon.buffer(0)
            if aoi.geometry:
                return Point(float(aoi.geometry[0][0]), float(aoi.geometry[0][1]))
            node = graph.nodes[aoi.node_id]
            return Point(node.x, node.y)
        if object_type == "ENTRANCE":
            entrance = graph.entrances[object_id]
            return Point(entrance.x, entrance.y)
        raise ValueError(f"Unsupported visible object: {object_type}/{object_id}")

    def evaluate(
        self,
        samples: list[TrajectorySample],
        target_geometry: Any,
        graph: UrbanGraph,
        profile: AgentProfile,
        salience: float,
        rng: random.Random,
    ) -> VisibilityResult:
        from shapely.geometry import LineString, Point, Polygon
        from shapely.ops import nearest_points

        if not samples:
            return VisibilityResult(False, 0.0, 1.0, math.inf, 180.0, False, None)
        scale = max(1e-6, float(graph.metadata.get("distance_scale", 1.0)))
        view_range = max(scale * profile.visual_search_radius, scale * 0.05)
        half_fov = max(1.0, min(179.0, profile.visual_field_of_view_deg / 2.0))
        occluders = []
        for item in graph.occluders.values():
            if item.opacity <= 0 or len(item.geometry) < 2:
                continue
            geometry = Polygon(item.geometry) if len(item.geometry) >= 3 else LineString(item.geometry)
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            occluders.append((item, geometry))

        best: tuple[float, float, float, bool, TrajectorySample] | None = None
        for sample in samples:
            observer = Point(sample.x, sample.y)
            _, target_point = nearest_points(observer, target_geometry)
            distance = observer.distance(target_point)
            target_heading = sample.heading_deg if distance <= 1e-8 else _heading(sample.x, sample.y, target_point.x, target_point.y)
            relative = 0.0 if distance <= 1e-8 else _angular_difference(target_heading, sample.heading_deg)
            if distance > view_range or abs(relative) > half_fov:
                continue
            sight_line = LineString([(sample.x, sample.y), (target_point.x, target_point.y)])
            los = True
            if distance > 1e-8:
                for item, occluder in occluders:
                    if occluder.distance(observer) < 0.25 or occluder.distance(target_point) < 0.25:
                        continue
                    if sight_line.crosses(occluder) or sight_line.within(occluder) or sight_line.intersection(occluder).length > 0.25:
                        if item.opacity >= 0.5:
                            los = False
                            break
            if not los:
                probability = 0.0
            else:
                distance_factor = math.exp(-1.35 * distance / view_range)
                angle_factor = max(0.0, math.cos(math.radians(abs(relative)))) ** 0.75
                probability = (
                    salience * profile.visual_attention * profile.visual_detection_skill
                    * distance_factor * angle_factor
                )
                if distance <= max(0.5, scale * 0.01):
                    probability = max(probability, 0.98)
                probability = max(0.0, min(0.99, probability))
            candidate = (probability, distance, relative, los, sample)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            return VisibilityResult(False, 0.0, 1.0, math.inf, 180.0, False, None)
        probability, distance, relative, los, sample = best
        draw = rng.random()
        return VisibilityResult(los and draw < probability, probability, draw, distance, relative, los, sample)
