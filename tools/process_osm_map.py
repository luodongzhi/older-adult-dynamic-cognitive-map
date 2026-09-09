from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import iterparse

import networkx as nx
import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, polygonize, unary_union
from shapely.strtree import STRtree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cogmap_sim.gis import load_graph_json, save_graph_json  # noqa: E402
from cogmap_sim.graph import UrbanGraph  # noqa: E402
from cogmap_sim.models import AOI, Edge, EdgeStatus, Entrance, Node, Occluder  # noqa: E402


WALKABLE_HIGHWAYS = {
    "primary", "primary_link", "secondary", "secondary_link", "tertiary",
    "tertiary_link", "unclassified", "residential", "living_street", "service",
    "pedestrian", "footway", "path", "steps", "cycleway", "corridor", "track",
}
RESIDENTIAL_BUILDINGS = {
    "apartments", "residential", "house", "detached", "terrace", "dormitory",
    "bungalow", "semidetached_house",
}
WORK_BUILDINGS = {
    "office", "commercial", "industrial", "school", "hospital", "university",
    "kindergarten", "public", "civic", "government", "train_station",
}
WORK_AMENITIES = {
    "bank", "school", "kindergarten", "university", "college", "hospital", "clinic",
    "police", "fire_station", "library", "community_centre", "research_institute",
    "theatre", "cinema", "arts_centre", "post_office", "townhall", "courthouse",
}
CAFE_AMENITIES = {"cafe", "restaurant", "fast_food", "food_court", "bar", "pub", "ice_cream"}
PARK_LEISURE = {"park", "garden", "playground", "nature_reserve"}
PARK_LANDUSE = {"forest", "recreation_ground", "village_green"}


@dataclass(slots=True)
class OSMWay:
    osm_id: str
    refs: list[str]
    tags: dict[str, str]


@dataclass(slots=True)
class OSMRelation:
    osm_id: str
    members: list[tuple[str, str, str]]
    tags: dict[str, str]


@dataclass(slots=True)
class AOISource:
    aoi_id: str
    osm_type: str
    osm_id: str
    function: str
    name: str
    geometry: Polygon
    source_geometry: str
    tags: dict[str, str]


@dataclass(slots=True)
class EntranceDetail:
    entrance_id: str
    aoi_id: str
    node_id: str
    x: float
    y: float
    source: str
    inference_quality: float
    entrance_type: str
    access: str
    osm_id: str
    connector_node_id: str
    connector_length_m: float
    is_primary: bool = False


def parse_osm(path: Path) -> tuple[
    dict[str, tuple[float, float, dict[str, str]]],
    dict[str, OSMWay],
    list[OSMRelation],
    dict[str, float],
]:
    nodes: dict[str, tuple[float, float, dict[str, str]]] = {}
    ways: dict[str, OSMWay] = {}
    relations: list[OSMRelation] = []
    bounds: dict[str, float] = {}
    for event, element in iterparse(path, events=("start", "end")):
        if event == "start" and element.tag == "bounds":
            bounds = {key: float(value) for key, value in element.attrib.items() if key in {
                "minlat", "minlon", "maxlat", "maxlon",
            }}
            continue
        if event != "end":
            continue
        if element.tag == "node":
            tags = {item.attrib["k"]: item.attrib["v"] for item in element.findall("tag")}
            nodes[element.attrib["id"]] = (
                float(element.attrib["lon"]), float(element.attrib["lat"]), tags,
            )
            element.clear()
        elif element.tag == "way":
            tags = {item.attrib["k"]: item.attrib["v"] for item in element.findall("tag")}
            refs = [item.attrib["ref"] for item in element.findall("nd")]
            ways[element.attrib["id"]] = OSMWay(element.attrib["id"], refs, tags)
            element.clear()
        elif element.tag == "relation":
            tags = {item.attrib["k"]: item.attrib["v"] for item in element.findall("tag")}
            members = [
                (item.attrib.get("type", ""), item.attrib.get("ref", ""), item.attrib.get("role", ""))
                for item in element.findall("member")
            ]
            relations.append(OSMRelation(element.attrib["id"], members, tags))
            element.clear()
    if not bounds and nodes:
        longitudes = [item[0] for item in nodes.values()]
        latitudes = [item[1] for item in nodes.values()]
        bounds = {
            "minlon": min(longitudes), "maxlon": max(longitudes),
            "minlat": min(latitudes), "maxlat": max(latitudes),
        }
    return nodes, ways, relations, bounds


def choose_metric_crs(bounds: dict[str, float]) -> CRS:
    center_lon = (bounds["minlon"] + bounds["maxlon"]) / 2
    center_lat = (bounds["minlat"] + bounds["maxlat"]) / 2
    zone = int((center_lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if center_lat >= 0 else 32700) + zone)


def project_nodes(
    raw_nodes: dict[str, tuple[float, float, dict[str, str]]], transformer: Transformer,
) -> dict[str, tuple[float, float]]:
    projected: dict[str, tuple[float, float]] = {}
    for osm_id, (longitude, latitude, _) in raw_nodes.items():
        x, y = transformer.transform(longitude, latitude)
        projected[osm_id] = (float(x), float(y))
    return projected


def is_walkable(tags: dict[str, str]) -> bool:
    highway = tags.get("highway", "")
    if highway not in WALKABLE_HIGHWAYS or tags.get("area") == "yes":
        return False
    access = tags.get("access", "").lower()
    foot = tags.get("foot", "").lower()
    if foot in {"no", "private", "use_sidepath"}:
        return False
    if access in {"no", "private"} and foot not in {"yes", "designated", "permissive"}:
        return False
    return True


def road_characteristics(tags: dict[str, str]) -> tuple[float, float, float, float]:
    highway = tags.get("highway", "")
    if highway == "steps":
        speed, visibility, safety, comfort = 45.0, 0.68, 0.58, 0.42
    elif highway in {"footway", "pedestrian", "path", "corridor"}:
        speed, visibility, safety, comfort = 76.0, 0.82, 0.83, 0.82
    elif highway in {"living_street", "residential", "service", "unclassified"}:
        speed, visibility, safety, comfort = 72.0, 0.72, 0.72, 0.68
    else:
        speed, visibility, safety, comfort = 68.0, 0.76, 0.62, 0.58
    if tags.get("lit") == "yes":
        safety = min(0.95, safety + 0.08)
    if tags.get("surface") in {"unpaved", "gravel", "dirt", "ground", "sand"}:
        comfort = max(0.3, comfort - 0.18)
        speed *= 0.86
    return speed, visibility, safety, comfort


def build_walk_graph(
    ways: dict[str, OSMWay], projected: dict[str, tuple[float, float]],
) -> tuple[dict[str, Node], dict[str, Edge], dict[str, dict[str, str]], dict[str, Any]]:
    nodes: dict[str, Node] = {}
    edges: dict[str, Edge] = {}
    edge_tags: dict[str, dict[str, str]] = {}
    for way in ways.values():
        if not is_walkable(way.tags):
            continue
        valid_refs = [ref for ref in way.refs if ref in projected]
        for sequence, (left_ref, right_ref) in enumerate(zip(valid_refs, valid_refs[1:])):
            left_xy, right_xy = projected[left_ref], projected[right_ref]
            length = math.hypot(right_xy[0] - left_xy[0], right_xy[1] - left_xy[1])
            if length < 0.05:
                continue
            left_id, right_id = f"N_OSM_{left_ref}", f"N_OSM_{right_ref}"
            nodes.setdefault(left_id, Node(left_id, *left_xy))
            nodes.setdefault(right_id, Node(right_id, *right_xy))
            edge_id = f"E_W{way.osm_id}_{sequence:04d}"
            speed, visibility, safety, comfort = road_characteristics(way.tags)
            edges[edge_id] = Edge(
                edge_id=edge_id,
                from_node=left_id,
                to_node=right_id,
                length=length,
                base_travel_time=max(0.01, length / speed),
                status=EdgeStatus.OPEN,
                visibility=visibility,
                safety=safety,
                comfort=comfort,
                geometry=[left_xy, right_xy],
                name=preferred_name(way.tags, ""),
                road_class=way.tags.get("highway", ""),
            )
            edge_tags[edge_id] = {
                "highway": way.tags.get("highway", ""),
                "name": preferred_name(way.tags, ""),
                "source": "OSM_WAY",
                "osm_id": way.osm_id,
            }
    if not edges:
        raise ValueError("No walkable OSM ways were found")

    topology = nx.Graph()
    topology.add_nodes_from(nodes)
    topology.add_edges_from((edge.from_node, edge.to_node) for edge in edges.values())
    components = sorted(nx.connected_components(topology), key=len, reverse=True)
    largest = components[0]
    input_node_count = len(nodes)
    input_edge_count = len(edges)
    nodes = {node_id: node for node_id, node in nodes.items() if node_id in largest}
    edges = {
        edge_id: edge for edge_id, edge in edges.items()
        if edge.from_node in largest and edge.to_node in largest
    }
    edge_tags = {edge_id: edge_tags[edge_id] for edge_id in edges}
    qc = {
        "walk_components_before_pruning": len(components),
        "walk_nodes_before_pruning": input_node_count,
        "walk_edges_before_pruning": input_edge_count,
        "largest_component_node_ratio": len(largest) / max(1, input_node_count),
        "walk_nodes_after_pruning": len(nodes),
        "walk_edges_after_pruning": len(edges),
    }
    return nodes, edges, edge_tags, qc


def preferred_name(tags: dict[str, str], fallback: str) -> str:
    for key in (
        "name:zh-Hans", "name:zh", "name", "addr:housename:zh", "addr:housename",
        "name:en", "brand:zh", "brand",
    ):
        if tags.get(key):
            return tags[key].strip()
    return fallback


def classify_function(tags: dict[str, str]) -> str | None:
    amenity = tags.get("amenity", "").lower()
    building = tags.get("building", "").lower()
    leisure = tags.get("leisure", "").lower()
    landuse = tags.get("landuse", "").lower()
    if amenity in CAFE_AMENITIES:
        return "CAFE"
    if tags.get("shop") or amenity == "marketplace" or building == "retail":
        return "SHOP"
    if leisure in PARK_LEISURE or landuse in PARK_LANDUSE:
        return "PARK"
    if building in RESIDENTIAL_BUILDINGS:
        return "HOME"
    if tags.get("office") or building in WORK_BUILDINGS or amenity in WORK_AMENITIES:
        return "WORK"
    if landuse in {"commercial", "industrial", "institutional", "governmental"}:
        return "WORK"
    if tags.get("tourism") in {"hotel", "museum", "gallery"}:
        return "WORK"
    return None


def polygon_parts(geometry: Any) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return [part for part in geometry.geoms if not part.is_empty]
    if geometry.geom_type == "GeometryCollection":
        parts: list[Polygon] = []
        for part in geometry.geoms:
            parts.extend(polygon_parts(part))
        return parts
    return []


def way_polygon(way: OSMWay, projected: dict[str, tuple[float, float]]) -> Polygon | None:
    coordinates = [projected[ref] for ref in way.refs if ref in projected]
    if len(coordinates) < 4 or coordinates[0] != coordinates[-1]:
        return None
    polygon = Polygon(coordinates)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    parts = polygon_parts(polygon)
    return max(parts, key=lambda part: part.area) if parts else None


def relation_polygons(
    relation: OSMRelation,
    ways: dict[str, OSMWay],
    projected: dict[str, tuple[float, float]],
) -> list[Polygon]:
    if relation.tags.get("type") == "building":
        outline_refs = [ref for kind, ref, role in relation.members if kind == "way" and role == "outline"]
        parts = [way_polygon(ways[ref], projected) for ref in outline_refs if ref in ways]
        return [part for part in parts if part is not None]

    outer_lines: list[LineString] = []
    inner_lines: list[LineString] = []
    for kind, ref, role in relation.members:
        if kind != "way" or ref not in ways:
            continue
        coordinates = [projected[node_ref] for node_ref in ways[ref].refs if node_ref in projected]
        if len(coordinates) < 2:
            continue
        line = LineString(coordinates)
        if role == "inner":
            inner_lines.append(line)
        elif role in {"outer", "", "outline"}:
            outer_lines.append(line)
    if not outer_lines:
        return []
    outer = unary_union(list(polygonize(unary_union(outer_lines))))
    if outer.is_empty:
        fallback = [way_polygon(ways[ref], projected) for kind, ref, role in relation.members if kind == "way" and role != "inner" and ref in ways]
        outer = unary_union([part for part in fallback if part is not None])
    if inner_lines and not outer.is_empty:
        inner = unary_union(list(polygonize(unary_union(inner_lines))))
        if not inner.is_empty:
            outer = outer.difference(inner)
    return polygon_parts(outer)


def create_aoi_sources(
    raw_nodes: dict[str, tuple[float, float, dict[str, str]]],
    ways: dict[str, OSMWay],
    relations: list[OSMRelation],
    projected: dict[str, tuple[float, float]],
) -> tuple[list[AOISource], list[tuple[str, str, Polygon, dict[str, str]]], set[str]]:
    sources: list[AOISource] = []
    building_geometries: list[tuple[str, str, Polygon, dict[str, str]]] = []
    consumed_aoi_ways: set[str] = set()
    consumed_building_ways: set[str] = set()

    for relation in relations:
        parts = relation_polygons(relation, ways, projected)
        if not parts:
            continue
        function = classify_function(relation.tags)
        if function:
            for index, polygon in enumerate(parts, start=1):
                aoi_id = f"AOI_R{relation.osm_id}" + (f"_P{index}" if len(parts) > 1 else "")
                fallback = f"{function_name(function)} OSM R{relation.osm_id}"
                sources.append(AOISource(
                    aoi_id, "relation", relation.osm_id, function,
                    preferred_name(relation.tags, fallback), polygon, "OSM_POLYGON", relation.tags,
                ))
            consumed_aoi_ways.update(
                ref for kind, ref, role in relation.members if kind == "way" and role in {"outer", "outline"}
            )
        if relation.tags.get("building") and relation.tags.get("building") != "no":
            for index, polygon in enumerate(parts, start=1):
                object_id = f"OCC_R{relation.osm_id}" + (f"_P{index}" if len(parts) > 1 else "")
                building_geometries.append((object_id, relation.osm_id, polygon, relation.tags))
            consumed_building_ways.update(
                ref for kind, ref, role in relation.members if kind == "way" and role in {"outer", "outline"}
            )

    for way in ways.values():
        polygon = way_polygon(way, projected)
        if polygon is None:
            continue
        function = classify_function(way.tags)
        if function and way.osm_id not in consumed_aoi_ways:
            aoi_id = f"AOI_W{way.osm_id}"
            fallback = f"{function_name(function)} OSM W{way.osm_id}"
            sources.append(AOISource(
                aoi_id, "way", way.osm_id, function,
                preferred_name(way.tags, fallback), polygon, "OSM_POLYGON", way.tags,
            ))
        if way.tags.get("building") and way.tags.get("building") != "no" and way.osm_id not in consumed_building_ways:
            building_geometries.append((f"OCC_W{way.osm_id}", way.osm_id, polygon, way.tags))

    for osm_id, (_, _, tags) in raw_nodes.items():
        function = classify_function(tags)
        if function is None or osm_id not in projected:
            continue
        point = Point(projected[osm_id])
        polygon = point.buffer(4.0, quad_segs=4)
        aoi_id = f"AOI_N{osm_id}"
        fallback = f"{function_name(function)} OSM N{osm_id}"
        sources.append(AOISource(
            aoi_id, "node", osm_id, function,
            preferred_name(tags, fallback), polygon, "OSM_POINT_BUFFER_4M", tags,
        ))

    sources.sort(key=lambda item: item.aoi_id)
    return sources, building_geometries, consumed_aoi_ways


def function_name(function: str) -> str:
    return {"HOME": "住宅", "WORK": "工作地", "CAFE": "餐饮", "PARK": "公园", "SHOP": "商店"}.get(function, function)


def aoi_parameters(source: AOISource) -> tuple[float, float, float, int]:
    defaults = {
        "HOME": (0.52, 0.76, 0.68),
        "WORK": (0.63, 0.72, 0.63),
        "CAFE": (0.74, 0.74, 0.78),
        "PARK": (0.82, 0.82, 0.88),
        "SHOP": (0.70, 0.71, 0.69),
    }
    attractiveness, safety, comfort = defaults[source.function]
    if preferred_name(source.tags, ""):
        attractiveness = min(0.95, attractiveness + 0.04)
    area = max(source.geometry.area, 25.0)
    capacity = max(10, min(5000, int(area / 15.0)))
    return attractiveness, safety, comfort, capacity


def entrance_candidates(
    raw_nodes: dict[str, tuple[float, float, dict[str, str]]],
    projected: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for osm_id, (_, _, tags) in raw_nodes.items():
        entrance_value = tags.get("entrance", "")
        barrier_value = tags.get("barrier", "")
        if not entrance_value and barrier_value not in {"gate", "lift_gate", "sally_port"}:
            continue
        if osm_id not in projected:
            continue
        source = "OSM_ENTRANCE" if entrance_value else "OSM_GATE"
        candidates.append({
            "osm_id": osm_id,
            "point": Point(projected[osm_id]),
            "source": source,
            "inference_quality": 1.0 if entrance_value else 0.82,
            "entrance_type": entrance_value or barrier_value,
            "access": tags.get("access", ""),
            "tags": tags,
        })
    return candidates


def associate_explicit_entrances(
    candidates: list[dict[str, Any]], aoi_sources: list[AOISource], tolerance_m: float = 4.0,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_aoi: dict[str, list[dict[str, Any]]] = defaultdict(list)
    orphaned: list[dict[str, Any]] = []
    geometries = [source.geometry for source in aoi_sources]
    tree = STRtree(geometries)
    for candidate in candidates:
        point = candidate["point"]
        indexes = tree.query(point.buffer(tolerance_m))
        scored: list[tuple[float, float, int]] = []
        for raw_index in indexes:
            index = int(raw_index)
            geometry = geometries[index]
            boundary_distance = point.distance(geometry.boundary)
            if geometry.covers(point) or boundary_distance <= tolerance_m:
                scored.append((boundary_distance, geometry.area, index))
        if not scored:
            orphaned.append(candidate)
            continue
        _, _, selected = min(scored)
        by_aoi[aoi_sources[selected].aoi_id].append(candidate)
    return by_aoi, orphaned


def build_aois_and_entrances(
    aoi_sources: list[AOISource],
    explicit_by_aoi: dict[str, list[dict[str, Any]]],
    base_nodes: dict[str, Node],
    edges: dict[str, Edge],
    edge_tags: dict[str, dict[str, str]],
) -> tuple[dict[str, AOI], dict[str, Entrance], list[EntranceDetail], list[dict[str, Any]]]:
    base_node_ids = list(base_nodes)
    base_points = [Point(base_nodes[node_id].x, base_nodes[node_id].y) for node_id in base_node_ids]
    node_tree = STRtree(base_points)
    entrances: dict[str, Entrance] = {}
    details: list[EntranceDetail] = []
    aois: dict[str, AOI] = {}
    connector_rows: list[dict[str, Any]] = []
    coordinate_nodes = {
        (round(node.x, 2), round(node.y, 2)): node_id for node_id, node in base_nodes.items()
    }

    def nearest_base_node(point: Point) -> str:
        return base_node_ids[int(node_tree.nearest(point))]

    def attach_entrance(
        source: AOISource,
        point: Point,
        entrance_id: str,
        entrance_source: str,
        inference_quality: float,
        entrance_type: str,
        access: str,
        osm_id: str,
        explicit_osm_node: str | None,
    ) -> EntranceDetail:
        direct_node_id = f"N_OSM_{explicit_osm_node}" if explicit_osm_node else ""
        connector_node_id = nearest_base_node(point)
        if direct_node_id in base_nodes:
            entrance_node_id = direct_node_id
            connector_node_id = direct_node_id
            connector_length = 0.0
        else:
            coordinate_key = (round(point.x, 2), round(point.y, 2))
            entrance_node_id = coordinate_nodes.get(coordinate_key, "")
            if not entrance_node_id:
                entrance_node_id = f"N_ENT_{len([key for key in base_nodes if key.startswith('N_ENT_')]) + 1:06d}"
                base_nodes[entrance_node_id] = Node(entrance_node_id, float(point.x), float(point.y))
                coordinate_nodes[coordinate_key] = entrance_node_id
            target = base_nodes[connector_node_id]
            connector_length = math.hypot(point.x - target.x, point.y - target.y)
            if entrance_node_id != connector_node_id and connector_length > 0.05:
                edge_id = f"E_ACCESS_{entrance_id}"
                suffix = 2
                while edge_id in edges:
                    edge_id = f"E_ACCESS_{entrance_id}_{suffix}"
                    suffix += 1
                edges[edge_id] = Edge(
                    edge_id, entrance_node_id, connector_node_id,
                    connector_length, max(0.01, connector_length / 70.0),
                    EdgeStatus.OPEN, 0.72, 0.74, 0.68, False,
                    [(float(point.x), float(point.y)), (target.x, target.y)],
                    "AOI access connector", "access_connector",
                )
                edge_tags[edge_id] = {
                    "highway": "access_connector", "name": "AOI access connector",
                    "source": entrance_source, "osm_id": osm_id,
                }
                connector_rows.append({
                    "entrance_id": entrance_id, "node_id": connector_node_id,
                    "length_m": connector_length, "source": entrance_source,
                    "geometry": [(float(point.x), float(point.y)), (target.x, target.y)],
                })
        status = EdgeStatus.CLOSED if access.lower() == "no" else EdgeStatus.OPEN
        conspicuity = {
            "main": 0.92, "yes": 0.76, "gate": 0.72,
            "INFERRED_BOUNDARY": 0.52, "POI_POINT": 0.62,
        }.get(entrance_type, 0.68)
        entrances[entrance_id] = Entrance(
            entrance_id, source.aoi_id, entrance_node_id,
            float(point.x), float(point.y), status, conspicuity,
        )
        return EntranceDetail(
            entrance_id, source.aoi_id, entrance_node_id,
            float(point.x), float(point.y), entrance_source, inference_quality,
            entrance_type, access, osm_id, connector_node_id, connector_length,
        )

    for source in aoi_sources:
        source_details: list[EntranceDetail] = []
        explicit = sorted(
            explicit_by_aoi.get(source.aoi_id, []),
            key=lambda item: (item["entrance_type"] != "main", item["source"] != "OSM_ENTRANCE"),
        )
        for candidate in explicit:
            entrance_id = f"ENT_OSM_N{candidate['osm_id']}"
            if entrance_id in entrances:
                continue
            source_details.append(attach_entrance(
                source=source,
                point=candidate["point"],
                entrance_id=entrance_id,
                entrance_source=candidate["source"],
                inference_quality=float(candidate["inference_quality"]),
                entrance_type=str(candidate["entrance_type"]),
                access=str(candidate["access"]),
                osm_id=str(candidate["osm_id"]),
                explicit_osm_node=str(candidate["osm_id"]),
            ))

        if not source_details:
            nearest_id = nearest_base_node(source.geometry)
            nearest_road_point = Point(base_nodes[nearest_id].x, base_nodes[nearest_id].y)
            if source.source_geometry.startswith("OSM_POINT"):
                entrance_point = source.geometry.centroid
                entrance_source = "POI_POINT"
                entrance_type = "POI_POINT"
            else:
                entrance_point, _ = nearest_points(source.geometry.boundary, nearest_road_point)
                entrance_source = "INFERRED_NEAREST_WALK_NODE"
                entrance_type = "INFERRED_BOUNDARY"
            distance = entrance_point.distance(nearest_road_point)
            inference_quality = 0.72 if distance <= 3 else 0.62 if distance <= 10 else 0.50 if distance <= 30 else 0.35
            source_details.append(attach_entrance(
                source=source,
                point=entrance_point,
                entrance_id=f"ENT_INF_{source.aoi_id}",
                entrance_source=entrance_source,
                inference_quality=inference_quality,
                entrance_type=entrance_type,
                access=str(source.tags.get("access", "")),
                osm_id=source.osm_id,
                explicit_osm_node=None,
            ))

        source_details[0].is_primary = True
        details.extend(source_details)
        primary = source_details[0]
        attractiveness, safety, comfort, capacity = aoi_parameters(source)
        aois[source.aoi_id] = AOI(
            source.aoi_id, source.name, primary.node_id, source.function,
            attractiveness, safety, comfort, capacity, False,
            [(float(x), float(y)) for x, y in source.geometry.exterior.coords],
            feature_kind="POI" if source.source_geometry.startswith("OSM_POINT") else "AOI",
        )
    return aois, entrances, details, connector_rows


def build_occluders(
    geometries: list[tuple[str, str, Polygon, dict[str, str]]],
) -> tuple[dict[str, Occluder], dict[str, dict[str, str]]]:
    occluders: dict[str, Occluder] = {}
    metadata: dict[str, dict[str, str]] = {}
    for object_id, osm_id, polygon, tags in geometries:
        if polygon.is_empty or polygon.area < 1:
            continue
        opacity = 0.72 if tags.get("building") == "roof" else 1.0
        occluders[object_id] = Occluder(
            object_id,
            [(float(x), float(y)) for x, y in polygon.exterior.coords],
            opacity,
            "BUILDING",
        )
        metadata[object_id] = {"source": "OSM_BUILDING", "osm_id": osm_id}
    return occluders, metadata


def _legacy_suggest_agent_bindings(graph: UrbanGraph) -> list[dict[str, str]]:
    homes = [item for item in graph.aois.values() if item.function == "HOME"]
    works = [item for item in graph.aois.values() if item.function == "WORK"]
    if len(homes) < 3 or len(works) < 3:
        raise ValueError("The processed map needs at least three HOME and three WORK AOIs")
    all_x = [node.x for node in graph.nodes.values()]
    all_y = [node.y for node in graph.nodes.values()]
    xmin, xmax, ymin, ymax = min(all_x), max(all_x), min(all_y), max(all_y)
    targets = [
        ((xmin + 0.18 * (xmax - xmin), ymin + 0.42 * (ymax - ymin)),
         (xmin + 0.78 * (xmax - xmin), ymin + 0.58 * (ymax - ymin))),
        ((xmin + 0.82 * (xmax - xmin), ymin + 0.32 * (ymax - ymin)),
         (xmin + 0.22 * (xmax - xmin), ymin + 0.68 * (ymax - ymin))),
        ((xmin + 0.48 * (xmax - xmin), ymin + 0.26 * (ymax - ymin)),
         (xmin + 0.52 * (xmax - xmin), ymin + 0.70 * (ymax - ymin))),
    ]
    used_homes: set[str] = set()
    used_works: set[str] = set()

    def choose(candidates: list[AOI], target: tuple[float, float], used: set[str]) -> AOI:
        available = [item for item in candidates if item.aoi_id not in used]
        named = [item for item in available if not item.name.startswith(("住宅 OSM", "工作地 OSM"))]
        pool = named or available
        selected = min(
            pool,
            key=lambda item: math.hypot(
                graph.nodes[item.node_id].x - target[0], graph.nodes[item.node_id].y - target[1],
            ),
        )
        used.add(selected.aoi_id)
        return selected

    agent_ids = ["A001_active_senior", "A002_routine_senior", "A003_limited_senior"]
    suggestions: list[dict[str, str]] = []
    for agent_id, (home_target, anchor_target) in zip(agent_ids, targets):
        home = choose(homes, home_target, used_homes)
        anchor = choose(works, anchor_target, used_works)
        route = graph.shortest_path(home.node_id, anchor.node_id)
        if route is None:
            raise ValueError(f"Suggested HOME/routine-anchor pair is disconnected for {agent_id}")
        distance, travel_time = graph.path_metrics(route)
        suggestions.append({
            "agent_id": agent_id,
            "home_aoi_id": home.aoi_id,
            "home_name": home.name,
            "routine_anchor_aoi_id": anchor.aoi_id,
            "routine_anchor_name": anchor.name,
            "route_distance_m": f"{distance:.1f}",
            "route_time_min": f"{travel_time:.1f}",
        })
    return suggestions


def _legacy_suggest_intervention(graph: UrbanGraph) -> list[dict[str, Any]]:
    shops = [item for item in graph.aois.values() if item.function == "SHOP"]
    if not shops:
        return []
    named_shops = [item for item in shops if not item.name.startswith("商店 OSM")]
    candidates = named_shops or shops
    center_x = statistics.mean(node.x for node in graph.nodes.values())
    center_y = statistics.mean(node.y for node in graph.nodes.values())
    selected = min(
        candidates,
        key=lambda item: math.hypot(
            graph.nodes[item.node_id].x - center_x, graph.nodes[item.node_id].y - center_y,
        ),
    )
    return [{
        "intervention_id": "INT_OSM_DAY2_SHOP_TO_PARK",
        "effective_day": 2,
        "operation_type": "AOI_FUNCTION_CHANGE",
        "target_type": "AOI",
        "target_id": selected.aoi_id,
        "before_value": "SHOP",
        "after_value": "PARK",
        "visibility_level": 0.9,
        "announcement_level": 0.0,
        "signage_support": 0.0,
    }]


def _distance_graph(graph: UrbanGraph) -> nx.Graph:
    topology = nx.Graph()
    for edge in graph.edges.values():
        topology.add_edge(edge.from_node, edge.to_node, weight=float(edge.length))
    return topology


def suggest_agent_bindings(graph: UrbanGraph) -> list[dict[str, Any]]:
    """Create map-specific older-adult homes, anchors and a shared meeting place."""
    homes = [item for item in graph.aois.values() if item.function == "HOME"]
    meetings = [item for item in graph.aois.values() if item.function in {"PARK", "CAFE"}]
    anchors = [
        item for item in graph.aois.values()
        if item.function in {"WORK", "SHOP", "PARK", "CAFE"}
    ]
    if len(homes) < 3 or not meetings or len(anchors) < 3:
        raise ValueError(
            "The processed map needs at least three HOME AOIs, one PARK/CAFE, "
            "and three non-residential routine destinations"
        )

    specs = [
        {
            "agent_id": "A001_active_senior", "radius_m": 1200.0,
            "home_target_m": 720.0, "anchor_target_m": 650.0,
            "anchor_preferences": ["PARK", "CAFE", "SHOP", "WORK"],
        },
        {
            "agent_id": "A002_routine_senior", "radius_m": 850.0,
            "home_target_m": 500.0, "anchor_target_m": 430.0,
            "anchor_preferences": ["SHOP", "PARK", "CAFE", "WORK"],
        },
        {
            "agent_id": "A003_limited_senior", "radius_m": 600.0,
            "home_target_m": 280.0, "anchor_target_m": 300.0,
            "anchor_preferences": ["SHOP", "PARK", "CAFE", "WORK"],
        },
    ]
    topology = _distance_graph(graph)

    def is_named(item: AOI) -> bool:
        return bool(item.name.strip()) and "OSM" not in item.name.upper()

    best: tuple[float, float, AOI, dict[str, AOI], dict[str, float]] | None = None
    for radius_multiplier in (1.0, 1.25, 1.5, 2.0):
        for meeting in meetings:
            distances = nx.single_source_dijkstra_path_length(
                topology, meeting.node_id, weight="weight",
            )
            assigned: dict[str, AOI] = {}
            assigned_distances: dict[str, float] = {}
            used: set[str] = set()
            feasible = True
            # Bind the most mobility-constrained older resident first.
            for spec in sorted(specs, key=lambda item: item["radius_m"]):
                available = [
                    home for home in homes
                    if home.aoi_id not in used
                    and distances.get(home.node_id, math.inf)
                    <= spec["radius_m"] * radius_multiplier
                ]
                if not available:
                    feasible = False
                    break
                selected = min(
                    available,
                    key=lambda home: (
                        abs(distances[home.node_id] - spec["home_target_m"]),
                        0 if is_named(home) else 35,
                        home.aoi_id,
                    ),
                )
                assigned[spec["agent_id"]] = selected
                assigned_distances[spec["agent_id"]] = float(distances[selected.node_id])
                used.add(selected.aoi_id)
            if not feasible:
                continue
            score = sum(
                abs(assigned_distances[spec["agent_id"]] - spec["home_target_m"])
                for spec in specs
            )
            score += 0.0 if meeting.function == "PARK" else 80.0
            score += 0.0 if is_named(meeting) else 40.0
            candidate = (score, radius_multiplier, meeting, assigned, assigned_distances)
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is not None:
            break
    if best is None:
        raise ValueError(
            "No connected older-adult residential cluster could be generated around a PARK/CAFE"
        )

    _, radius_multiplier, meeting, assigned_homes, meeting_distances = best
    used_anchors: set[str] = {meeting.aoi_id}
    suggestions: list[dict[str, Any]] = []
    for spec in specs:
        home = assigned_homes[spec["agent_id"]]
        distances = nx.single_source_dijkstra_path_length(topology, home.node_id, weight="weight")
        available = [
            item for item in anchors
            if item.aoi_id not in used_anchors
            and distances.get(item.node_id, math.inf)
            <= spec["radius_m"] * radius_multiplier
        ]
        if not available:
            raise ValueError(f"No routine anchor is reachable for {spec['agent_id']}")
        preference = {name: index for index, name in enumerate(spec["anchor_preferences"])}
        anchor = min(
            available,
            key=lambda item: (
                preference.get(item.function, 99) * 220
                + abs(distances[item.node_id] - spec["anchor_target_m"]),
                0 if is_named(item) else 30,
                item.aoi_id,
            ),
        )
        used_anchors.add(anchor.aoi_id)
        anchor_distance = float(distances[anchor.node_id])
        route = graph.shortest_path(home.node_id, anchor.node_id, cost=lambda edge: edge.length)
        if route is None:
            raise ValueError(
                f"Suggested HOME/routine-anchor pair is disconnected for {spec['agent_id']}"
            )
        _, travel_time = graph.path_metrics(route)
        suggestions.append({
            "agent_id": spec["agent_id"],
            "home_aoi_id": home.aoi_id,
            "home_name": home.name,
            "routine_anchor_aoi_id": anchor.aoi_id,
            "routine_anchor_name": anchor.name,
            "routine_anchor_function": anchor.function,
            "regular_meeting_aoi_ids": [meeting.aoi_id],
            "meeting_name": meeting.name,
            "meeting_function": meeting.function,
            "home_to_anchor_distance_m": round(anchor_distance, 1),
            "home_to_meeting_distance_m": round(meeting_distances[spec["agent_id"]], 1),
            "route_time_min": round(travel_time, 1),
            "activity_radius_m": spec["radius_m"],
            "binding_radius_multiplier": radius_multiplier,
        })
    return suggestions


def suggest_intervention(
    graph: UrbanGraph, agent_bindings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate two valid interventions inside the selected agents' activity system."""
    shops = [item for item in graph.aois.values() if item.function == "SHOP"]
    if not shops or not agent_bindings:
        return []
    topology = _distance_graph(graph)
    meeting_id = str(agent_bindings[0]["regular_meeting_aoi_ids"][0])
    meeting = graph.aois[meeting_id]
    meeting_distances = nx.single_source_dijkstra_path_length(
        topology, meeting.node_id, weight="weight",
    )
    named_shops = [item for item in shops if item.name.strip() and "OSM" not in item.name.upper()]
    candidates = named_shops or shops
    selected_shop = min(
        candidates,
        key=lambda item: (meeting_distances.get(item.node_id, math.inf), item.aoi_id),
    )

    route_pairs: list[tuple[str, str, list[str], float]] = []
    for binding in agent_bindings:
        for destination_id in (str(binding["routine_anchor_aoi_id"]), meeting_id):
            start = graph.aois[str(binding["home_aoi_id"])].node_id
            goal = graph.aois[destination_id].node_id
            route = graph.shortest_path(start, goal, cost=lambda edge: edge.length)
            if route:
                route_pairs.append((start, goal, route, graph.path_metrics(route)[0]))

    usage: Counter[str] = Counter(
        edge_id for _, _, route, _ in route_pairs for edge_id in route
        if graph.edges[edge_id].road_class != "access_connector"
    )
    road_scores: list[tuple[float, str]] = []
    for edge_id, count in usage.items():
        edge = graph.edges[edge_id]
        if edge.length < 3.0:
            continue
        detour_gain = 0.0
        affected = 0
        for start, goal, route, baseline in route_pairs:
            if edge_id not in route:
                continue
            alternate = graph.shortest_path(
                start,
                goal,
                allowed=lambda candidate, blocked=edge_id: (
                    candidate.status == EdgeStatus.OPEN and candidate.edge_id != blocked
                ),
                cost=lambda candidate: candidate.length,
            )
            if alternate is None:
                continue
            detour_gain += max(0.0, graph.path_metrics(alternate)[0] - baseline)
            affected += 1
        if affected:
            road_scores.append((count * 1000.0 + detour_gain, edge_id))
    selected_road = max(road_scores)[1] if road_scores else ""

    events: list[dict[str, Any]] = [{
        "intervention_id": "INT_DAY03_LOCAL_SHOP_TO_PARK",
        "effective_day": 3,
        "operation_type": "AOI_FUNCTION_CHANGE",
        "target_type": "AOI",
        "target_id": selected_shop.aoi_id,
        "before_value": "SHOP",
        "after_value": "PARK",
        "visibility_level": 0.9,
        "announcement_level": 0.0,
        "visual_salience": 0.9,
        "official_announcement": 0.0,
        "signage_support": 0.0,
    }]
    if selected_road:
        events.append({
            "intervention_id": "INT_DAY06_ACTIVITY_ROUTE_CLOSE",
            "effective_day": 6,
            "operation_type": "ROAD_CLOSE",
            "target_type": "ROAD_EDGE",
            "target_id": selected_road,
            "before_value": "OPEN",
            "after_value": "CLOSED",
            "visibility_level": 0.95,
            "announcement_level": 0.0,
            "visual_salience": 0.95,
            "official_announcement": 0.0,
            "signage_support": 0.0,
        })
    return events


def write_projection_sidecars(base_path: Path, crs: CRS) -> None:
    base_path.with_suffix(".prj").write_text(crs.to_wkt(version="WKT1_ESRI"), encoding="utf-8")
    base_path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")


def write_shapefiles(
    output_dir: Path,
    graph: UrbanGraph,
    edge_tags: dict[str, dict[str, str]],
    aoi_sources: list[AOISource],
    entrance_details: list[EntranceDetail],
    occluder_metadata: dict[str, dict[str, str]],
    connector_rows: list[dict[str, Any]],
    crs: CRS,
) -> None:
    roads_path = output_dir / "roads.shp"
    with shapefile.Writer(str(roads_path), shapeType=shapefile.POLYLINE, encoding="utf-8") as writer:
        writer.field("ROAD_ID", "C", size=80)
        writer.field("FROM_ID", "C", size=80)
        writer.field("TO_ID", "C", size=80)
        writer.field("STATUS", "C", size=12)
        writer.field("TRAVEL_M", "N", size=18, decimal=8)
        writer.field("VISIBILITY", "N", size=10, decimal=6)
        writer.field("SAFETY", "N", size=10, decimal=6)
        writer.field("COMFORT", "N", size=10, decimal=6)
        writer.field("HIGHWAY", "C", size=32)
        writer.field("SOURCE", "C", size=32)
        writer.field("OSM_ID", "C", size=32)
        writer.field("NAME", "C", size=160)
        writer.field("ROAD_CLASS", "C", size=32)
        for edge in graph.edges.values():
            tags = edge_tags.get(edge.edge_id, {})
            writer.line([[[float(x), float(y)] for x, y in edge.geometry]])
            writer.record(
                edge.edge_id, edge.from_node, edge.to_node, str(edge.status),
                edge.base_travel_time, edge.visibility, edge.safety, edge.comfort,
                tags.get("highway", ""), tags.get("source", ""), tags.get("osm_id", ""),
                edge.name, edge.road_class,
            )
    write_projection_sidecars(roads_path, crs)

    nodes_path = output_dir / "nodes.shp"
    with shapefile.Writer(str(nodes_path), shapeType=shapefile.POINT, encoding="utf-8") as writer:
        writer.field("NODE_ID", "C", size=80)
        writer.field("SOURCE", "C", size=24)
        writer.field("OSM_ID", "C", size=32)
        for node in graph.nodes.values():
            source = "OSM_NODE" if node.node_id.startswith("N_OSM_") else "AOI_ENTRANCE"
            osm_id = node.node_id.removeprefix("N_OSM_") if source == "OSM_NODE" else ""
            writer.point(node.x, node.y)
            writer.record(node.node_id, source, osm_id)
    write_projection_sidecars(nodes_path, crs)

    source_by_id = {item.aoi_id: item for item in aoi_sources}
    aois_path = output_dir / "aois.shp"
    with shapefile.Writer(str(aois_path), shapeType=shapefile.POLYGON, encoding="utf-8") as writer:
        writer.field("AOI_ID", "C", size=80)
        writer.field("NAME", "C", size=160)
        writer.field("FUNCTION", "C", size=16)
        writer.field("ACCESS_NOD", "C", size=80)
        writer.field("ATTRACT", "N", size=10, decimal=6)
        writer.field("SAFETY", "N", size=10, decimal=6)
        writer.field("COMFORT", "N", size=10, decimal=6)
        writer.field("CAPACITY", "N", size=10, decimal=0)
        writer.field("SOURCE", "C", size=32)
        writer.field("FEATURE", "C", size=8)
        writer.field("OSM_ID", "C", size=32)
        for aoi in graph.aois.values():
            source = source_by_id[aoi.aoi_id]
            writer.poly([[[float(x), float(y)] for x, y in aoi.geometry]])
            writer.record(
                aoi.aoi_id, aoi.name, aoi.function, aoi.node_id,
                aoi.attractiveness, aoi.safety, aoi.comfort, aoi.capacity,
                source.source_geometry, aoi.feature_kind,
                f"{source.osm_type[0].upper()}{source.osm_id}",
            )
    write_projection_sidecars(aois_path, crs)

    detail_by_id = {item.entrance_id: item for item in entrance_details}
    entrances_path = output_dir / "entrances.shp"
    with shapefile.Writer(str(entrances_path), shapeType=shapefile.POINT, encoding="utf-8") as writer:
        writer.field("ENTRY_ID", "C", size=80)
        writer.field("AOI_ID", "C", size=80)
        writer.field("NODE_ID", "C", size=80)
        writer.field("STATUS", "C", size=12)
        writer.field("VISIBLE", "N", size=10, decimal=6)
        writer.field("SOURCE", "C", size=36)
        writer.field("INFER_Q", "N", size=10, decimal=6)
        writer.field("ENT_TYPE", "C", size=32)
        writer.field("ACCESS", "C", size=20)
        writer.field("OSM_ID", "C", size=32)
        writer.field("PRIMARY", "L")
        for entrance in graph.entrances.values():
            detail = detail_by_id[entrance.entrance_id]
            writer.point(entrance.x, entrance.y)
            writer.record(
                entrance.entrance_id, entrance.aoi_id, entrance.node_id, str(entrance.status),
                entrance.conspicuity, detail.source, detail.inference_quality,
                detail.entrance_type, detail.access, detail.osm_id, detail.is_primary,
            )
    write_projection_sidecars(entrances_path, crs)

    buildings_path = output_dir / "buildings.shp"
    with shapefile.Writer(str(buildings_path), shapeType=shapefile.POLYGON, encoding="utf-8") as writer:
        writer.field("OBJECT_ID", "C", size=80)
        writer.field("OPACITY", "N", size=10, decimal=6)
        writer.field("OBJ_TYPE", "C", size=20)
        writer.field("SOURCE", "C", size=24)
        writer.field("OSM_ID", "C", size=32)
        for occluder in graph.occluders.values():
            metadata = occluder_metadata.get(occluder.object_id, {})
            writer.poly([[[float(x), float(y)] for x, y in occluder.geometry]])
            writer.record(
                occluder.object_id, occluder.opacity, occluder.object_type,
                metadata.get("source", "OSM_BUILDING"), metadata.get("osm_id", ""),
            )
    write_projection_sidecars(buildings_path, crs)

    connectors_path = output_dir / "entrance_connectors.shp"
    with shapefile.Writer(str(connectors_path), shapeType=shapefile.POLYLINE, encoding="utf-8") as writer:
        writer.field("ENTRY_ID", "C", size=80)
        writer.field("NODE_ID", "C", size=80)
        writer.field("LENGTH_M", "N", size=16, decimal=4)
        writer.field("SOURCE", "C", size=36)
        for row in connector_rows:
            writer.line([[[float(x), float(y)] for x, y in row["geometry"]]])
            writer.record(row["entrance_id"], row["node_id"], row["length_m"], row["source"])
    write_projection_sidecars(connectors_path, crs)


def write_catalogs(
    output_dir: Path,
    graph: UrbanGraph,
    aoi_sources: list[AOISource],
    entrance_details: list[EntranceDetail],
) -> None:
    sources = {item.aoi_id: item for item in aoi_sources}
    with (output_dir / "aoi_catalog.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "aoi_id", "name", "function", "access_node", "osm_type", "osm_id",
            "source_geometry", "attractiveness", "safety", "comfort", "capacity",
        ])
        writer.writeheader()
        for aoi in graph.aois.values():
            source = sources[aoi.aoi_id]
            writer.writerow({
                "aoi_id": aoi.aoi_id, "name": aoi.name, "function": aoi.function,
                "access_node": aoi.node_id, "osm_type": source.osm_type,
                "osm_id": source.osm_id, "source_geometry": source.source_geometry,
                "attractiveness": aoi.attractiveness, "safety": aoi.safety,
                "comfort": aoi.comfort, "capacity": aoi.capacity,
            })
    with (output_dir / "entrance_catalog.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list(asdict(entrance_details[0]).keys()) if entrance_details else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(asdict(item) for item in entrance_details)


def make_qc(
    input_path: Path,
    raw_nodes: dict[str, tuple[float, float, dict[str, str]]],
    ways: dict[str, OSMWay],
    relations: list[OSMRelation],
    graph: UrbanGraph,
    walk_qc: dict[str, Any],
    entrance_details: list[EntranceDetail],
    explicit_candidates: list[dict[str, Any]],
    orphaned: list[dict[str, Any]],
    bounds: dict[str, float],
    crs: CRS,
) -> dict[str, Any]:
    connector_lengths = [item.connector_length_m for item in entrance_details]
    entrance_sources = Counter(item.source for item in entrance_details)
    functions = Counter(item.function for item in graph.aois.values())
    topology = nx.Graph()
    topology.add_nodes_from(graph.nodes)
    topology.add_edges_from((edge.from_node, edge.to_node) for edge in graph.edges.values())
    components = list(nx.connected_components(topology))
    return {
        "source": {
            "path": str(input_path.resolve()),
            "sha256": file_sha256(input_path),
            "osm_license": "ODbL 1.0; © OpenStreetMap contributors",
            "bounds_wgs84": bounds,
            "raw_node_count": len(raw_nodes),
            "raw_way_count": len(ways),
            "raw_relation_count": len(relations),
        },
        "projection": {"authority": crs.to_authority(), "name": crs.name, "metric": True},
        "walk_network": {
            **walk_qc,
            "final_node_count": len(graph.nodes),
            "final_edge_count": len(graph.edges),
            "final_component_count": len(components),
            "final_largest_component_ratio": max((len(item) for item in components), default=0) / max(1, len(graph.nodes)),
        },
        "facilities": {
            "aoi_count": len(graph.aois),
            "by_function": dict(sorted(functions.items())),
            "building_occluder_count": len(graph.occluders),
        },
        "entrances": {
            "total": len(graph.entrances),
            "by_source": dict(sorted(entrance_sources.items())),
            "raw_osm_entrance_or_gate_count": len(explicit_candidates),
            "used_osm_entrance_or_gate_count": len(explicit_candidates) - len(orphaned),
            "orphan_osm_entrance_or_gate_count": len(orphaned),
            "aoi_without_entrance_count": len(graph.aois) - len({item.aoi_id for item in entrance_details}),
            "connector_length_mean_m": statistics.mean(connector_lengths) if connector_lengths else 0.0,
            "connector_length_median_m": statistics.median(connector_lengths) if connector_lengths else 0.0,
            "connector_length_max_m": max(connector_lengths, default=0.0),
            "connector_over_50m_count": sum(length > 50 for length in connector_lengths),
            "connector_over_100m_count": sum(length > 100 for length in connector_lengths),
            "orphan_osm_ids": [item["osm_id"] for item in orphaned],
        },
    }


def write_qc_html(
    path: Path,
    graph: UrbanGraph,
    entrance_details: list[EntranceDetail],
    qc: dict[str, Any],
) -> None:
    all_x = [node.x for node in graph.nodes.values()]
    all_y = [node.y for node in graph.nodes.values()]
    xmin, xmax, ymin, ymax = min(all_x), max(all_x), min(all_y), max(all_y)
    width, height, padding = 1400.0, 820.0, 25.0
    scale = min((width - 2 * padding) / max(1.0, xmax - xmin), (height - 2 * padding) / max(1.0, ymax - ymin))

    def xy(x: float, y: float) -> tuple[float, float]:
        return padding + (x - xmin) * scale, height - padding - (y - ymin) * scale

    road_lines = []
    for edge in graph.edges.values():
        if edge.edge_id.startswith("E_ACCESS_"):
            continue
        points = " ".join(f"{xy(x, y)[0]:.2f},{xy(x, y)[1]:.2f}" for x, y in edge.geometry)
        road_lines.append(f'<polyline points="{points}"/>')
    aoi_colors = {"HOME": "#d8b15b", "WORK": "#7694a7", "CAFE": "#df886d", "PARK": "#78ad82", "SHOP": "#b28d7d"}
    aoi_shapes = []
    for aoi in graph.aois.values():
        points = " ".join(f"{xy(x, y)[0]:.2f},{xy(x, y)[1]:.2f}" for x, y in aoi.geometry)
        title = html.escape(f"{aoi.name} | {aoi.function} | {aoi.aoi_id}")
        aoi_shapes.append(
            f'<polygon points="{points}" fill="{aoi_colors.get(aoi.function, "#aaa")}"><title>{title}</title></polygon>'
        )
    detail_by_id = {item.entrance_id: item for item in entrance_details}
    entrance_shapes = []
    for entrance in graph.entrances.values():
        detail = detail_by_id[entrance.entrance_id]
        cx, cy = xy(entrance.x, entrance.y)
        color = "#0b7a61" if detail.source in {"OSM_ENTRANCE", "OSM_GATE"} else "#f1a62a"
        title = html.escape(
            f"{entrance.entrance_id} | {detail.source} | inference quality={detail.inference_quality:.2f} | connector={detail.connector_length_m:.1f}m"
        )
        entrance_shapes.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="2.6" fill="{color}"><title>{title}</title></circle>')

    function_rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{value}</td></tr>"
        for key, value in qc["facilities"]["by_function"].items()
    )
    source_rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{value}</td></tr>"
        for key, value in qc["entrances"]["by_source"].items()
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>OSM 地图处理质量检查</title>
<style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;background:#f3f1e8;color:#26332f;margin:0;padding:24px}}
.wrap{{max-width:1480px;margin:auto}} h1{{margin:0 0 8px}} .note{{color:#60706a;margin-bottom:18px}}
.cards{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px}}
.card,.panel{{background:#fffdf7;border:1px solid #d8d6ca;border-radius:14px;padding:15px}}
.n{{font-size:28px;font-weight:700;color:#116c5b}} table{{border-collapse:collapse;width:100%}}td{{padding:6px;border-bottom:1px solid #eee}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}}
svg{{width:100%;height:auto;background:#fbfaf5;border:1px solid #ddd;border-radius:12px}}
.roads polyline{{fill:none;stroke:#627875;stroke-width:.55;opacity:.55}}.aois polygon{{stroke:#344b46;stroke-width:.45;opacity:.72}}
.legend span{{display:inline-block;margin-right:18px}}.dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:5px}}
@media(max-width:900px){{.cards,.grid{{grid-template-columns:1fr 1fr}}}}
</style></head><body><div class="wrap">
<h1>OSM → Cognitive Map 地图质量检查</h1>
<div class="note">原始 map.osm 未修改。鼠标停在 AOI 或入口上可以查看名称、来源和连接距离。</div>
<div class="cards">
<div class="card"><div>最终路网节点</div><div class="n">{len(graph.nodes):,}</div></div>
<div class="card"><div>最终道路边</div><div class="n">{len(graph.edges):,}</div></div>
<div class="card"><div>AOI</div><div class="n">{len(graph.aois):,}</div></div>
<div class="card"><div>入口</div><div class="n">{len(graph.entrances):,}</div></div>
<div class="card"><div>建筑遮挡物</div><div class="n">{len(graph.occluders):,}</div></div>
</div>
<div class="panel"><div class="legend">
<span><i class="dot" style="background:#0b7a61"></i>OSM明确入口</span>
<span><i class="dot" style="background:#f1a62a"></i>算法推断入口</span>
<span>平均连接距离 {qc['entrances']['connector_length_mean_m']:.1f} m</span>
<span>超过50 m：{qc['entrances']['connector_over_50m_count']}</span>
<span>未关联OSM入口：{qc['entrances']['orphan_osm_entrance_or_gate_count']}</span>
</div></div>
<div class="panel" style="margin-top:14px"><svg viewBox="0 0 {width:.0f} {height:.0f}" aria-label="processed map">
<g class="roads">{''.join(road_lines)}</g><g class="aois">{''.join(aoi_shapes)}</g><g class="entrances">{''.join(entrance_shapes)}</g>
</svg></div>
<div class="grid"><div class="panel"><h2>AOI分类</h2><table>{function_rows}</table></div>
<div class="panel"><h2>入口来源</h2><table>{source_rows}</table></div></div>
<div class="panel" style="margin-top:14px"><h2>重要说明</h2>
<p>橙色入口是算法根据AOI边界与最近步行路网节点推断的候选入口，并非OSM明确事实。跨城市实验必须按入口来源分层检验。</p>
<p>坐标系：{html.escape(qc['projection']['name'])}；地图来源：© OpenStreetMap contributors，ODbL 1.0。</p></div>
</div></body></html>"""
    path.write_text(page, encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_run_example(
    output_dir: Path, agent_bindings: list[dict[str, str]], current_settings: dict[str, Any],
) -> dict[str, Any]:
    relative = output_dir.relative_to(PROJECT_ROOT).as_posix()
    settings = dict(current_settings)
    settings.update({
        "roads_shp": f"{relative}/roads.shp",
        "aois_shp": f"{relative}/aois.shp",
        "nodes_shp": f"{relative}/nodes.shp",
        "entrances_shp": f"{relative}/entrances.shp",
        "occluders_shp": f"{relative}/buildings.shp",
        "signage_shp": None,
        "field_mapping": {
            "edge_id": "ROAD_ID", "from_node": "FROM_ID", "to_node": "TO_ID",
            "status": "STATUS", "travel_time": "TRAVEL_M", "visibility": "VISIBILITY",
            "safety": "SAFETY", "comfort": "COMFORT", "node_id": "NODE_ID",
            "road_name": "NAME", "road_class": "ROAD_CLASS",
            "aoi_id": "AOI_ID", "name": "NAME", "function": "FUNCTION",
            "access_node": "ACCESS_NOD", "attractiveness": "ATTRACT",
            "feature_kind": "FEATURE",
            "entrance_id": "ENTRY_ID", "aoi_ref": "AOI_ID", "conspicuity": "VISIBLE",
            "object_id": "OBJECT_ID", "opacity": "OPACITY", "object_type": "OBJ_TYPE",
        },
        "snap_tolerance": 0.5,
        "map_cache": f"{relative}/imported_map.json",
        "osm_source": "osmmap/map.osm",
        "agent_binding_mode": "generated",
        "agent_bindings_file": f"{relative}/agent_bindings.json",
        "interventions": f"{relative}/interventions.json",
        "agents": [
            {
                "agent_id": item["agent_id"],
                "home_aoi_id": item["home_aoi_id"],
                "routine_anchor_aoi_id": item["routine_anchor_aoi_id"],
                "regular_meeting_aoi_ids": item["regular_meeting_aoi_ids"],
            }
            for item in agent_bindings
        ],
        "minimum_agent_age": 60,
        "face_to_face_enabled": True,
        "copresence_min_overlap_minutes": 8.0,
        "experiment_id": "osm_older_adults",
        "output_database": "output/osm_older_adults.db",
    })
    return settings


def write_readme(output_dir: Path, qc: dict[str, Any]) -> None:
    text = f"""# OSM 标准地图包

来源：`../map.osm`，原始文件未修改。地图数据版权属于 OpenStreetMap contributors，许可为 ODbL 1.0。

## 可直接使用的文件

- `imported_map.json`：当前项目 `load_graph_json` 可以直接读取的完整地图；
- `roads.shp`：步行路网和入口连接线；
- `nodes.shp`：路网节点与入口节点；
- `aois.shp`：HOME、WORK、CAFE、PARK、SHOP 五类设施；
- `entrances.shp`：OSM明确入口与算法推断入口；
- `buildings.shp`：视觉遮挡物；
- `entrance_connectors.shp`：入口到步行路网的连接质量检查线；
- `map_qc.html` / `map_qc.json`：处理质量报告；
- `aoi_catalog.csv` / `entrance_catalog.csv`：可人工检查的对象目录；
- `agent_bindings.json`：三名老年Agent的住宅、日常锚点和共同线下会面点；
- `interventions.json`：算法为新地图生成的候选干预，不会覆盖正式实验干预；
- `pycharm_run_osm.example.json`：旧版单文件格式的参考示例，不会覆盖当前实验包。

正式运行读取 `experiments/current/`。地图处理完成后，请检查候选对象，再把确认后的内容复制到
`experiments/current/interventions.json`；不要把自动生成的候选直接当作已审定实验设计。

## 本次结果

- 节点：{qc['walk_network']['final_node_count']}
- 道路：{qc['walk_network']['final_edge_count']}
- AOI：{qc['facilities']['aoi_count']}
- 入口：{qc['entrances']['total']}
- 建筑遮挡物：{qc['facilities']['building_occluder_count']}
- OSM明确入口或门被设施采用：{qc['entrances']['used_osm_entrance_or_gate_count']}
- 算法推断入口必须通过 `SOURCE` 与 `CONF` 字段识别，不能当作OSM真实入口。

如需重新处理：

```powershell
python tools/process_osm_map.py --input osmmap/map.osm --output osmmap/processed
```
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def process(input_path: Path, output_dir: Path) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_nodes, ways, relations, bounds = parse_osm(input_path)
    metric_crs = choose_metric_crs(bounds)
    transformer = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    projected = project_nodes(raw_nodes, transformer)
    nodes, edges, edge_tags, walk_qc = build_walk_graph(ways, projected)
    aoi_sources, building_geometries, _ = create_aoi_sources(raw_nodes, ways, relations, projected)
    if not aoi_sources:
        raise ValueError("No HOME/WORK/CAFE/PARK/SHOP AOIs could be derived from the OSM tags")
    explicit = entrance_candidates(raw_nodes, projected)
    explicit_by_aoi, orphaned = associate_explicit_entrances(explicit, aoi_sources)
    aois, entrances, entrance_details, connector_rows = build_aois_and_entrances(
        aoi_sources, explicit_by_aoi, nodes, edges, edge_tags,
    )
    occluders, occluder_metadata = build_occluders(building_geometries)
    graph = UrbanGraph(nodes, edges, aois, entrances=entrances, occluders=occluders)
    qc = make_qc(
        input_path, raw_nodes, ways, relations, graph, walk_qc, entrance_details,
        explicit, orphaned, bounds, metric_crs,
    )
    source_metadata = {
        item.aoi_id: {
            "osm_type": item.osm_type, "osm_id": item.osm_id,
            "source_geometry": item.source_geometry,
            "selected_tags": {
                key: value for key, value in item.tags.items()
                if key in {"name", "name:zh", "name:en", "building", "amenity", "shop", "leisure", "landuse", "office", "access", "opening_hours"}
            },
        }
        for item in aoi_sources
    }
    graph.metadata = {
        "schema_version": "1.0.1",
        "source": "openstreetmap_xml",
        "source_path": str(input_path),
        "source_hashes": {str(input_path): file_sha256(input_path)},
        "osm_license": "ODbL 1.0; © OpenStreetMap contributors",
        "crs_wkt": metric_crs.to_wkt(),
        "crs_authority": metric_crs.to_authority(),
        "crs_is_metric": True,
        "distance_scale": 60.0,
        "map_qc": qc,
        "aoi_sources": source_metadata,
        "entrance_sources": {item.entrance_id: asdict(item) for item in entrance_details},
    }

    write_shapefiles(
        output_dir, graph, edge_tags, aoi_sources, entrance_details,
        occluder_metadata, connector_rows, metric_crs,
    )
    write_catalogs(output_dir, graph, aoi_sources, entrance_details)
    (output_dir / "map_qc.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    write_qc_html(output_dir / "map_qc.html", graph, entrance_details, qc)
    bindings = suggest_agent_bindings(graph)
    (output_dir / "agent_bindings.json").write_text(
        json.dumps(bindings, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    intervention = suggest_intervention(graph, bindings)
    (output_dir / "interventions.json").write_text(
        json.dumps(intervention, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    current_settings_path = PROJECT_ROOT / "pycharm_run.json"
    current_settings = json.loads(current_settings_path.read_text(encoding="utf-8")) if current_settings_path.exists() else {}
    # 新入口文件只保存当前实验目录。OSM 处理器读取该实验的 map/run 参数作为示例模板，
    # 但生成内容仍只写入 processed/，不会覆盖正式 experiments/ 输入。
    if "active_experiment" in current_settings:
        experiment_dir = PROJECT_ROOT / str(current_settings["active_experiment"])
        template_settings: dict[str, Any] = {}
        for name in ("map.json", "run.json"):
            template_path = experiment_dir / name
            if template_path.exists():
                template_settings.update(json.loads(template_path.read_text(encoding="utf-8")))
        current_settings = template_settings
    run_example = build_run_example(output_dir, bindings, current_settings)
    (output_dir / "pycharm_run_osm.example.json").write_text(
        json.dumps(run_example, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    write_readme(output_dir, qc)
    # Write the cache last so run_pycharm's mtime check will not rebuild it from SHP.
    save_graph_json(graph, output_dir / "imported_map.json")

    loaded = load_graph_json(output_dir / "imported_map.json")
    if not loaded.nodes or not loaded.edges or not loaded.aois or not loaded.entrances:
        raise RuntimeError("Generated imported_map.json failed the project load check")
    return {
        "input": str(input_path), "output": str(output_dir),
        "nodes": len(graph.nodes), "edges": len(graph.edges),
        "aois": len(graph.aois), "entrances": len(graph.entrances),
        "occluders": len(graph.occluders), "aoi_functions": qc["facilities"]["by_function"],
        "entrance_sources": qc["entrances"]["by_source"],
        "agent_bindings": bindings,
        "intervention": intervention,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OSM XML into the Cognitive Map simulation schema")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "osmmap" / "map.osm")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "osmmap" / "processed")
    args = parser.parse_args()
    summary = process(args.input, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
