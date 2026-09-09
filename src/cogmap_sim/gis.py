from __future__ import annotations

import json
import hashlib
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .graph import UrbanGraph
from .models import AOI, Edge, EdgeStatus, Entrance, Node, Occluder, Signage


FIELD_ALIASES = {
    "node_id": ["node_id", "nodeid", "id", "fid"],
    "edge_id": ["edge_id", "edgeid", "road_id", "roadid", "id", "fid"],
    "from_node": ["from_node", "fromnode", "from_id", "source", "u"],
    "to_node": ["to_node", "tonode", "to_id", "target", "v"],
    "status": ["status", "open", "roadstatus"],
    "travel_time": ["travel_time", "traveltime", "time", "cost"],
    "visibility": ["visibility", "visible", "vis"],
    "safety": ["safety", "safe"],
    "comfort": ["comfort"],
    "road_name": ["road_name", "street_name", "name"],
    "road_class": ["road_class", "highway", "class", "type"],
    "aoi_id": ["aoi_id", "aoiid", "poi_id", "id", "fid"],
    "name": ["name", "aoi_name", "poiname"],
    "function": ["function", "landuse", "type", "category", "class"],
    "access_node": ["access_node", "node_id", "nodeid", "entrance"],
    "attractiveness": ["attract", "attractiveness", "score"],
    "feature_kind": ["feature_kind", "feature", "kind", "source"],
    "entrance_id": ["entrance_id", "entry_id", "entrance", "id", "fid"],
    "aoi_ref": ["aoi_id", "aoi_ref", "parent_id"],
    "conspicuity": ["conspicuity", "salience", "visible"],
    "object_id": ["object_id", "building_id", "wall_id", "id", "fid"],
    "opacity": ["opacity", "opaque"],
    "object_type": ["object_type", "type", "class"],
    "sign_id": ["sign_id", "id", "fid"],
    "target_id": ["target_id", "target", "aoi_id"],
    "facing_deg": ["facing_deg", "bearing", "heading"],
    "range_m": ["range_m", "range", "distance"],
    "credibility": ["credibility", "trust", "reliability"],
}


def import_shapefiles(
    roads_path: str | Path,
    aois_path: str | Path,
    nodes_path: str | Path | None = None,
    snap_tolerance: float = 0.001,
    field_mapping: dict[str, str] | None = None,
    entrances_path: str | Path | None = None,
    occluders_path: str | Path | None = None,
    signage_path: str | Path | None = None,
) -> UrbanGraph:
    """Import road polylines and AOI point/polygon features from SHP files.

    If a node layer or from/to fields are absent, topology is inferred by snapping
    road endpoints. Coordinates are preserved in the source CRS.
    """
    try:
        import shapefile
    except ImportError as exc:
        raise RuntimeError("SHP support requires pyshp. Install with: pip install -e .[gis]") from exc

    mapping = field_mapping or {}
    nodes: dict[str, Node] = {}
    point_to_node: list[tuple[float, float, str]] = []
    if nodes_path:
        reader = shapefile.Reader(str(nodes_path))
        for index, shape_record in enumerate(reader.iterShapeRecords()):
            record = _record(shape_record.record, reader)
            point = shape_record.shape.points[0]
            node_id = str(_value(record, "node_id", mapping, f"N{index + 1:06d}"))
            nodes[node_id] = Node(node_id, float(point[0]), float(point[1]))
            point_to_node.append((float(point[0]), float(point[1]), node_id))
        reader.close()

    def snap_node(point: tuple[float, float]) -> str:
        x, y = float(point[0]), float(point[1])
        nearest = min(point_to_node, key=lambda item: math.hypot(x - item[0], y - item[1])) if point_to_node else None
        if nearest and math.hypot(x - nearest[0], y - nearest[1]) <= snap_tolerance:
            return nearest[2]
        node_id = f"N{len(nodes) + 1:06d}"
        nodes[node_id] = Node(node_id, x, y)
        point_to_node.append((x, y, node_id))
        return node_id

    edges: dict[str, Edge] = {}
    roads_reader = shapefile.Reader(str(roads_path))
    for index, shape_record in enumerate(roads_reader.iterShapeRecords()):
        points = shape_record.shape.points
        if len(points) < 2:
            continue
        record = _record(shape_record.record, roads_reader)
        edge_id = str(_value(record, "edge_id", mapping, f"E{index + 1:06d}"))
        explicit_from = _value(record, "from_node", mapping, None)
        explicit_to = _value(record, "to_node", mapping, None)
        from_node = str(explicit_from) if explicit_from is not None and str(explicit_from) in nodes else snap_node(points[0])
        to_node = str(explicit_to) if explicit_to is not None and str(explicit_to) in nodes else snap_node(points[-1])
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
        status_raw = str(_value(record, "status", mapping, "OPEN")).upper()
        status = EdgeStatus.CLOSED if status_raw in {"CLOSED", "CLOSE", "0", "FALSE", "NO"} else EdgeStatus.OPEN
        # When projected coordinates look like metres, estimate walking minutes
        # at roughly 80 m/min if no explicit travel-time field is supplied.
        default_travel_time = max(length / 80.0, 0.05) if length > 10 else max(length, 0.001)
        travel_time = float(_value(record, "travel_time", mapping, default_travel_time))
        edges[edge_id] = Edge(
            edge_id, from_node, to_node, max(length, 0.001), max(travel_time, 0.001), status,
            float(_value(record, "visibility", mapping, 0.65)),
            float(_value(record, "safety", mapping, 0.75)),
            float(_value(record, "comfort", mapping, 0.7)),
            geometry=[(float(point[0]), float(point[1])) for point in points],
            name=str(_value(record, "road_name", mapping, "")),
            road_class=str(_value(record, "road_class", mapping, "")),
        )
    roads_reader.close()

    if not edges:
        raise ValueError("The roads SHP contains no usable polyline features")

    aois: dict[str, AOI] = {}
    aoi_reader = shapefile.Reader(str(aois_path))
    for index, shape_record in enumerate(aoi_reader.iterShapeRecords()):
        record = _record(shape_record.record, aoi_reader)
        points = shape_record.shape.points
        if not points:
            continue
        aoi_id = str(_value(record, "aoi_id", mapping, f"AOI{index + 1:05d}"))
        access = _value(record, "access_node", mapping, None)
        if access is not None and str(access) in nodes:
            node_id = str(access)
        else:
            if len(points) == 1:
                center = points[0]
            else:
                xmin, ymin, xmax, ymax = shape_record.shape.bbox
                center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
            node_id = min(nodes.values(), key=lambda node: math.hypot(node.x - center[0], node.y - center[1])).node_id
        function = str(_value(record, "function", mapping, "OTHER")).upper()
        feature_kind_raw = str(_value(record, "feature_kind", mapping, ""))
        point_geometry = shape_record.shape.shapeType in {1, 8, 11, 18, 21, 28}
        feature_kind = "POI" if point_geometry or "POI" in feature_kind_raw.upper() or "POINT" in feature_kind_raw.upper() else "AOI"
        aois[aoi_id] = AOI(
            aoi_id, str(_value(record, "name", mapping, aoi_id)), node_id, function,
            float(_value(record, "attractiveness", mapping, 0.6)),
            float(_value(record, "safety", mapping, 0.75)),
            float(_value(record, "comfort", mapping, 0.7)),
            geometry=[(float(point[0]), float(point[1])) for point in points],
            feature_kind=feature_kind,
        )
    aoi_reader.close()
    if not aois:
        raise ValueError("The AOI SHP contains no usable point or polygon features")

    entrances: dict[str, Entrance] = {}
    if entrances_path:
        entrance_reader = shapefile.Reader(str(entrances_path))
        for index, shape_record in enumerate(entrance_reader.iterShapeRecords()):
            if not shape_record.shape.points:
                continue
            record = _record(shape_record.record, entrance_reader)
            x, y = map(float, shape_record.shape.points[0])
            entrance_id = str(_value(record, "entrance_id", mapping, f"ENT{index + 1:05d}"))
            aoi_id = str(_value(record, "aoi_ref", mapping, ""))
            if aoi_id not in aois:
                raise ValueError(f"Entrance {entrance_id} references missing AOI: {aoi_id}")
            node_id = min(nodes.values(), key=lambda node: math.hypot(node.x - x, node.y - y)).node_id
            status_raw = str(_value(record, "status", mapping, "OPEN")).upper()
            status = EdgeStatus.CLOSED if status_raw in {"CLOSED", "CLOSE", "0", "FALSE", "NO"} else EdgeStatus.OPEN
            entrances[entrance_id] = Entrance(
                entrance_id, aoi_id, node_id, x, y, status,
                float(_value(record, "conspicuity", mapping, 0.7)),
            )
        entrance_reader.close()

    occluders: dict[str, Occluder] = {}
    if occluders_path:
        occluder_reader = shapefile.Reader(str(occluders_path))
        for index, shape_record in enumerate(occluder_reader.iterShapeRecords()):
            points = [(float(x), float(y)) for x, y in shape_record.shape.points]
            if len(points) < 2:
                continue
            record = _record(shape_record.record, occluder_reader)
            object_id = str(_value(record, "object_id", mapping, f"OCC{index + 1:05d}"))
            occluders[object_id] = Occluder(
                object_id, points,
                float(_value(record, "opacity", mapping, 1.0)),
                str(_value(record, "object_type", mapping, "BUILDING")).upper(),
            )
        occluder_reader.close()

    signage: dict[str, Signage] = {}
    if signage_path:
        sign_reader = shapefile.Reader(str(signage_path))
        for index, shape_record in enumerate(sign_reader.iterShapeRecords()):
            if not shape_record.shape.points:
                continue
            record = _record(shape_record.record, sign_reader)
            x, y = map(float, shape_record.shape.points[0])
            sign_id = str(_value(record, "sign_id", mapping, f"SIGN{index + 1:05d}"))
            signage[sign_id] = Signage(
                sign_id=sign_id,
                target_id=str(_value(record, "target_id", mapping, "")),
                x=x,
                y=y,
                facing_deg=float(_value(record, "facing_deg", mapping, 0.0)),
                range_m=float(_value(record, "range_m", mapping, 60.0)),
                credibility=float(_value(record, "credibility", mapping, 0.8)),
            )
        sign_reader.close()

    lengths = sorted(edge.length for edge in edges.values())
    distance_scale = lengths[len(lengths) // 2] if lengths else 1.0
    prj = Path(roads_path).with_suffix(".prj")
    source_paths = [Path(roads_path), Path(aois_path)]
    source_paths.extend(Path(path) for path in (nodes_path, entrances_path, occluders_path, signage_path) if path)
    crs_wkt = prj.read_text(encoding="utf-8", errors="ignore") if prj.exists() else None
    crs_is_metric = None
    if crs_wkt:
        try:
            from pyproj import CRS
            crs = CRS.from_wkt(crs_wkt)
            crs_is_metric = bool(crs.is_projected and all(axis.unit_name.lower() in {"metre", "meter"} for axis in crs.axis_info))
        except (ImportError, ValueError):
            crs_is_metric = "unit[\"meter\"" in crs_wkt.lower() or "unit[\"metre\"" in crs_wkt.lower()
    try:
        import networkx as nx

        topology = nx.Graph()
        topology.add_nodes_from(nodes)
        topology.add_edges_from((edge.from_node, edge.to_node) for edge in edges.values())
        components = list(nx.connected_components(topology))
        component_count = len(components)
        largest_component_ratio = max((len(item) for item in components), default=0) / max(1, len(nodes))
        isolated_node_count = len(list(nx.isolates(topology)))
    except ImportError:  # pragma: no cover - optional diagnostic fallback
        component_count, largest_component_ratio, isolated_node_count = None, None, None
    try:
        from shapely.geometry import LineString, Polygon

        invalid_geometry_count = sum(not LineString(edge.geometry).is_valid for edge in edges.values() if edge.geometry)
        invalid_geometry_count += sum(
            not Polygon(aoi.geometry).is_valid for aoi in aois.values() if len(aoi.geometry) >= 3
        )
    except ImportError:  # pragma: no cover
        invalid_geometry_count = None
    metadata = {
        "schema_version": "1.0.1",
        "source": "shapefile",
        "roads_path": str(Path(roads_path).resolve()),
        "aois_path": str(Path(aois_path).resolve()),
        "nodes_path": str(Path(nodes_path).resolve()) if nodes_path else None,
        "entrances_path": str(Path(entrances_path).resolve()) if entrances_path else None,
        "occluders_path": str(Path(occluders_path).resolve()) if occluders_path else None,
        "signage_path": str(Path(signage_path).resolve()) if signage_path else None,
        "crs_wkt": crs_wkt,
        "crs_is_metric": crs_is_metric,
        "distance_scale": distance_scale,
        "source_hashes": {str(path.resolve()): shapefile_bundle_hash(path) for path in source_paths},
        "map_qc": {
            "node_count": len(nodes), "edge_count": len(edges), "aoi_count": len(aois),
            "entrance_count": len(entrances), "occluder_count": len(occluders),
            "duplicate_ids": 0, "metric_crs": crs_is_metric,
            "component_count": component_count,
            "largest_component_ratio": largest_component_ratio,
            "isolated_node_count": isolated_node_count,
            "invalid_geometry_count": invalid_geometry_count,
        },
    }
    return UrbanGraph(nodes, edges, aois, metadata, entrances, occluders, signage)


def save_graph_json(graph: UrbanGraph, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "metadata": graph.metadata,
        "nodes": [asdict(node) for node in graph.nodes.values()],
        "edges": [{**asdict(edge), "status": str(edge.status)} for edge in graph.edges.values()],
        "aois": [asdict(aoi) for aoi in graph.aois.values()],
        "entrances": [{**asdict(item), "status": str(item.status)} for item in graph.entrances.values()],
        "occluders": [asdict(item) for item in graph.occluders.values()],
        "signage": [asdict(item) for item in graph.signage.values()],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def load_graph_json(path: str | Path) -> UrbanGraph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = {item["node_id"]: Node(**item) for item in data["nodes"]}
    edges = {}
    for item in data["edges"]:
        item["status"] = EdgeStatus(item["status"])
        edges[item["edge_id"]] = Edge(**item)
    aois = {item["aoi_id"]: AOI(**item) for item in data["aois"]}
    entrances = {}
    for item in data.get("entrances", []):
        item["status"] = EdgeStatus(item["status"])
        entrances[item["entrance_id"]] = Entrance(**item)
    occluders = {item["object_id"]: Occluder(**item) for item in data.get("occluders", [])}
    signage = {item["sign_id"]: Signage(**item) for item in data.get("signage", [])}
    return UrbanGraph(nodes, edges, aois, data.get("metadata"), entrances, occluders, signage)


def _record(record: Any, reader: Any) -> dict[str, Any]:
    if hasattr(record, "as_dict"):
        return {str(key).lower(): value for key, value in record.as_dict().items()}
    fields = [field[0] for field in reader.fields[1:]]
    return {str(key).lower(): value for key, value in zip(fields, record)}


def _value(record: dict[str, Any], logical_name: str, mapping: dict[str, str], default: Any) -> Any:
    requested = mapping.get(logical_name)
    if requested and requested.lower() in record and record[requested.lower()] not in (None, ""):
        return record[requested.lower()]
    for alias in FIELD_ALIASES[logical_name]:
        if alias.lower() in record and record[alias.lower()] not in (None, ""):
            return record[alias.lower()]
    return default


def shapefile_bundle_hash(path: Path) -> str:
    """Hash a shapefile and its sidecar files independently of its directory."""
    digest = hashlib.sha256()
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        component = path.with_suffix(suffix)
        if not component.exists():
            continue
        digest.update(component.name.lower().encode("utf-8"))
        with component.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()
