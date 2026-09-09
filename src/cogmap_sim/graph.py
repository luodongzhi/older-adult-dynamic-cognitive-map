from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Callable

from .models import AOI, Edge, EdgeStatus, Entrance, Node, Occluder, Signage


class UrbanGraph:
    def __init__(
        self,
        nodes: dict[str, Node],
        edges: dict[str, Edge],
        aois: dict[str, AOI],
        metadata: dict | None = None,
        entrances: dict[str, Entrance] | None = None,
        occluders: dict[str, Occluder] | None = None,
        signage: dict[str, Signage] | None = None,
    ):
        self.nodes = nodes
        self.edges = edges
        self.aois = aois
        self.metadata = metadata or {}
        self.entrances = entrances or {}
        self.occluders = occluders or {}
        self.signage = signage or {}
        self.adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for edge in edges.values():
            self.adjacency[edge.from_node].append((edge.to_node, edge.edge_id))
            self.adjacency[edge.to_node].append((edge.from_node, edge.edge_id))

    def clone(self) -> "UrbanGraph":
        import copy
        return copy.deepcopy(self)

    def shortest_path(
        self,
        start: str,
        goal: str,
        allowed: Callable[[Edge], bool] | None = None,
        cost: Callable[[Edge], float] | None = None,
    ) -> list[str] | None:
        if start == goal:
            return []
        allowed = allowed or (lambda e: e.status == EdgeStatus.OPEN)
        cost = cost or (lambda e: e.base_travel_time)
        queue: list[tuple[float, str]] = [(0.0, start)]
        best = {start: 0.0}
        parent: dict[str, tuple[str, str]] = {}
        while queue:
            dist, node = heapq.heappop(queue)
            if dist != best.get(node):
                continue
            if node == goal:
                break
            for nxt, edge_id in self.adjacency.get(node, []):
                edge = self.edges[edge_id]
                if not allowed(edge):
                    continue
                new_dist = dist + max(0.001, cost(edge))
                if new_dist < best.get(nxt, math.inf):
                    best[nxt] = new_dist
                    parent[nxt] = (node, edge_id)
                    heapq.heappush(queue, (new_dist, nxt))
        if goal not in parent:
            return None
        path: list[str] = []
        current = goal
        while current != start:
            previous, edge_id = parent[current]
            path.append(edge_id)
            current = previous
        path.reverse()
        return path

    def other_node(self, edge_id: str, current: str) -> str:
        edge = self.edges[edge_id]
        return edge.to_node if edge.from_node == current else edge.from_node

    def path_metrics(self, edge_ids: list[str]) -> tuple[float, float]:
        return (
            sum(self.edges[e].length for e in edge_ids),
            sum(self.edges[e].base_travel_time for e in edge_ids),
        )

    def node_distance(self, left: str, right: str) -> float:
        a, b = self.nodes[left], self.nodes[right]
        return math.hypot(a.x - b.x, a.y - b.y)

    def nearby_edges(self, node_id: str, radius: float = 1.25) -> list[str]:
        node = self.nodes[node_id]
        found = []
        for edge in self.edges.values():
            a, b = self.nodes[edge.from_node], self.nodes[edge.to_node]
            if min(math.hypot(node.x - a.x, node.y - a.y), math.hypot(node.x - b.x, node.y - b.y)) <= radius:
                found.append(edge.edge_id)
        return found

    def distance_from_route(self, route_edge_ids: list[str], target_edge_id: str | None = None, target_aoi_id: str | None = None) -> float:
        """Geometry distance; uses Shapely when installed and a node fallback otherwise."""
        route_lines = [self._edge_coordinates(self.edges[edge_id]) for edge_id in route_edge_ids if edge_id in self.edges]
        if not route_lines:
            return math.inf
        try:
            from shapely.geometry import LineString, MultiLineString, Point, Polygon
            route_geometry = MultiLineString([line for line in route_lines if len(line) >= 2])
            if target_edge_id:
                target = LineString(self._edge_coordinates(self.edges[target_edge_id]))
            else:
                aoi = self.aois[target_aoi_id or ""]
                if len(aoi.geometry) >= 3:
                    target = Polygon(aoi.geometry)
                else:
                    node = self.nodes[aoi.node_id]
                    target = Point(node.x, node.y)
            return float(route_geometry.distance(target))
        except (ImportError, ValueError):
            route_nodes = set()
            for edge_id in route_edge_ids:
                edge = self.edges[edge_id]
                route_nodes.update((edge.from_node, edge.to_node))
            if target_edge_id:
                target_nodes = (self.edges[target_edge_id].from_node, self.edges[target_edge_id].to_node)
            else:
                target_nodes = (self.aois[target_aoi_id or ""].node_id,)
            return min(self.node_distance(left, right) for left in route_nodes for right in target_nodes)

    def _edge_coordinates(self, edge: Edge) -> list[tuple[float, float]]:
        if edge.geometry:
            return [(float(point[0]), float(point[1])) for point in edge.geometry]
        start, end = self.nodes[edge.from_node], self.nodes[edge.to_node]
        return [(start.x, start.y), (end.x, end.y)]

    def intervention_object_ids(self) -> list[tuple[str, str]]:
        objects = [("ROAD_EDGE", edge.edge_id) for edge in self.edges.values() if edge.is_intervention]
        objects.extend(("AOI", aoi.aoi_id) for aoi in self.aois.values() if aoi.is_intervention)
        objects.extend(("ENTRANCE", entrance.entrance_id) for entrance in self.entrances.values() if entrance.is_intervention)
        return objects
