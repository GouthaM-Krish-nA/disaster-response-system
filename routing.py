"""
routing.py

A* pathfinding for "find the fastest safe route to a report location" —
now running over Chennai's REAL street network (downloaded once via
build_street_graph.py and saved to chennai_streets.graphml), instead of an
abstract coordinate grid. The A* algorithm itself is unchanged from before;
only the graph it searches over is now real roads instead of made-up points.

Algorithm: A* search with a great-circle-distance heuristic.
  - Nodes: real road intersections (from OpenStreetMap, via OSMnx).
  - Edges: real road segments, weighted by their actual length in meters.
  - Heuristic h(n): straight-line ("as the crow flies") distance from n to
    the goal, in meters. This is admissible (never overestimates) because no
    real road route can be shorter than a straight line between two points —
    which is what guarantees A* finds the optimal route, not just a good one.
  - Blockage: intersections within a threshold radius of an active road-block
    report have their touching edges' cost multiplied by a large penalty
    factor, so A* strongly prefers avoiding them without making them
    impassable (a real "blocked" road usually still has a slower detour).

Requires chennai_streets.graphml to exist (run build_street_graph.py once
to create it) — this file is loaded ONCE, when the server starts, not on
every route request.
"""

import heapq
import math
from dataclasses import dataclass

import osmnx as ox

from data_generator import CENTER_LAT, CENTER_LNG, SPREAD

GRAPH_PATH = "chennai_streets.graphml"
BLOCK_RADIUS_METERS = 1300   # ~1.3km — same real-world scale as the old grid version
BLOCK_PENALTY = 40           # edges near a block cost 40x more (strongly avoided, not impossible)

NodeId = int  # OSMnx node ids are plain integers


def _load_graph():
    print(f"[routing] loading street network from {GRAPH_PATH} ...")
    G = ox.load_graphml(GRAPH_PATH)
    print(f"[routing] loaded {len(G.nodes)} intersections, {len(G.edges)} road segments")
    return G


GRAPH = _load_graph()


def _node_latlng(node: NodeId) -> tuple[float, float]:
    data = GRAPH.nodes[node]
    return data["y"], data["x"]  # OSMnx stores latitude as 'y', longitude as 'x'


def _haversine_m(a_latlng, b_latlng) -> float:
    """
    Great-circle distance in meters between two (lat, lng) points.
    Used instead of flat Euclidean distance because lat/lng degrees don't
    correspond to equal real-world distances — this matters more now that
    coordinates span real street data rather than a small synthetic grid.
    """
    lat1, lng1 = math.radians(a_latlng[0]), math.radians(a_latlng[1])
    lat2, lng2 = math.radians(b_latlng[0]), math.radians(b_latlng[1])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(a))  # Earth radius ~6,371,000 m


def _nearest_node(lat: float, lng: float) -> NodeId:
    return ox.distance.nearest_nodes(GRAPH, X=lng, Y=lat)


def _blocked_nodes(active_road_blocks: list[dict]) -> set[NodeId]:
    """Real intersections within BLOCK_RADIUS_METERS of an active, serious road block."""
    blocked = set()
    for report in active_road_blocks:
        center = (report["lat"], report["lng"])
        for node in GRAPH.nodes:
            if _haversine_m(center, _node_latlng(node)) <= BLOCK_RADIUS_METERS:
                blocked.add(node)
    return blocked


@dataclass
class RouteResult:
    waypoints: list[dict]   # [{"lat":..., "lng":...}, ...]
    total_cost: float
    blocked_nodes_avoided: int
    nodes_expanded: int      # useful to report in your viva: search effort


def find_route(start_latlng: tuple[float, float], goal_latlng: tuple[float, float],
                active_road_blocks: list[dict]) -> RouteResult:
    """
    A* search from start to goal over the real street graph, avoiding
    (penalizing) intersections near active road-block reports.
    """
    start = _nearest_node(*start_latlng)
    goal = _nearest_node(*goal_latlng)
    blocked = _blocked_nodes(active_road_blocks)
    goal_latlng_real = _node_latlng(goal)

    open_set = [(0.0, start)]
    came_from: dict[NodeId, NodeId] = {}
    g_score = {start: 0.0}
    nodes_expanded = 0
    visited = set()  # avoids reprocessing settled nodes — matters now with ~35k nodes, not ~150

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            break

        for neighbor in GRAPH.neighbors(current):
            # A MultiDiGraph can have more than one edge between the same two
            # nodes (e.g. a service lane); take the shortest of them.
            edge_data = min(GRAPH[current][neighbor].values(), key=lambda d: d.get("length", 1.0))
            edge_cost = float(edge_data.get("length", 1.0))
            if neighbor in blocked or current in blocked:
                edge_cost *= BLOCK_PENALTY

            tentative_g = g_score[current] + edge_cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = _haversine_m(_node_latlng(neighbor), goal_latlng_real)  # admissible heuristic
                heapq.heappush(open_set, (tentative_g + h, neighbor))

    # Reconstruct path
    path = [goal]
    while path[-1] != start and path[-1] in came_from:
        path.append(came_from[path[-1]])
    path.reverse()

    waypoints = [{"lat": _node_latlng(n)[0], "lng": _node_latlng(n)[1]} for n in path]
    # Snap just the endpoints of the drawn line to the exact report/HQ
    # coordinates, same reasoning as before: the nearest real intersection
    # can be a short distance from the precise coordinate.
    if waypoints:
        waypoints[0] = {"lat": start_latlng[0], "lng": start_latlng[1]}
        waypoints[-1] = {"lat": goal_latlng[0], "lng": goal_latlng[1]}
    avoided = sum(1 for n in path if n in blocked)

    return RouteResult(
        waypoints=waypoints,
        total_cost=round(g_score.get(goal, float("inf")), 2),
        blocked_nodes_avoided=avoided,
        nodes_expanded=nodes_expanded,
    )


# Fixed rescue HQ — nearest real intersection to where the old grid's corner
# used to be, so HQ stays in roughly the same place as before.
_hq_lat = CENTER_LAT - SPREAD
_hq_lng = CENTER_LNG - SPREAD
_hq_node = _nearest_node(_hq_lat, _hq_lng)
RESCUE_HQ = _node_latlng(_hq_node)