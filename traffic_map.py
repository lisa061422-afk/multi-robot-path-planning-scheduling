"""Traffic-network map utilities for centralized path/schedule search.

This module intentionally models only the coarse resource layer first:
each intersection is one resource, roads are graph edges, and every exposed
directional slot on the outside of the graph can become an entrance/exit port.

The next layer can attach per-intersection conflict-space data to the same
intersection IDs without changing the route enumeration API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple


Direction = str
IntersectionId = int
PortId = int

DIR_ORDER: Tuple[Direction, ...] = ("L", "D", "R", "U")
DIR_DELTA: Mapping[Direction, Tuple[int, int]] = {
    "L": (-1, 0),
    "D": (0, -1),
    "R": (1, 0),
    "U": (0, 1),
}
OPPOSITE: Mapping[Direction, Direction] = {
    "L": "R",
    "D": "U",
    "R": "L",
    "U": "D",
}
DIRECTION_ALIASES: Mapping[Direction, Direction] = {
    "L": "L",
    "LEFT": "L",
    "W": "L",
    "WEST": "L",
    "D": "D",
    "DOWN": "D",
    "S": "D",
    "SOUTH": "D",
    "R": "R",
    "RIGHT": "R",
    "E": "R",
    "EAST": "R",
    "U": "U",
    "UP": "U",
    "N": "U",
    "NORTH": "U",
}

ROUTE_ID_BY_ENTRY_EXIT: Mapping[Tuple[Direction, Direction], int] = {
    ("D", "L"): 1,
    ("D", "U"): 2,
    ("D", "R"): 3,
    ("R", "D"): 4,
    ("R", "L"): 5,
    ("R", "U"): 6,
    ("U", "R"): 7,
    ("U", "D"): 8,
    ("U", "L"): 9,
    ("L", "U"): 10,
    ("L", "R"): 11,
    ("L", "D"): 12,
}

TURN_BY_ROUTE_ID: Mapping[int, str] = {
    1: "left",
    2: "straight",
    3: "right",
    4: "left",
    5: "straight",
    6: "right",
    7: "left",
    8: "straight",
    9: "right",
    10: "left",
    11: "straight",
    12: "right",
}

DEFAULT_TURN_DURATION: Mapping[str, float] = {
    "left": 3.0 * math.pi / 4.0,
    "straight": 2.0,
    "right": math.pi / 4.0,
}

DEFAULT_TURN_SPACE_DURATIONS: Mapping[str, Tuple[float, ...]] = {
    "left": (math.pi / 4.0, math.pi / 4.0, math.pi / 4.0),
    "straight": (1.0, 1.0),
    "right": (math.pi / 4.0,),
}


@dataclass(frozen=True)
class Port:
    """A boundary channel attached to one side of an intersection."""

    id: PortId
    intersection: IntersectionId
    direction: Direction


@dataclass(frozen=True)
class Road:
    """An internal road/buffer connecting two intersections."""

    id: int
    a: IntersectionId
    b: IntersectionId

    @property
    def endpoints(self) -> Tuple[IntersectionId, IntersectionId]:
        return tuple(sorted((self.a, self.b)))


@dataclass(frozen=True)
class IntersectionTraversal:
    """How one route option passes through one intersection."""

    intersection: IntersectionId
    path_index: int
    entry_dir: Direction
    exit_dir: Direction
    turn: str
    route_id: int
    execution_time: float
    space_durations: Tuple[float, ...]


@dataclass(frozen=True)
class BranchPoint:
    """A route-choice point for one OD pair.

    `prefix` is the intersection sequence already traversed up to `intersection`.
    `next_intersections` are the feasible next intersection choices appearing in
    the enumerated OD path set for that exact prefix.
    """

    intersection: IntersectionId
    path_index: int
    prefix: Tuple[IntersectionId, ...]
    next_intersections: Tuple[IntersectionId, ...]


@dataclass(frozen=True)
class RouteOption:
    """One simple path from an entrance port to an exit port."""

    id: int
    entrance: PortId
    exit: PortId
    intersections: Tuple[IntersectionId, ...]
    edges: Tuple[Tuple[IntersectionId, IntersectionId], ...]
    traversals: Tuple[IntersectionTraversal, ...]
    branch_points: Tuple[BranchPoint, ...]

    @property
    def resource_sequence(self) -> Tuple[IntersectionId, ...]:
        """Coarse scheduling resources: one resource per intersection."""

        return self.intersections

    @property
    def execution_times(self) -> Tuple[float, ...]:
        """Coarse execution time at each intersection in `resource_sequence`."""

        return tuple(item.execution_time for item in self.traversals)


@dataclass(frozen=True)
class VehicleRouteOptions:
    """All route choices for one vehicle OD request."""

    vehicle_id: int
    entrance: PortId
    exit: PortId
    options: Tuple[RouteOption, ...]

    @property
    def branch_intersections(self) -> Tuple[IntersectionId, ...]:
        out: List[IntersectionId] = []
        seen = set()
        for opt in self.options:
            for bp in opt.branch_points:
                if bp.intersection not in seen:
                    seen.add(bp.intersection)
                    out.append(bp.intersection)
        return tuple(out)

    def next_choices_after(
        self, prefix: Sequence[IntersectionId]
    ) -> Tuple[IntersectionId, ...]:
        """Return feasible next intersections after the current traveled prefix.

        This is the runtime query the centralized decision tree will need at a
        significant moment. If the returned tuple has length > 1, the vehicle is
        currently at a route-choice branch point.
        """

        prefix_tuple = tuple(prefix)
        choices = set()
        for option in self.options:
            path = option.intersections
            if len(prefix_tuple) >= len(path):
                continue
            if path[: len(prefix_tuple)] == prefix_tuple:
                choices.add(path[len(prefix_tuple)])
        return tuple(sorted(choices))

    def is_branch_after(self, prefix: Sequence[IntersectionId]) -> bool:
        return len(self.next_choices_after(prefix)) > 1


class TrafficMap:
    """A grid-embedded intersection graph with automatically generated ports."""

    def __init__(
        self,
        *,
        coords: Mapping[IntersectionId, Tuple[int, int]],
        edges: Iterable[Tuple[IntersectionId, IntersectionId]],
        ports: Optional[Iterable[Tuple[IntersectionId, Direction]]] = None,
        auto_ports: bool = True,
        preserve_port_order: bool = False,
        road_ids: Optional[Mapping[Tuple[IntersectionId, IntersectionId], int]] = None,
        name: str = "manual",
    ) -> None:
        if not coords:
            raise ValueError("coords must contain at least one intersection")
        self.name = name
        self.coords: Dict[IntersectionId, Tuple[int, int]] = {
            int(i): (int(x), int(y)) for i, (x, y) in coords.items()
        }
        self._validate_intersection_ids()

        self.adjacency: Dict[IntersectionId, Dict[Direction, Optional[IntersectionId]]] = {
            i: {d: None for d in DIR_ORDER} for i in self.coords
        }
        self.edges: FrozenSet[FrozenSet[IntersectionId]] = self._build_edges(edges)
        self.roads: Dict[int, Road] = self._build_roads(road_ids)
        self.road_by_edge: Dict[FrozenSet[IntersectionId], int] = {
            frozenset(road.endpoints): road.id for road in self.roads.values()
        }

        port_specs = list(ports or [])
        if auto_ports:
            port_specs.extend(self._free_direction_slots())
        self.ports: Dict[PortId, Port] = self._build_ports(
            port_specs,
            preserve_order=preserve_port_order,
        )
        self.port_by_location: Dict[Tuple[IntersectionId, Direction], PortId] = {
            (p.intersection, p.direction): p.id for p in self.ports.values()
        }

    @classmethod
    def from_grid(
        cls,
        coords: Mapping[IntersectionId, Tuple[int, int]],
        *,
        edges: Optional[Iterable[Tuple[IntersectionId, IntersectionId]]] = None,
        excluded_edges: Optional[Iterable[Tuple[IntersectionId, IntersectionId]]] = None,
        auto_ports: bool = True,
        ports: Optional[Iterable[Tuple[IntersectionId, Direction]]] = None,
        preserve_port_order: bool = False,
        road_ids: Optional[Mapping[Tuple[IntersectionId, IntersectionId], int]] = None,
        name: str = "manual_grid",
    ) -> "TrafficMap":
        """Build a map from intersection coordinates.

        If `edges` is omitted, every axis-adjacent unit grid neighbor is linked.
        If `edges` is supplied, only those manual edges are used.
        """

        clean_coords = {int(i): (int(x), int(y)) for i, (x, y) in coords.items()}
        if edges is None:
            excluded = {frozenset((int(a), int(b))) for a, b in (excluded_edges or [])}
            coord_to_id = {xy: i for i, xy in clean_coords.items()}
            derived_edges: List[Tuple[int, int]] = []
            for i, (x, y) in clean_coords.items():
                for d in ("R", "U"):
                    dx, dy = DIR_DELTA[d]
                    j = coord_to_id.get((x + dx, y + dy))
                    if j is None:
                        continue
                    key = frozenset((i, j))
                    if key not in excluded:
                        derived_edges.append((i, j))
            edges = derived_edges
        return cls(
            coords=clean_coords,
            edges=edges,
            ports=ports,
            auto_ports=auto_ports,
            preserve_port_order=preserve_port_order,
            road_ids=road_ids,
            name=name,
        )

    @classmethod
    def rectangular_grid(
        cls,
        width: int,
        height: int,
        *,
        name: Optional[str] = None,
    ) -> "TrafficMap":
        """Create a width x height grid with all perimeter slots as ports.

        IDs are assigned bottom-to-top, left-to-right:
        `(x, y) -> 1 + y * width + x`.
        """

        if width < 1 or height < 1:
            raise ValueError("width and height must be positive")
        coords = {
            1 + y * width + x: (x, y)
            for y in range(height)
            for x in range(width)
        }
        return cls.from_grid(coords, name=name or f"grid_{width}x{height}")

    @classmethod
    def paper_2x2(cls) -> "TrafficMap":
        """The 2x2 layout with ports ordered counterclockwise around the boundary."""

        return cls.from_grid(
            coords={
                1: (0, 0),
                2: (1, 0),
                3: (1, 1),
                4: (0, 1),
            },
            ports=[
                (1, "L"),  # port 1
                (1, "D"),  # port 2
                (2, "D"),  # port 3
                (2, "R"),  # port 4
                (3, "R"),  # port 5
                (3, "U"),  # port 6
                (4, "U"),  # port 7
                (4, "L"),  # port 8
            ],
            auto_ports=False,
            preserve_port_order=True,
            road_ids={
                (1, 2): 5,
                (2, 3): 6,
                (3, 4): 7,
                (4, 1): 8,
            },
            name="paper_2x2",
        )

    @classmethod
    def paper_3x3(cls) -> "TrafficMap":
        """A fixed 3x3 layout with ports ordered counterclockwise around the boundary."""

        return cls.from_grid(
            coords={
                1: (0, 0),
                2: (1, 0),
                3: (2, 0),
                4: (0, 1),
                5: (1, 1),
                6: (2, 1),
                7: (0, 2),
                8: (1, 2),
                9: (2, 2),
            },
            ports=[
                (1, "L"),  # port 1
                (1, "D"),  # port 2
                (2, "D"),  # port 3
                (3, "D"),  # port 4
                (3, "R"),  # port 5
                (6, "R"),  # port 6
                (9, "R"),  # port 7
                (9, "U"),  # port 8
                (8, "U"),  # port 9
                (7, "U"),  # port 10
                (7, "L"),  # port 11
                (4, "L"),  # port 12
            ],
            auto_ports=False,
            preserve_port_order=True,
            road_ids={
                (1, 2): 10,
                (2, 3): 11,
                (4, 5): 12,
                (5, 6): 13,
                (7, 8): 14,
                (8, 9): 15,
                (1, 4): 16,
                (4, 7): 17,
                (2, 5): 18,
                (5, 8): 19,
                (3, 6): 20,
                (6, 9): 21,
            },
            name="paper_3x3",
        )

    @classmethod
    def ppt_2x2(cls) -> "TrafficMap":
        """The 2x2 layout using the port numbering shown in the case-study PPT."""

        return cls.from_grid(
            coords={
                1: (0, 0),
                2: (1, 0),
                3: (1, 1),
                4: (0, 1),
            },
            ports=[
                (1, "D"),  # port 1
                (2, "D"),  # port 2
                (2, "R"),  # port 3
                (3, "R"),  # port 4
                (3, "U"),  # port 5
                (4, "U"),  # port 6
                (4, "L"),  # port 7
                (1, "L"),  # port 8
            ],
            auto_ports=False,
            preserve_port_order=True,
            road_ids={
                (1, 2): 5,
                (2, 3): 6,
                (3, 4): 7,
                (4, 1): 8,
            },
            name="ppt_2x2",
        )

    @property
    def intersection_ids(self) -> Tuple[IntersectionId, ...]:
        return tuple(sorted(self.coords))

    @property
    def port_ids(self) -> Tuple[PortId, ...]:
        return tuple(sorted(self.ports))

    @property
    def road_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self.roads))

    def find_port(self, intersection: IntersectionId, direction: Direction) -> PortId:
        """Return the port attached at `(intersection, direction)`."""

        key = (intersection, self._normalize_direction(direction))
        if key not in self.port_by_location:
            raise KeyError(f"no port at intersection {intersection} direction {direction}")
        return self.port_by_location[key]

    def port_location(self, port_id: PortId) -> Tuple[IntersectionId, Direction]:
        """Return `(intersection, direction)` for a global port ID."""

        port = self._require_port(port_id)
        return port.intersection, port.direction

    def describe_port(self, port_id: PortId) -> str:
        """Return one human-readable port description."""

        intersection, direction = self.port_location(port_id)
        return f"port {port_id}: intersection {intersection} {direction}"

    def neighbors(self, intersection: IntersectionId) -> Tuple[IntersectionId, ...]:
        self._require_intersection(intersection)
        values = [v for v in self.adjacency[intersection].values() if v is not None]
        return tuple(sorted(values))

    def road_id_between(self, a: IntersectionId, b: IntersectionId) -> int:
        """Return the global road/buffer ID between two adjacent intersections."""

        key = frozenset((a, b))
        if key not in self.road_by_edge:
            raise KeyError(f"no road between intersection {a} and {b}")
        return self.road_by_edge[key]

    def enumerate_intersection_paths(
        self,
        entrance: PortId,
        exit: PortId,
        *,
        max_hops: Optional[int] = None,
        max_paths: Optional[int] = None,
    ) -> Tuple[Tuple[IntersectionId, ...], ...]:
        """Enumerate all simple intersection paths for one OD pair.

        `max_hops` counts intersections in the path. By default, it is the
        number of intersections, which means every returned path is simple.
        """

        start = self._require_port(entrance).intersection
        goal = self._require_port(exit).intersection
        if entrance == exit:
            raise ValueError("entrance and exit ports must be different")

        if max_hops is None:
            max_hops = len(self.coords)
        if max_hops < 1:
            raise ValueError("max_hops must be positive")

        if start == goal:
            return ((start,),)

        found: List[Tuple[IntersectionId, ...]] = []

        def dfs(cur: IntersectionId, path: List[IntersectionId], visited: set[int]) -> None:
            if max_paths is not None and len(found) >= max_paths:
                return
            if len(path) > max_hops:
                return
            if cur == goal:
                found.append(tuple(path))
                return
            for nb in self.neighbors(cur):
                if nb in visited:
                    continue
                visited.add(nb)
                path.append(nb)
                dfs(nb, path, visited)
                path.pop()
                visited.remove(nb)

        dfs(start, [start], {start})
        return tuple(found)

    def route_options(
        self,
        entrance: PortId,
        exit: PortId,
        *,
        max_hops: Optional[int] = None,
        max_paths: Optional[int] = None,
    ) -> Tuple[RouteOption, ...]:
        """Build RouteOption objects and mark path-dependent branch points."""

        paths = self.enumerate_intersection_paths(
            entrance, exit, max_hops=max_hops, max_paths=max_paths
        )
        prefix_choices = self._prefix_next_choices(paths)

        options: List[RouteOption] = []
        for opt_id, path in enumerate(paths, start=1):
            edges = tuple((path[i], path[i + 1]) for i in range(len(path) - 1))
            traversals = self._build_traversals(path, entrance, exit)
            if not traversals:
                continue
            bps: List[BranchPoint] = []
            for idx in range(len(path) - 1):
                prefix = path[: idx + 1]
                choices = prefix_choices.get(prefix, ())
                if len(choices) > 1:
                    bps.append(
                        BranchPoint(
                            intersection=path[idx],
                            path_index=idx,
                            prefix=prefix,
                            next_intersections=choices,
                        )
                    )
            options.append(
                RouteOption(
                    id=opt_id,
                    entrance=entrance,
                    exit=exit,
                    intersections=path,
                    edges=edges,
                    traversals=traversals,
                    branch_points=tuple(bps),
                )
            )
        return tuple(options)

    def vehicle_route_options(
        self,
        vehicle_id: int,
        entrance: PortId,
        exit: PortId,
        *,
        max_hops: Optional[int] = None,
        max_paths: Optional[int] = None,
    ) -> VehicleRouteOptions:
        return VehicleRouteOptions(
            vehicle_id=vehicle_id,
            entrance=entrance,
            exit=exit,
            options=self.route_options(
                entrance, exit, max_hops=max_hops, max_paths=max_paths
            ),
        )

    def shortest_route_option(
        self,
        entrance: PortId,
        exit: PortId,
        *,
        road_time: float = 3.0,
        max_hops: Optional[int] = None,
        max_paths: Optional[int] = None,
    ) -> RouteOption:
        """Pick one predetermined route for the fixed-path scheduler.

        The rule is intentionally simple:
        1. minimize the number of intersections in the path;
        2. if tied, minimize sum(intersection execution times) + road_count * road_time;
        3. if still tied, use the stable enumeration order.
        """

        options = self.route_options(
            entrance,
            exit,
            max_hops=max_hops,
            max_paths=max_paths,
        )
        if not options:
            raise ValueError(f"no feasible route from port {entrance} to port {exit}")

        return min(
            options,
            key=lambda opt: (
                len(opt.resource_sequence),
                sum(opt.execution_times) + len(opt.edges) * road_time,
                opt.id,
            ),
        )

    def describe_ports(self) -> List[str]:
        """Stable human-readable port table."""

        return [
            f"port {p.id}: intersection {p.intersection} {p.direction}"
            for p in (self.ports[i] for i in self.port_ids)
        ]

    def describe_roads(self) -> List[str]:
        """Stable human-readable road table."""

        return [
            f"road {r.id}: intersection {r.endpoints[0]} <-> {r.endpoints[1]}"
            for r in (self.roads[i] for i in self.road_ids)
        ]

    def write_svg(self, path: str | Path, *, scale: int = 120, margin: int = 80) -> Path:
        """Write a simple topology drawing with intersections, roads, and ports.

        This intentionally avoids plotting dependencies so the map can be
        checked immediately on a clean Python install.
        """

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        xs = [xy[0] for xy in self.coords.values()]
        ys = [xy[1] for xy in self.coords.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = (max_x - min_x) * scale + 2 * margin
        height = (max_y - min_y) * scale + 2 * margin

        def point(x: float, y: float) -> Tuple[float, float]:
            px = margin + (x - min_x) * scale
            py = margin + (max_y - y) * scale
            return px, py

        def text(x: float, y: float, label: str, size: int = 14, weight: str = "400") -> str:
            return (
                f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" '
                f'dominant-baseline="middle" font-family="Arial" '
                f'font-size="{size}" font-weight="{weight}">{label}</text>'
            )

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
        ]

        for edge in sorted(self.edges, key=lambda e: tuple(sorted(e))):
            a, b = sorted(edge)
            ax, ay = point(*self.coords[a])
            bx, by = point(*self.coords[b])
            parts.append(
                f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
                'stroke="#555" stroke-width="10" stroke-linecap="round"/>'
            )
            road_id = self.road_id_between(a, b)
            parts.append(
                f'<rect x="{(ax + bx) / 2 - 18:.1f}" y="{(ay + by) / 2 - 12:.1f}" '
                'width="36" height="24" rx="4" fill="white" '
                'stroke="#777" stroke-width="1.5"/>'
            )
            parts.append(text((ax + bx) / 2, (ay + by) / 2, f"B{road_id}", size=11, weight="700"))

        port_offset = 0.38
        for port_id in self.port_ids:
            port = self.ports[port_id]
            ix, iy = self.coords[port.intersection]
            dx, dy = DIR_DELTA[port.direction]
            px, py = point(ix + dx * port_offset, iy + dy * port_offset)
            parts.append(
                f'<rect x="{px - 18:.1f}" y="{py - 14:.1f}" width="36" height="28" '
                'rx="4" fill="#ffd166" stroke="#b77900" stroke-width="2"/>'
            )
            parts.append(text(px, py, f"P{port.id}", size=12, weight="700"))
            lx, ly = point(ix + dx * (port_offset + 0.24), iy + dy * (port_offset + 0.24))
            parts.append(text(lx, ly, port.direction, size=11))

        for int_id in self.intersection_ids:
            x, y = point(*self.coords[int_id])
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="28" '
                'fill="#3b82f6" stroke="#1e3a8a" stroke-width="3"/>'
            )
            parts.append(
                text(x, y, f"I{int_id}", size=16, weight="700").replace(
                    'font-weight="700"', 'font-weight="700" fill="white"'
                )
            )

        parts.append("</svg>")
        out_path.write_text("\n".join(parts), encoding="utf-8")
        return out_path

    def traversal_profile(
        self,
        intersection: IntersectionId,
        entry_dir: Direction,
        exit_dir: Direction,
        *,
        path_index: int = 0,
    ) -> IntersectionTraversal:
        """Return routeId, turn type, and execution time for one intersection pass."""

        self._require_intersection(intersection)
        entry = self._normalize_direction(entry_dir)
        exit_ = self._normalize_direction(exit_dir)
        key = (entry, exit_)
        if key not in ROUTE_ID_BY_ENTRY_EXIT:
            raise ValueError(
                f"invalid U-turn through intersection {intersection}: {entry} -> {exit_}"
            )
        route_id = ROUTE_ID_BY_ENTRY_EXIT[key]
        turn = TURN_BY_ROUTE_ID[route_id]
        return IntersectionTraversal(
            intersection=intersection,
            path_index=path_index,
            entry_dir=entry,
            exit_dir=exit_,
            turn=turn,
            route_id=route_id,
            execution_time=DEFAULT_TURN_DURATION[turn],
            space_durations=DEFAULT_TURN_SPACE_DURATIONS[turn],
        )

    def _build_traversals(
        self,
        path: Tuple[IntersectionId, ...],
        entrance: PortId,
        exit: PortId,
    ) -> Tuple[IntersectionTraversal, ...]:
        traversals: List[IntersectionTraversal] = []
        for idx, intersection in enumerate(path):
            if idx == 0:
                entry_dir = self.port_location(entrance)[1]
            else:
                entry_dir = self._direction_between(intersection, path[idx - 1])

            if idx == len(path) - 1:
                exit_dir = self.port_location(exit)[1]
            else:
                exit_dir = self._direction_between(intersection, path[idx + 1])

            try:
                traversals.append(
                    self.traversal_profile(
                        intersection,
                        entry_dir,
                        exit_dir,
                        path_index=idx,
                    )
                )
            except ValueError:
                return ()
        return tuple(traversals)

    def _validate_intersection_ids(self) -> None:
        ids = sorted(self.coords)
        if len(ids) != len(set(ids)):
            raise ValueError("intersection IDs must be unique")
        if any(i <= 0 for i in ids):
            raise ValueError("intersection IDs must be positive integers")
        if len(set(self.coords.values())) != len(self.coords):
            raise ValueError("intersection coordinates must be unique")

    def _build_edges(
        self, edges: Iterable[Tuple[IntersectionId, IntersectionId]]
    ) -> FrozenSet[FrozenSet[IntersectionId]]:
        seen = set()
        for a_raw, b_raw in edges:
            a, b = int(a_raw), int(b_raw)
            self._require_intersection(a)
            self._require_intersection(b)
            if a == b:
                raise ValueError(f"self-loop edge is not allowed: {a}")
            direction = self._direction_between(a, b)
            opposite = OPPOSITE[direction]
            if self.adjacency[a][direction] is not None:
                raise ValueError(f"intersection {a} already has an edge/slot at {direction}")
            if self.adjacency[b][opposite] is not None:
                raise ValueError(f"intersection {b} already has an edge/slot at {opposite}")
            self.adjacency[a][direction] = b
            self.adjacency[b][opposite] = a
            seen.add(frozenset((a, b)))
        return frozenset(seen)

    def _build_roads(
        self,
        road_ids: Optional[Mapping[Tuple[IntersectionId, IntersectionId], int]],
    ) -> Dict[int, Road]:
        normalized: Dict[FrozenSet[IntersectionId], int] = {}
        if road_ids is not None:
            for (a_raw, b_raw), road_id_raw in road_ids.items():
                a, b = int(a_raw), int(b_raw)
                key = frozenset((a, b))
                if key not in self.edges:
                    raise ValueError(f"road id {road_id_raw} references non-edge ({a}, {b})")
                road_id = int(road_id_raw)
                if road_id <= 0:
                    raise ValueError("road IDs must be positive integers")
                if road_id in normalized.values():
                    raise ValueError(f"duplicate road ID {road_id}")
                normalized[key] = road_id

        next_id = max(self.coords) + 1
        roads: Dict[int, Road] = {}
        for edge in sorted(self.edges, key=lambda item: tuple(sorted(item))):
            a, b = sorted(edge)
            road_id = normalized.get(edge)
            if road_id is None:
                while next_id in roads:
                    next_id += 1
                road_id = next_id
                next_id += 1
            roads[road_id] = Road(road_id, a, b)
        return roads

    def _build_ports(
        self,
        port_specs: Iterable[Tuple[IntersectionId, Direction]],
        *,
        preserve_order: bool,
    ) -> Dict[PortId, Port]:
        unique: List[Tuple[IntersectionId, Direction]] = []
        seen = set()
        for int_id_raw, dir_raw in port_specs:
            int_id = int(int_id_raw)
            direction = self._normalize_direction(dir_raw)
            self._require_intersection(int_id)
            key = (int_id, direction)
            if key in seen:
                continue
            if self.adjacency[int_id][direction] is not None:
                raise ValueError(
                    f"cannot place port at intersection {int_id} {direction}; "
                    "slot is occupied by an internal edge"
                )
            seen.add(key)
            unique.append(key)

        if not preserve_order:
            unique.sort(key=lambda item: (item[0], DIR_ORDER.index(item[1])))
        return {
            idx: Port(idx, int_id, direction)
            for idx, (int_id, direction) in enumerate(unique, start=1)
        }

    def _free_direction_slots(self) -> List[Tuple[IntersectionId, Direction]]:
        slots: List[Tuple[IntersectionId, Direction]] = []
        for int_id in sorted(self.coords):
            for direction in DIR_ORDER:
                if self.adjacency[int_id][direction] is None:
                    slots.append((int_id, direction))
        return slots

    def _direction_between(self, a: IntersectionId, b: IntersectionId) -> Direction:
        ax, ay = self.coords[a]
        bx, by = self.coords[b]
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy > 0:
            return "U"
        if dx == 0 and dy < 0:
            return "D"
        if dy == 0 and dx > 0:
            return "R"
        if dy == 0 and dx < 0:
            return "L"
        raise ValueError(
            f"edge ({a}, {b}) is not axis-aligned; coords {self.coords[a]} -> {self.coords[b]}"
        )

    def _require_intersection(self, intersection: IntersectionId) -> None:
        if intersection not in self.coords:
            raise KeyError(f"unknown intersection {intersection}")

    def _require_port(self, port_id: PortId) -> Port:
        if port_id not in self.ports:
            raise KeyError(f"unknown port {port_id}")
        return self.ports[port_id]

    @staticmethod
    def _normalize_direction(direction: Direction) -> Direction:
        d = str(direction).upper()
        if d not in DIRECTION_ALIASES:
            raise ValueError(f"bad direction {direction!r}; expected one of {DIR_ORDER}")
        return DIRECTION_ALIASES[d]

    @staticmethod
    def _prefix_next_choices(
        paths: Sequence[Tuple[IntersectionId, ...]]
    ) -> Dict[Tuple[IntersectionId, ...], Tuple[IntersectionId, ...]]:
        tmp: Dict[Tuple[IntersectionId, ...], set[int]] = {}
        for path in paths:
            for idx in range(len(path) - 1):
                prefix = path[: idx + 1]
                tmp.setdefault(prefix, set()).add(path[idx + 1])
        return {prefix: tuple(sorted(choices)) for prefix, choices in tmp.items()}
