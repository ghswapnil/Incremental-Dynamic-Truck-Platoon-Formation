"""
road_network.py — Core Data Structures for Temporal Road Network

Provides:
    RoadSegment    — A directed edge with an interval tree tracking truck occupancy.
    HubSpokeNetwork — The full graph with insert/query/remove operations.

Complexity:
    insert_truck_route:       O(k × log n)   — k segments in route, n existing intervals
    query_platoon_candidates: O(k × (log n + m)) — m overlapping trucks per segment
    remove_truck_route:       O(k × n)       — linear scan per segment (infrequent op)
"""

from intervaltree import IntervalTree


class RoadSegment:
    """
    A directed edge in the hub-and-spoke graph.

    The `occupancy_tree` is an IntervalTree that tracks when trucks are
    physically present on this segment.  Each interval stores:
        [entry_time, exit_time)  with data = {'truck_id': str}

    Attributes:
        start_hub (str):  Name of the origin hub.
        end_hub (str):    Name of the destination hub.
        distance_km (float): Length of this segment in kilometres.
        segment_id (str):  Unique identifier "(start_hub -> end_hub)".
        occupancy_tree (IntervalTree): Temporal occupancy index.
    """

    def __init__(self, start_hub: str, end_hub: str, distance_km: float):
        self.start_hub = start_hub
        self.end_hub = end_hub
        self.distance_km = distance_km
        self.segment_id = f"{start_hub} -> {end_hub}"
        self.occupancy_tree = IntervalTree()

    def insert_occupancy(self, entry_time: float, exit_time: float, truck_id: str):
        """
        Record that `truck_id` occupies this segment during [entry_time, exit_time).

        Args:
            entry_time: Timestamp (seconds) when the truck enters this segment.
            exit_time:  Timestamp (seconds) when the truck exits this segment.
                        Must be strictly greater than entry_time.
            truck_id:   Unique identifier for the truck.
        """
        if exit_time <= entry_time:
            raise ValueError(
                f"exit_time ({exit_time}) must be > entry_time ({entry_time}) "
                f"on segment {self.segment_id}"
            )
        self.occupancy_tree.addi(entry_time, exit_time, data={'truck_id': truck_id})

    def query_overlapping_trucks(self, entry_time: float, exit_time: float) -> set:
        """
        Find all truck IDs whose occupancy overlaps with [entry_time, exit_time).

        Returns:
            set[str]: Truck IDs that overlap in time on this segment.
        """
        if exit_time <= entry_time:
            return set()
        overlaps = self.occupancy_tree.overlap(entry_time, exit_time)
        return {interval.data['truck_id'] for interval in overlaps}

    def remove_truck(self, truck_id: str):
        """
        Remove all intervals belonging to `truck_id` from this segment.
        Used when re-assigning a truck to a different departure time.
        """
        to_remove = [
            iv for iv in self.occupancy_tree
            if iv.data['truck_id'] == truck_id
        ]
        for iv in to_remove:
            self.occupancy_tree.remove(iv)

    @property
    def truck_count(self) -> int:
        """Number of distinct trucks currently registered on this segment."""
        return len({iv.data['truck_id'] for iv in self.occupancy_tree})

    def __repr__(self):
        return (
            f"RoadSegment({self.segment_id}, "
            f"{self.distance_km}km, "
            f"{self.truck_count} trucks)"
        )


class HubSpokeNetwork:
    """
    A directed graph of hubs connected by RoadSegments, with temporal
    occupancy tracking for platoon candidate detection.

    The network supports:
        - Adding segments (edges) between hubs
        - Inserting a truck's full route (space-time footprint)
        - Querying for platoon candidates that overlap a proposed route
        - Removing a truck from all segments

    Attributes:
        segments (dict):   (start_hub, end_hub) -> RoadSegment
        hubs (set):        All hub names in the network
        adjacency (dict):  hub -> [list of neighboring hubs with outgoing edges]
    """

    def __init__(self):
        self.segments = {}       # (str, str) -> RoadSegment
        self.hubs = set()
        self.adjacency = {}      # str -> list[str]

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_segment(self, start_hub: str, end_hub: str, distance_km: float):
        """
        Add a directed segment (edge) from start_hub to end_hub.

        If the segment already exists, it is silently skipped.
        Both hubs are registered automatically.
        """
        key = (start_hub, end_hub)
        if key in self.segments:
            return  # already exists

        self.segments[key] = RoadSegment(start_hub, end_hub, distance_km)

        # Register hubs
        self.hubs.add(start_hub)
        self.hubs.add(end_hub)

        # Update adjacency
        if start_hub not in self.adjacency:
            self.adjacency[start_hub] = []
        self.adjacency[start_hub].append(end_hub)

        # Ensure end_hub has an entry (even if no outgoing edges yet)
        if end_hub not in self.adjacency:
            self.adjacency[end_hub] = []

    def get_segment(self, start_hub: str, end_hub: str) -> RoadSegment:
        """
        Retrieve the RoadSegment for a given (start_hub, end_hub) pair.

        Raises:
            KeyError: If no such segment exists.
        """
        key = (start_hub, end_hub)
        if key not in self.segments:
            raise KeyError(
                f"No segment found from '{start_hub}' to '{end_hub}'. "
                f"Available segments: {list(self.segments.keys())}"
            )
        return self.segments[key]

    def get_outgoing_segments(self, hub: str) -> list:
        """
        Return all RoadSegments that originate from the given hub.

        Returns:
            list[RoadSegment]
        """
        if hub not in self.adjacency:
            return []
        return [
            self.segments[(hub, neighbor)]
            for neighbor in self.adjacency[hub]
        ]

    # ------------------------------------------------------------------
    # Truck route operations
    # ------------------------------------------------------------------

    def insert_truck_route(self, route_path: list, truck_id: str, departure_time: float):
        """
        Record a truck's space-time footprint across the network.

        Walks the route segment by segment, computing entry/exit times
        and inserting an interval into each segment's occupancy tree.

        Args:
            route_path: List of tuples (start_hub, end_hub, travel_time_seconds).
                        Each tuple represents one segment the truck traverses.
            truck_id:   Unique identifier for the truck.
            departure_time: Timestamp (seconds) when the truck begins its journey.

        Complexity: O(k × log n) where k = len(route_path), n = existing intervals.
        """
        current_time = departure_time

        for (start_hub, end_hub, travel_time) in route_path:
            segment = self.get_segment(start_hub, end_hub)
            entry_time = current_time
            exit_time = current_time + travel_time

            segment.insert_occupancy(entry_time, exit_time, truck_id)
            current_time = exit_time

    def query_platoon_candidates(self, route_path: list, departure_time: float) -> set:
        """
        Find all trucks that spatio-temporally overlap with a proposed route.

        For each segment in the route, queries the segment's interval tree
        for trucks whose occupancy overlaps in time. Returns the union of
        all matching truck IDs across all segments.

        Args:
            route_path: List of tuples (start_hub, end_hub, travel_time_seconds).
            departure_time: Proposed departure timestamp (seconds).

        Returns:
            set[str]: Truck IDs that overlap in both space and time.

        Complexity: O(k × (log n + m)) where k = segments, n = intervals, m = matches.
        """
        candidate_trucks = set()
        current_time = departure_time

        for (start_hub, end_hub, travel_time) in route_path:
            segment = self.get_segment(start_hub, end_hub)
            entry_time = current_time
            exit_time = current_time + travel_time

            overlapping = segment.query_overlapping_trucks(entry_time, exit_time)
            candidate_trucks.update(overlapping)

            current_time = exit_time

        return candidate_trucks

    def remove_truck_route(self, truck_id: str):
        """
        Remove a truck from ALL segments in the network.

        Useful when re-assigning a truck to a different departure time
        or route during the greedy optimisation in Phase 3.

        Args:
            truck_id: The truck to remove.
        """
        for segment in self.segments.values():
            segment.remove_truck(truck_id)

    # ------------------------------------------------------------------
    # Utility / introspection
    # ------------------------------------------------------------------

    def get_all_trucks(self) -> set:
        """Return the set of all truck IDs currently in the network."""
        trucks = set()
        for segment in self.segments.values():
            for iv in segment.occupancy_tree:
                trucks.add(iv.data['truck_id'])
        return trucks

    def get_route_distance(self, route_path: list) -> float:
        """
        Calculate total distance (km) for a given route_path.

        Args:
            route_path: List of tuples (start_hub, end_hub, travel_time_seconds).

        Returns:
            Total distance in kilometres.
        """
        total = 0.0
        for (start_hub, end_hub, _travel_time) in route_path:
            segment = self.get_segment(start_hub, end_hub)
            total += segment.distance_km
        return total

    def get_shortest_path(self, origin: str, destination: str) -> list:
        """
        Find the shortest-distance path from origin to destination
        using Dijkstra's algorithm.

        Uses actual segment distances (km) as edge weights, so this
        returns the path with the minimum total distance — not just
        the fewest hops.

        Returns:
            List of hub names forming the path, e.g. ['Peenya', 'Kengeri', 'Bidadi'].
            Returns empty list if no path exists.
        """
        if origin == destination:
            return [origin]
        if origin not in self.hubs or destination not in self.hubs:
            return []

        # Dijkstra's algorithm (weighted by segment distance_km)
        import heapq
        # Priority queue: (cumulative_distance, hub_name, path_so_far)
        pq = [(0, origin, [origin])]
        visited = set()

        while pq:
            dist, current, path = heapq.heappop(pq)

            if current == destination:
                return path

            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    seg = self.get_segment(current, neighbor)
                    new_dist = dist + seg.distance_km
                    heapq.heappush(pq, (new_dist, neighbor, path + [neighbor]))

        return []

    def __repr__(self):
        return (
            f"HubSpokeNetwork("
            f"{len(self.hubs)} hubs, "
            f"{len(self.segments)} segments, "
            f"{len(self.get_all_trucks())} trucks)"
        )
