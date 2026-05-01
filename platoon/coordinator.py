"""
coordinator.py — The Greedy Platoon Decision Engine

Processes trucks as they arrive and makes optimal platoon assignments using:
  1. Departure time candidate generation (sampling the allowed time window)
  2. Spatio-temporal overlap detection (via the interval-tree-backed network)
  3. Platoon validation (size, overlap duration, speed compatibility)
  4. Net fuel savings evaluation
  5. Greedy commitment (best option wins, locked immediately)

Usage:
    from platoon.coordinator import PlatoonCoordinator
    from platoon.network_config import build_default_network
    from platoon.travel_time import TravelTimeMatrix

    network = build_default_network()
    matrix = TravelTimeMatrix.build_synthetic()
    coordinator = PlatoonCoordinator(network, matrix)

    truck = {
        'id': 'T001',
        'origin': 'Peenya',
        'destination': 'Mysuru',
        'earliest_departure': 28800,   # 8:00 AM
        'latest_departure': 36000,     # 10:00 AM
    }
    coordinator.process_truck_arrival(truck)
"""

import time as _time
from platoon.road_network import HubSpokeNetwork
from platoon.travel_time import TravelTimeMatrix
from platoon.fuel_physics import (
    calculate_solo_fuel,
    calculate_platoon_fuel,
    calculate_join_cost,
    calculate_net_savings,
    MAX_PLATOON_SIZE,
    BASE_FUEL_RATE,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Departure time sampling granularity (seconds)
CANDIDATE_INTERVAL = 300  # 5 minutes

# Minimum overlap duration to form a valid platoon (seconds)
MIN_OVERLAP_DURATION = 1200  # 20 minutes — ensures meaningful fuel benefit

# Maximum speed difference for platoon compatibility (km/h)
MAX_SPEED_DIFF = 25.0

# Default assumed speed for trucks (km/h) — used when not otherwise specified
DEFAULT_SPEED_KMH = 60.0


class PlatoonCoordinator:
    """
    Greedy online platoon assignment engine.

    Processes trucks one at a time as they arrive. For each truck:
      1. Generates candidate departure times within the allowed window
      2. For each candidate, queries the network for overlapping trucks
      3. Validates potential platoons (size, overlap, speed)
      4. Picks the option with maximum net fuel savings
      5. Commits the truck to the network (irreversible for that truck)

    Attributes:
        network (HubSpokeNetwork): The temporal road network.
        travel_matrix (TravelTimeMatrix): Time-dependent travel times.
        truck_assignments (dict): truck_id → assignment details.
        platoon_registry (dict): platoon_id → list of truck_ids.
        query_time_log (list): Per-query latency in seconds (for thesis metrics).
    """

    def __init__(self, network: HubSpokeNetwork, travel_matrix: TravelTimeMatrix):
        self.network = network
        self.travel_matrix = travel_matrix

        # truck_id → {route, departure, speed_kmh, platoon_id, position, ...}
        self.truck_assignments = {}

        # platoon_id → [truck_id, ...] ordered by join time
        # platoon_id is the truck_id of the first truck (leader)
        self.platoon_registry = {}

        # Performance tracking
        self.query_time_log = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_truck_arrival(self, truck: dict) -> dict:
        """
        Process a single truck arrival and assign it optimally.

        Args:
            truck: Dictionary with keys:
                - 'id' (str): Unique truck identifier
                - 'origin' (str): Starting hub
                - 'destination' (str): Ending hub
                - 'earliest_departure' (float): Earliest allowed departure (seconds)
                - 'latest_departure' (float): Latest allowed departure (seconds)
                - 'speed_kmh' (float, optional): Preferred speed. Default: 60.

        Returns:
            Assignment dict with departure time, route, platoon info, and savings.
        """
        truck_id = truck['id']
        speed_kmh = truck.get('speed_kmh', DEFAULT_SPEED_KMH)

        # Step 1: Find the hub-level path
        hub_path = self.network.get_shortest_path(truck['origin'], truck['destination'])
        if len(hub_path) < 2:
            raise ValueError(
                f"No valid path from {truck['origin']} to {truck['destination']}"
            )

        # Step 2: Generate candidate departure times
        candidates = self._generate_departure_candidates(
            hub_path, truck['earliest_departure'], truck['latest_departure']
        )

        # Step 3: Evaluate each candidate
        best_option = None
        best_savings = -float('inf')

        for candidate in candidates:
            t0 = _time.perf_counter()
            savings, platoon_info = self._evaluate_candidate(
                truck_id, candidate, speed_kmh
            )
            elapsed = _time.perf_counter() - t0
            self.query_time_log.append(elapsed)

            if savings > best_savings:
                best_savings = savings
                best_option = {
                    'departure_time': candidate['departure_time'],
                    'route_path': candidate['route_path'],
                    'total_travel_time': candidate['total_travel_time'],
                    'savings': savings,
                    'platoon_info': platoon_info,
                }

        # Step 4: If no candidates generated (shouldn't happen), use earliest
        if best_option is None:
            total_time, route_path = self.travel_matrix.get_full_route_travel_time(
                hub_path, truck['earliest_departure']
            )
            best_option = {
                'departure_time': truck['earliest_departure'],
                'route_path': route_path,
                'total_travel_time': total_time,
                'savings': 0.0,
                'platoon_info': None,
            }

        # Step 5: Commit
        self._commit_truck(truck_id, best_option, speed_kmh)

        return self.truck_assignments[truck_id]

    # ------------------------------------------------------------------
    # Step 2: Generate departure time candidates
    # ------------------------------------------------------------------

    def _generate_departure_candidates(self, hub_path: list,
                                        earliest: float, latest: float) -> list:
        """
        Sample the departure time window at CANDIDATE_INTERVAL granularity.

        Returns:
            List of dicts: {departure_time, route_path, total_travel_time}
        """
        candidates = []
        current = earliest

        while current <= latest:
            total_time, route_path = self.travel_matrix.get_full_route_travel_time(
                hub_path, current
            )
            candidates.append({
                'departure_time': current,
                'route_path': route_path,
                'total_travel_time': total_time,
            })
            current += CANDIDATE_INTERVAL

        # Always include the latest departure if not already sampled
        if candidates and candidates[-1]['departure_time'] < latest:
            total_time, route_path = self.travel_matrix.get_full_route_travel_time(
                hub_path, latest
            )
            candidates.append({
                'departure_time': latest,
                'route_path': route_path,
                'total_travel_time': total_time,
            })

        return candidates

    # ------------------------------------------------------------------
    # Step 3: Evaluate a candidate departure time
    # ------------------------------------------------------------------

    def _evaluate_candidate(self, truck_id: str, candidate: dict,
                             speed_kmh: float) -> tuple:
        """
        Find the best platoon opportunity for this candidate departure.

        Returns:
            (net_savings_liters, platoon_info_dict_or_None)
        """
        route_path = candidate['route_path']
        departure_time = candidate['departure_time']

        # Query the network for overlapping trucks
        overlapping_trucks = self.network.query_platoon_candidates(
            route_path, departure_time
        )

        # Remove self if already in network (shouldn't happen for new trucks)
        overlapping_trucks.discard(truck_id)

        if not overlapping_trucks:
            return 0.0, None

        # Validate each potential platoon partner
        valid_platoons = []
        for other_id in overlapping_trucks:
            platoon_info = self._validate_platoon(
                truck_id, other_id, route_path, departure_time, speed_kmh
            )
            if platoon_info['valid']:
                valid_platoons.append(platoon_info)

        if not valid_platoons:
            return 0.0, None

        # Pick the best platoon (maximum net savings)
        best = max(valid_platoons, key=lambda p: p['net_savings'])
        return best['net_savings'], best

    # ------------------------------------------------------------------
    # Step 3b: Platoon validation
    # ------------------------------------------------------------------

    def _validate_platoon(self, new_truck_id: str, existing_truck_id: str,
                           new_route: list, new_departure: float,
                           new_speed: float) -> dict:
        """
        Check if joining an existing truck/platoon is valid.

        Validates:
          - Platoon size limit (max 4)
          - Overlap duration (≥ 20 minutes)
          - Speed compatibility (≤ 25 km/h difference)
          - Net fuel savings > 0 (platooning must be cheaper than solo)

        Returns:
            Dict with 'valid', 'reason', 'net_savings', 'platoon_id', 'position', etc.
        """
        existing = self.truck_assignments.get(existing_truck_id)
        if existing is None:
            return {'valid': False, 'reason': 'truck_not_found'}

        # --- Rule 1: Platoon size limit ---
        platoon_id = existing.get('platoon_id', existing_truck_id)
        current_size = self._get_platoon_size(platoon_id)
        if current_size >= MAX_PLATOON_SIZE:
            return {'valid': False, 'reason': 'platoon_full'}

        # --- Rule 2: Overlap duration ---
        overlap_duration, overlap_distance = self._calculate_overlap(
            new_route, new_departure,
            existing['route_path'], existing['departure_time']
        )
        if overlap_duration < MIN_OVERLAP_DURATION:
            return {
                'valid': False,
                'reason': 'insufficient_overlap',
                'overlap_duration': overlap_duration,
            }

        # --- Rule 3: Speed compatibility ---
        existing_speed = existing.get('speed_kmh', DEFAULT_SPEED_KMH)
        speed_diff = abs(new_speed - existing_speed)
        if speed_diff > MAX_SPEED_DIFF:
            return {
                'valid': False,
                'reason': 'speed_incompatible',
                'speed_diff': speed_diff,
            }

        # --- Calculate fuel savings ---
        position = current_size  # New truck joins at the end
        solo_fuel = calculate_solo_fuel(overlap_distance)
        platoon_fuel = calculate_platoon_fuel(overlap_distance, position)
        join_cost = calculate_join_cost(speed_diff)
        net_savings = calculate_net_savings(solo_fuel, platoon_fuel, join_cost)

        # --- Rule 4: Platooning must actually save fuel ---
        if net_savings <= 0:
            return {
                'valid': False,
                'reason': 'no_net_savings',
                'net_savings': net_savings,
                'join_cost': join_cost,
                'overlap_distance': overlap_distance,
            }

        return {
            'valid': True,
            'net_savings': net_savings,
            'platoon_id': platoon_id,
            'partner_truck_id': existing_truck_id,
            'position': position,
            'overlap_duration': overlap_duration,
            'overlap_distance': overlap_distance,
            'speed_diff': speed_diff,
            'solo_fuel': solo_fuel,
            'platoon_fuel': platoon_fuel,
            'join_cost': join_cost,
        }

    # ------------------------------------------------------------------
    # Step 4: Commit the decision
    # ------------------------------------------------------------------

    def _commit_truck(self, truck_id: str, chosen_option: dict, speed_kmh: float):
        """
        Lock in the decision: insert into network, update registries.
        """
        departure_time = chosen_option['departure_time']
        route_path = chosen_option['route_path']
        platoon_info = chosen_option.get('platoon_info')

        # Insert into the temporal road network
        self.network.insert_truck_route(route_path, truck_id, departure_time)

        # Determine platoon assignment
        if platoon_info and platoon_info.get('valid', False):
            platoon_id = platoon_info['platoon_id']
            position = platoon_info['position']

            # Register in platoon
            if platoon_id not in self.platoon_registry:
                # Create platoon entry — leader is already assigned
                self.platoon_registry[platoon_id] = [platoon_id]
            self.platoon_registry[platoon_id].append(truck_id)

            platoon_partners = list(self.platoon_registry[platoon_id])

            # Update all existing members' platoon_partners lists
            for member_id in self.platoon_registry[platoon_id]:
                if member_id != truck_id and member_id in self.truck_assignments:
                    self.truck_assignments[member_id]['platoon_partners'] = list(
                        self.platoon_registry[platoon_id]
                    )
        else:
            platoon_id = truck_id  # Solo = own platoon
            position = 0
            platoon_partners = []

        # Calculate total route distance
        total_distance = self.network.get_route_distance(route_path)

        # Solo fuel for the full route (not just overlap)
        solo_fuel_full = calculate_solo_fuel(total_distance)

        # Actual fuel: solo for non-overlap segments + platoon for overlap
        if platoon_info and platoon_info.get('valid', False):
            overlap_dist = platoon_info['overlap_distance']
            non_overlap_dist = total_distance - overlap_dist
            actual_fuel = (
                calculate_solo_fuel(non_overlap_dist) +
                calculate_platoon_fuel(overlap_dist, position) +
                platoon_info['join_cost']
            )
        else:
            actual_fuel = solo_fuel_full

        # Store assignment
        self.truck_assignments[truck_id] = {
            'truck_id': truck_id,
            'route_path': route_path,
            'departure_time': departure_time,
            'total_travel_time': chosen_option['total_travel_time'],
            'speed_kmh': speed_kmh,
            'platoon_id': platoon_id,
            'position': position,
            'platoon_partners': platoon_partners,
            'total_distance_km': total_distance,
            'solo_fuel': solo_fuel_full,
            'actual_fuel': actual_fuel,
            'fuel_savings': solo_fuel_full - actual_fuel,
            'platoon_info': platoon_info,
        }

    # ------------------------------------------------------------------
    # Overlap calculation
    # ------------------------------------------------------------------

    def _calculate_overlap(self, route_a: list, dep_a: float,
                            route_b: list, dep_b: float) -> tuple:
        """
        Calculate the temporal and spatial overlap between two routes.

        Walks both routes segment by segment and finds segments where
        both trucks are present at the same time.

        Returns:
            (overlap_duration_seconds, overlap_distance_km)
        """
        # Build time-windows per segment for each route
        windows_a = self._build_segment_windows(route_a, dep_a)
        windows_b = self._build_segment_windows(route_b, dep_b)

        total_overlap_time = 0.0
        total_overlap_distance = 0.0

        for seg_key, (entry_a, exit_a) in windows_a.items():
            if seg_key in windows_b:
                entry_b, exit_b = windows_b[seg_key]

                # Calculate temporal overlap
                overlap_start = max(entry_a, entry_b)
                overlap_end = min(exit_a, exit_b)

                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    total_overlap_time += overlap_duration

                    # Overlap distance: proportional to time overlap on this segment
                    segment = self.network.get_segment(seg_key[0], seg_key[1])
                    seg_duration_a = exit_a - entry_a
                    if seg_duration_a > 0:
                        distance_fraction = overlap_duration / seg_duration_a
                        total_overlap_distance += segment.distance_km * distance_fraction

        return total_overlap_time, total_overlap_distance

    def _build_segment_windows(self, route_path: list, departure_time: float) -> dict:
        """
        Build a dict of (start_hub, end_hub) → (entry_time, exit_time)
        for each segment in the route.
        """
        windows = {}
        current_time = departure_time

        for (start_hub, end_hub, travel_time) in route_path:
            entry = current_time
            exit_t = current_time + travel_time
            windows[(start_hub, end_hub)] = (entry, exit_t)
            current_time = exit_t

        return windows

    # ------------------------------------------------------------------
    # Platoon helpers
    # ------------------------------------------------------------------

    def _get_platoon_size(self, platoon_id: str) -> int:
        """Get the current size of a platoon."""
        if platoon_id in self.platoon_registry:
            return len(self.platoon_registry[platoon_id])
        # If the platoon_id is a truck that hasn't been joined yet, it's solo
        if platoon_id in self.truck_assignments:
            return 1
        return 0

    # ------------------------------------------------------------------
    # Metrics for thesis (Phase 4/5)
    # ------------------------------------------------------------------

    def calculate_total_fuel(self) -> float:
        """Total actual fuel consumed across all trucks."""
        return sum(a['actual_fuel'] for a in self.truck_assignments.values())

    def calculate_total_solo_fuel(self) -> float:
        """Total fuel that would have been consumed if all drove solo."""
        return sum(a['solo_fuel'] for a in self.truck_assignments.values())

    def calculate_platoon_participation_rate(self) -> float:
        """Percentage of trucks that are in a platoon (not driving solo)."""
        if not self.truck_assignments:
            return 0.0
        platooned = sum(
            1 for a in self.truck_assignments.values()
            if len(a['platoon_partners']) > 0
        )
        return (platooned / len(self.truck_assignments)) * 100

    def calculate_fuel_savings_percentage(self) -> float:
        """System-wide fuel savings as a percentage."""
        solo = self.calculate_total_solo_fuel()
        if solo == 0:
            return 0.0
        actual = self.calculate_total_fuel()
        return ((solo - actual) / solo) * 100

    def get_platoon_summary(self) -> list:
        """
        Return a summary of all formed platoons.

        Returns:
            List of dicts with platoon_id, size, trucks, total_savings.
        """
        summary = []
        seen = set()

        for truck_id, assignment in self.truck_assignments.items():
            pid = assignment['platoon_id']
            if pid in seen:
                continue
            seen.add(pid)

            if pid in self.platoon_registry:
                members = self.platoon_registry[pid]
                total_savings = sum(
                    self.truck_assignments[m]['fuel_savings']
                    for m in members
                    if m in self.truck_assignments
                )
                summary.append({
                    'platoon_id': pid,
                    'size': len(members),
                    'trucks': list(members),
                    'total_savings_liters': round(total_savings, 2),
                })

        # Sort by size (largest first)
        summary.sort(key=lambda p: p['size'], reverse=True)
        return summary

    def print_summary(self):
        """Print a comprehensive summary of results."""
        total_trucks = len(self.truck_assignments)
        if total_trucks == 0:
            print("No trucks processed.")
            return

        solo_fuel = self.calculate_total_solo_fuel()
        actual_fuel = self.calculate_total_fuel()
        savings_pct = self.calculate_fuel_savings_percentage()
        platoon_rate = self.calculate_platoon_participation_rate()

        print(f"\n{'='*60}")
        print(f"  PLATOON COORDINATOR SUMMARY")
        print(f"{'='*60}")
        print(f"  Total trucks processed: {total_trucks}")
        print(f"  Platoon participation:  {platoon_rate:.1f}%")
        print(f"  Total solo fuel:        {solo_fuel:.2f} L")
        print(f"  Total actual fuel:      {actual_fuel:.2f} L")
        print(f"  Total fuel saved:       {solo_fuel - actual_fuel:.2f} L")
        print(f"  Savings percentage:     {savings_pct:.2f}%")

        # Platoon details
        platoons = self.get_platoon_summary()
        multi_platoons = [p for p in platoons if p['size'] > 1]
        if multi_platoons:
            print(f"\n  Platoons formed: {len(multi_platoons)}")
            for p in multi_platoons[:10]:  # Show top 10
                print(f"    {p['platoon_id']}: {p['size']} trucks "
                      f"({', '.join(p['trucks'])}) "
                      f"→ saved {p['total_savings_liters']:.2f}L")

        # Query performance
        if self.query_time_log:
            import numpy as np
            avg_ms = np.mean(self.query_time_log) * 1000
            p95_ms = np.percentile(self.query_time_log, 95) * 1000
            print(f"\n  Query performance:")
            print(f"    Average: {avg_ms:.3f} ms")
            print(f"    95th pct: {p95_ms:.3f} ms")

        print(f"{'='*60}")
