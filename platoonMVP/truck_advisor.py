"""
truck_advisor.py — Per-Truck Recommendation Engine

Wraps the PlatoonCoordinator to produce a comprehensive, actionable
travel plan for each truck that answers:

    1. When to leave  (optimal departure time)
    2. What path to take  (hub-by-hub route)
    3. At what average speed  (recommended cruising speed)
    4. Will it platoon?  (yes/no, with whom, at what position)
    5. How much fuel does it save?  (liters + percentage)
    6. When will it arrive?  (ETA + whether it meets the deadline)

Usage:
    from platoon.truck_advisor import TruckAdvisor

    advisor = TruckAdvisor()       # uses default network + synthetic matrix
    plan = advisor.submit_truck({
        'id': 'T001',
        'origin': 'Peenya',
        'destination': 'Mysuru',
        'earliest_departure': 28800,   # 8:00 AM
        'latest_departure': 36000,     # 10:00 AM
        'desired_arrival': 50400,      # 2:00 PM (optional deadline)
    })
    advisor.print_plan(plan)
"""

from platoon.network_config import build_default_network
from platoon.travel_time import TravelTimeMatrix
from platoon.coordinator import PlatoonCoordinator, DEFAULT_SPEED_KMH
from platoon.fuel_physics import BASE_FUEL_RATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seconds_to_time_str(seconds: float) -> str:
    """Convert seconds-from-midnight to 'HH:MM:SS' string."""
    seconds = seconds % 86400  # wrap around midnight
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _seconds_to_duration_str(seconds: float) -> str:
    """Convert a duration in seconds to 'Xh Ym Zs' string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h > 0:
        parts.append(f"{h}h")
    if m > 0:
        parts.append(f"{m}m")
    if s > 0 or not parts:
        parts.append(f"{s}s")
    return " ".join(parts)


def _route_path_to_hub_list(route_path: list) -> list:
    """Extract ordered list of hub names from route_path tuples."""
    if not route_path:
        return []
    hubs = [route_path[0][0]]
    for (_, end_hub, _) in route_path:
        hubs.append(end_hub)
    return hubs


# ---------------------------------------------------------------------------
# TruckPlan — the per-truck recommendation
# ---------------------------------------------------------------------------

class TruckPlan:
    """
    A comprehensive travel recommendation for a single truck.

    Attributes:
        truck_id (str):            Unique truck identifier.
        origin (str):              Starting hub.
        destination (str):         Ending hub.
        hub_route (list[str]):     Ordered hubs: ['Peenya', 'Kengeri', ...].
        departure_time (float):    Recommended departure (seconds from midnight).
        arrival_time (float):      Estimated arrival (seconds from midnight).
        total_travel_time (float): Journey duration in seconds.
        total_distance_km (float): Total route distance.
        recommended_speed (float): Cruising speed in km/h.
        is_platooning (bool):      True if the truck joins a platoon.
        platoon_size (int):        Number of trucks in the platoon.
        platoon_partners (list):   IDs of the other trucks in the platoon.
        position_in_platoon (int): 0 = leader, 1+ = follower.
        role (str):                'leader', 'follower_1', etc.
        solo_fuel (float):         Fuel if driving solo (litres).
        actual_fuel (float):       Fuel with platooning (litres).
        fuel_saved (float):        Litres saved.
        savings_pct (float):       Percentage saved.
        desired_arrival (float):   User's desired arrival time (or None).
        arrives_on_time (bool):    Whether ETA ≤ desired arrival.
        time_margin (float):       Seconds of slack (positive = early).
        segment_details (list):    Per-segment breakdown.
    """

    def __init__(self, truck_request: dict, assignment: dict,
                 hub_route: list, segment_details: list):
        self.truck_id = assignment['truck_id']
        self.origin = truck_request['origin']
        self.destination = truck_request['destination']

        # Route
        self.hub_route = hub_route
        self.segment_details = segment_details
        self.total_distance_km = assignment['total_distance_km']

        # Timing
        self.departure_time = assignment['departure_time']
        self.total_travel_time = assignment['total_travel_time']
        self.arrival_time = self.departure_time + self.total_travel_time

        # Speed
        self.recommended_speed = assignment['speed_kmh']

        # Platoon info
        partners = assignment.get('platoon_partners', [])
        self.is_platooning = len(partners) > 0
        self.platoon_partners = [p for p in partners if p != self.truck_id]
        self.platoon_size = len(partners) if partners else 1
        self.position_in_platoon = assignment['position']
        if self.position_in_platoon == 0:
            self.role = 'leader' if self.is_platooning else 'solo'
        else:
            self.role = f'follower_{self.position_in_platoon}'

        # Fuel
        self.solo_fuel = assignment['solo_fuel']
        self.actual_fuel = assignment['actual_fuel']
        self.fuel_saved = assignment['fuel_savings']
        self.savings_pct = (
            (self.fuel_saved / self.solo_fuel * 100)
            if self.solo_fuel > 0 else 0.0
        )

        # Deadline check
        self.desired_arrival = truck_request.get('desired_arrival', None)
        if self.desired_arrival is not None:
            self.arrives_on_time = self.arrival_time <= self.desired_arrival
            self.time_margin = self.desired_arrival - self.arrival_time
        else:
            self.arrives_on_time = None
            self.time_margin = None

    def to_dict(self) -> dict:
        """Serialize the plan to a plain dictionary."""
        return {
            'truck_id': self.truck_id,
            'origin': self.origin,
            'destination': self.destination,
            'hub_route': self.hub_route,
            'departure_time': _seconds_to_time_str(self.departure_time),
            'departure_time_seconds': self.departure_time,
            'arrival_time': _seconds_to_time_str(self.arrival_time),
            'arrival_time_seconds': self.arrival_time,
            'total_travel_time': _seconds_to_duration_str(self.total_travel_time),
            'total_distance_km': round(self.total_distance_km, 2),
            'recommended_speed_kmh': round(self.recommended_speed, 1),
            'is_platooning': self.is_platooning,
            'platoon_size': self.platoon_size,
            'platoon_partners': self.platoon_partners,
            'position_in_platoon': self.position_in_platoon,
            'role': self.role,
            'solo_fuel_liters': round(self.solo_fuel, 2),
            'actual_fuel_liters': round(self.actual_fuel, 2),
            'fuel_saved_liters': round(self.fuel_saved, 2),
            'savings_percentage': round(self.savings_pct, 2),
            'desired_arrival': (
                _seconds_to_time_str(self.desired_arrival)
                if self.desired_arrival else None
            ),
            'arrives_on_time': self.arrives_on_time,
            'time_margin': (
                _seconds_to_duration_str(abs(self.time_margin))
                if self.time_margin is not None else None
            ),
            'segment_details': self.segment_details,
        }


# ---------------------------------------------------------------------------
# TruckAdvisor — the main interface
# ---------------------------------------------------------------------------

class TruckAdvisor:
    """
    High-level interface that accepts truck requests and returns
    comprehensive travel plans.

    Wraps a PlatoonCoordinator internally. Each submitted truck is
    committed to the network, so subsequent trucks see the updated
    occupancy (just like the real system).

    Args:
        network: Optional pre-built HubSpokeNetwork.
        travel_matrix: Optional pre-built TravelTimeMatrix.
    """

    def __init__(self, network=None, travel_matrix=None):
        self.network = network or build_default_network()
        self.travel_matrix = travel_matrix or TravelTimeMatrix.build_synthetic()
        self.coordinator = PlatoonCoordinator(self.network, self.travel_matrix)
        self.plans = {}  # truck_id → TruckPlan

    def submit_truck(self, truck: dict) -> TruckPlan:
        """
        Submit a truck and get back a comprehensive travel plan.

        Args:
            truck: Dictionary with keys:
                - 'id' (str): Unique truck identifier
                - 'origin' (str): Starting hub
                - 'destination' (str): Ending hub
                - 'earliest_departure' (float): Seconds from midnight
                - 'latest_departure' (float): Seconds from midnight
                - 'speed_kmh' (float, optional): Preferred speed (default 60)
                - 'desired_arrival' (float, optional): Deadline in seconds
                    from midnight. If omitted, no deadline check is done.

        Returns:
            TruckPlan with the complete recommendation.
        """
        # Process through the coordinator (this commits the truck)
        self.coordinator.process_truck_arrival(truck)

        # Retrieve the stored assignment
        assignment = self.coordinator.truck_assignments[truck['id']]

        # Build the hub route
        hub_route = _route_path_to_hub_list(assignment['route_path'])

        # Build per-segment detail
        segment_details = []
        current_time = assignment['departure_time']
        for (start_hub, end_hub, travel_time) in assignment['route_path']:
            seg = self.network.get_segment(start_hub, end_hub)
            entry_time = current_time
            exit_time = current_time + travel_time
            avg_speed = (seg.distance_km / (travel_time / 3600)) if travel_time > 0 else 0

            segment_details.append({
                'from': start_hub,
                'to': end_hub,
                'distance_km': round(seg.distance_km, 2),
                'travel_time': _seconds_to_duration_str(travel_time),
                'travel_time_seconds': round(travel_time, 1),
                'entry_time': _seconds_to_time_str(entry_time),
                'exit_time': _seconds_to_time_str(exit_time),
                'avg_speed_kmh': round(avg_speed, 1),
            })
            current_time = exit_time

        plan = TruckPlan(truck, assignment, hub_route, segment_details)
        self.plans[truck['id']] = plan
        return plan

    def get_plan(self, truck_id: str) -> TruckPlan:
        """Retrieve a previously computed plan by truck ID."""
        if truck_id not in self.plans:
            raise KeyError(f"No plan found for truck '{truck_id}'")
        return self.plans[truck_id]

    def get_all_plans(self) -> list:
        """Return all plans as a list, sorted by departure time."""
        return sorted(
            self.plans.values(),
            key=lambda p: p.departure_time
        )

    # ------------------------------------------------------------------
    # Pretty-printing
    # ------------------------------------------------------------------

    @staticmethod
    def print_plan(plan: TruckPlan):
        """Print a comprehensive, human-readable travel plan."""
        W = 65
        print()
        print(f"╔{'═' * W}╗")
        print(f"║{'TRUCK TRAVEL PLAN':^{W}}║")
        print(f"╠{'═' * W}╣")

        # ── Identity ──
        print(f"║  Truck ID:        {plan.truck_id:<{W - 20}}║")
        print(f"║  Route:           {plan.origin} → {plan.destination:<{W - 22 - len(plan.origin)}}║")
        route_str = " → ".join(plan.hub_route)
        print(f"║  Path:            {route_str:<{W - 20}}║")
        print(f"║  Distance:        {plan.total_distance_km:.1f} km{' ' * (W - 28)}║")

        print(f"╠{'─' * W}╣")

        # ── Timing ──
        print(f"║  🕐 SCHEDULE{' ' * (W - 14)}║")
        dep_str = _seconds_to_time_str(plan.departure_time)
        arr_str = _seconds_to_time_str(plan.arrival_time)
        dur_str = _seconds_to_duration_str(plan.total_travel_time)
        print(f"║  Depart at:       {dep_str:<{W - 20}}║")
        print(f"║  Arrive at:       {arr_str:<{W - 20}}║")
        print(f"║  Travel time:     {dur_str:<{W - 20}}║")
        spd_str = f"{plan.recommended_speed:.1f} km/h"
        print(f"║  Avg speed:       {spd_str:<{W - 20}}║")

        # Deadline
        if plan.desired_arrival is not None:
            deadline_str = _seconds_to_time_str(plan.desired_arrival)
            margin_str = _seconds_to_duration_str(abs(plan.time_margin))
            if plan.arrives_on_time:
                status = f"✅ ON TIME ({margin_str} early)"
            else:
                status = f"❌ LATE by {margin_str}"
            print(f"║  Deadline:        {deadline_str:<{W - 20}}║")
            print(f"║  Status:          {status:<{W - 20}}║")

        print(f"╠{'─' * W}╣")

        # ── Platoon ──
        print(f"║  🚛 PLATOON INFO{' ' * (W - 18)}║")
        if plan.is_platooning:
            print(f"║  Platooning:      YES{' ' * (W - 23)}║")
            size_str = f"{plan.platoon_size} trucks"
            print(f"║  Platoon size:    {size_str:<{W - 20}}║")
            role_str = plan.role.replace('_', ' ').title()
            print(f"║  Your role:       {role_str:<{W - 20}}║")
            partners_str = ", ".join(plan.platoon_partners)
            if len(partners_str) > W - 20:
                partners_str = partners_str[:W - 23] + "..."
            print(f"║  Partners:        {partners_str:<{W - 20}}║")
        else:
            print(f"║  Platooning:      NO (driving solo){' ' * (W - 37)}║")

        print(f"╠{'─' * W}╣")

        # ── Fuel ──
        print(f"║  ⛽ FUEL ANALYSIS{' ' * (W - 19)}║")
        solo_str = f"{plan.solo_fuel:.2f} L"
        actual_str = f"{plan.actual_fuel:.2f} L"
        saved_str = f"{plan.fuel_saved:.2f} L ({plan.savings_pct:.1f}%)"
        print(f"║  Solo fuel:       {solo_str:<{W - 20}}║")
        print(f"║  Actual fuel:     {actual_str:<{W - 20}}║")
        if plan.fuel_saved > 0:
            print(f"║  Fuel SAVED:      {saved_str:<{W - 20}}║")
        else:
            print(f"║  Fuel saved:      0.00 L (no platoon){' ' * (W - 39)}║")

        print(f"╠{'─' * W}╣")

        # ── Segment breakdown ──
        print(f"║  📍 SEGMENT-BY-SEGMENT BREAKDOWN{' ' * (W - 35)}║")
        print(f"║  {'Segment':<25} {'Dist':>6} {'Time':>8} {'Enter':>8} {'Exit':>8}║")
        print(f"║  {'─' * 25} {'─' * 6} {'─' * 8} {'─' * 8} {'─' * 8}║")
        for seg in plan.segment_details:
            seg_label = f"{seg['from']}→{seg['to']}"
            if len(seg_label) > 25:
                seg_label = seg_label[:22] + "..."
            print(f"║  {seg_label:<25} "
                  f"{seg['distance_km']:>5.1f}k "
                  f"{seg['travel_time']:>8} "
                  f"{seg['entry_time']:>8} "
                  f"{seg['exit_time']:>8}║")

        print(f"╚{'═' * W}╝")
        print()

    def print_all_plans(self):
        """Print plans for all submitted trucks."""
        plans = self.get_all_plans()
        print(f"\n{'█' * 67}")
        print(f"{'█' * 3}{'ALL TRUCK PLANS (' + str(len(plans)) + ' trucks)':^61}{'█' * 3}")
        print(f"{'█' * 67}")
        for plan in plans:
            self.print_plan(plan)

    def print_summary_table(self):
        """Print a compact summary table of all truck plans."""
        plans = self.get_all_plans()
        if not plans:
            print("  No trucks submitted yet.")
            return

        W = 95
        print(f"\n{'═' * W}")
        print(f"  {'ID':<6} {'Route':<20} {'Depart':>8} {'Arrive':>8} "
              f"{'Dist':>7} {'Solo':>7} {'Actual':>7} {'Saved':>7} "
              f"{'Platoon':>8} {'On Time':>8}")
        print(f"  {'─' * 6} {'─' * 20} {'─' * 8} {'─' * 8} "
              f"{'─' * 7} {'─' * 7} {'─' * 7} {'─' * 7} "
              f"{'─' * 8} {'─' * 8}")

        for p in plans:
            route = f"{p.origin[:8]}→{p.destination[:8]}"
            dep = _seconds_to_time_str(p.departure_time)
            arr = _seconds_to_time_str(p.arrival_time)
            plat = f"{p.platoon_size}T" if p.is_platooning else "solo"
            if p.arrives_on_time is None:
                on_time = "—"
            elif p.arrives_on_time:
                on_time = "✅"
            else:
                on_time = "❌"

            print(f"  {p.truck_id:<6} {route:<20} {dep:>8} {arr:>8} "
                  f"{p.total_distance_km:>6.1f}k "
                  f"{p.solo_fuel:>6.2f}L "
                  f"{p.actual_fuel:>6.2f}L "
                  f"{p.fuel_saved:>6.2f}L "
                  f"{plat:>8} {on_time:>8}")

        print(f"{'═' * W}")

        # Totals
        total_solo = sum(p.solo_fuel for p in plans)
        total_actual = sum(p.actual_fuel for p in plans)
        total_saved = total_solo - total_actual
        pct = (total_saved / total_solo * 100) if total_solo > 0 else 0
        platooned = sum(1 for p in plans if p.is_platooning)

        print(f"\n  📊 Fleet Summary:")
        print(f"     Total trucks:      {len(plans)}")
        print(f"     Platooning:        {platooned}/{len(plans)} "
              f"({platooned / len(plans) * 100:.0f}%)")
        print(f"     Total solo fuel:   {total_solo:.2f} L")
        print(f"     Total actual fuel: {total_actual:.2f} L")
        print(f"     Total saved:       {total_saved:.2f} L ({pct:.1f}%)")
        print()


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("╔" + "═" * 63 + "╗")
    print("║" + " TRUCK ADVISOR — Per-Truck Recommendation Demo ".center(63) + "║")
    print("╚" + "═" * 63 + "╝")

    advisor = TruckAdvisor()

    # Submit a batch of sample trucks
    sample_trucks = [
        {
            'id': 'TRUCK-A',
            'origin': 'Peenya',
            'destination': 'Mysuru',
            'earliest_departure': 28800,    # 08:00 AM
            'latest_departure': 36000,      # 10:00 AM
            'speed_kmh': 60,
            'desired_arrival': 50400,       # 02:00 PM
        },
        {
            'id': 'TRUCK-B',
            'origin': 'Peenya',
            'destination': 'Mysuru',
            'earliest_departure': 29400,    # 08:10 AM
            'latest_departure': 37800,      # 10:30 AM
            'speed_kmh': 62,
            'desired_arrival': 54000,       # 03:00 PM
        },
        {
            'id': 'TRUCK-C',
            'origin': 'Peenya',
            'destination': 'Mandya',
            'earliest_departure': 30000,    # 08:20 AM
            'latest_departure': 36000,      # 10:00 AM
            'speed_kmh': 58,
            'desired_arrival': 43200,       # 12:00 PM
        },
        {
            'id': 'TRUCK-D',
            'origin': 'Kengeri',
            'destination': 'Mysuru',
            'earliest_departure': 28800,    # 08:00 AM
            'latest_departure': 32400,      # 09:00 AM
            'speed_kmh': 63,
            'desired_arrival': 39600,       # 11:00 AM — tight deadline!
        },
        {
            'id': 'TRUCK-E',
            'origin': 'Mysuru',
            'destination': 'Peenya',
            'earliest_departure': 50400,    # 02:00 PM
            'latest_departure': 57600,      # 04:00 PM
            'speed_kmh': 60,
        },
    ]

    print("\n  Submitting 5 trucks...\n")

    for truck in sample_trucks:
        plan = advisor.submit_truck(truck)
        advisor.print_plan(plan)

    # Print compact summary
    advisor.print_summary_table()
