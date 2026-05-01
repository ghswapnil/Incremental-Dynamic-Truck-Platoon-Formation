"""
travel_time.py — Time-Dependent Travel Time Matrix

Provides a 96-slot (15-minute intervals × 24 hours) travel time lookup
for each segment in the hub-and-spoke network.

Two modes:
  1. Synthetic: Rush-hour patterns (7-10 AM, 5-8 PM) with configurable multipliers
  2. Real data: See `integration/traffic_data.py` which builds a real travel time
     matrix from empirical NH275 congestion data or a Kaggle CSV file.

Usage:
    from platoon.travel_time import TravelTimeMatrix
    matrix = TravelTimeMatrix.build_synthetic()
"""

import math
from platoon.network_config import ROUTE_SEGMENTS, build_default_network


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base free-flow speed on the highway corridor (km/h)
FREE_FLOW_SPEED_KMH = 60.0

# Rush-hour definitions: (start_hour, end_hour, congestion_multiplier)
# These multiply the base travel time during congested periods
RUSH_HOUR_WINDOWS = [
    (7, 10, 2.0),    # Morning rush: 7:00 AM – 10:00 AM, 2× slower
    (17, 20, 2.0),   # Evening rush: 5:00 PM – 8:00 PM, 2× slower
]

# Moderate congestion shoulders around rush hours
SHOULDER_WINDOWS = [
    (6, 7, 1.3),     # Pre-morning: 6:00 – 7:00 AM, 1.3× slower
    (10, 11, 1.3),   # Post-morning: 10:00 – 11:00 AM
    (16, 17, 1.3),   # Pre-evening: 4:00 – 5:00 PM
    (20, 21, 1.3),   # Post-evening: 8:00 – 9:00 PM
]

# Number of time slots per day (every 15 minutes)
SLOTS_PER_DAY = 96
SLOT_DURATION_SECONDS = 900  # 15 minutes


class TravelTimeMatrix:
    """
    Time-dependent travel time lookup for the hub-and-spoke network.

    Stores a dictionary:
        matrix[(start_hub, end_hub)][slot_index] = travel_time_minutes

    where slot_index ∈ [0, 95] maps to 15-minute intervals of the day.

    Attributes:
        matrix (dict): Nested lookup of travel times.
        distances (dict): (start_hub, end_hub) -> distance_km for reference.
    """

    def __init__(self):
        self.matrix = {}      # (str, str) -> {int: float}  (slot → minutes)
        self.distances = {}   # (str, str) -> float (km)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def build_synthetic(cls, network=None):
        """
        Build a synthetic travel time matrix with rush-hour patterns.

        Uses the segments from ROUTE_SEGMENTS (both forward and reverse).
        Free-flow speed: 60 km/h, with 2× multiplier during rush hours
        and 1.3× during shoulder periods.

        Args:
            network: Optional HubSpokeNetwork to read distances from.
                     If None, uses ROUTE_SEGMENTS directly.

        Returns:
            TravelTimeMatrix with all 96 time slots populated.
        """
        instance = cls()

        # Collect all segment pairs (forward + reverse)
        segment_pairs = []
        for start, end, dist in ROUTE_SEGMENTS:
            segment_pairs.append((start, end, dist))
            segment_pairs.append((end, start, dist))  # Reverse

        for start_hub, end_hub, distance_km in segment_pairs:
            key = (start_hub, end_hub)
            instance.distances[key] = distance_km
            instance.matrix[key] = {}

            # Base travel time at free-flow speed
            base_minutes = (distance_km / FREE_FLOW_SPEED_KMH) * 60

            for slot in range(SLOTS_PER_DAY):
                hour = slot * 15 / 60  # Convert slot to fractional hour
                multiplier = _get_congestion_multiplier(hour)
                instance.matrix[key][slot] = base_minutes * multiplier

        return instance

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_travel_time(self, start_hub: str, end_hub: str, departure_time: float):
        """
        Look up travel time for a segment at a given departure time.

        Args:
            start_hub: Origin hub name.
            end_hub: Destination hub name.
            departure_time: Departure timestamp in seconds from midnight
                            (0 = midnight, 28800 = 8:00 AM, etc.)

        Returns:
            Tuple of (travel_time_seconds, route_path)
            where route_path = [(start_hub, end_hub, travel_time_seconds)]

        Raises:
            KeyError: If the segment is not in the matrix.
        """
        key = (start_hub, end_hub)
        if key not in self.matrix:
            raise KeyError(
                f"No travel time data for segment '{start_hub}' → '{end_hub}'. "
                f"Available: {list(self.matrix.keys())}"
            )

        slot = _time_to_slot(departure_time)
        travel_minutes = self.matrix[key][slot]
        travel_seconds = travel_minutes * 60

        route_path = [(start_hub, end_hub, travel_seconds)]
        return travel_seconds, route_path

    def get_full_route_travel_time(self, hub_sequence: list, departure_time: float):
        """
        Compute total travel time and route_path for a multi-hop journey.

        Args:
            hub_sequence: e.g. ['Peenya', 'Kengeri', 'Bidadi', 'Ramanagara']
            departure_time: Departure timestamp in seconds from midnight.

        Returns:
            Tuple of (total_travel_time_seconds, route_path)
            where route_path is a list of (start_hub, end_hub, segment_travel_seconds).
        """
        route_path = []
        current_time = departure_time
        total_time = 0.0

        for i in range(len(hub_sequence) - 1):
            start = hub_sequence[i]
            end = hub_sequence[i + 1]

            travel_seconds, _ = self.get_travel_time(start, end, current_time)
            route_path.append((start, end, travel_seconds))

            current_time += travel_seconds
            total_time += travel_seconds

        return total_time, route_path

    def get_slot_label(self, slot: int) -> str:
        """Convert a slot index (0-95) to a time label like '08:15'."""
        total_minutes = slot * 15
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours:02d}:{minutes:02d}"

    def get_travel_time_profile(self, start_hub: str, end_hub: str) -> list:
        """
        Return the full 24-hour travel time profile for a segment.

        Returns:
            List of (time_label, travel_minutes) tuples for all 96 slots.
        """
        key = (start_hub, end_hub)
        if key not in self.matrix:
            raise KeyError(f"No data for {start_hub} → {end_hub}")

        return [
            (self.get_slot_label(slot), self.matrix[key][slot])
            for slot in range(SLOTS_PER_DAY)
        ]

    def __repr__(self):
        return f"TravelTimeMatrix({len(self.matrix)} segments, {SLOTS_PER_DAY} slots/day)"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _time_to_slot(departure_time: float) -> int:
    """
    Convert a departure time (seconds from midnight) to a slot index [0, 95].

    Times beyond 24h are wrapped around (modulo).
    """
    seconds_in_day = departure_time % 86400
    slot = int(seconds_in_day // SLOT_DURATION_SECONDS)
    return min(slot, SLOTS_PER_DAY - 1)  # Clamp to valid range


def _get_congestion_multiplier(hour: float) -> float:
    """
    Return the congestion multiplier for a fractional hour of the day.

    Checks rush-hour windows first, then shoulder windows.
    Returns 1.0 (free-flow) if no window matches.
    """
    for start_h, end_h, multiplier in RUSH_HOUR_WINDOWS:
        if start_h <= hour < end_h:
            return multiplier

    for start_h, end_h, multiplier in SHOULDER_WINDOWS:
        if start_h <= hour < end_h:
            return multiplier

    return 1.0  # Free-flow
