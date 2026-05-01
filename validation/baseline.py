"""
baseline.py — No-Platooning Baseline

Computes the total fuel consumption if every truck drives solo
at its earliest departure time. This is the reference point
against which the platooning system is measured.

Usage:
    from validation.baseline import run_baseline
    baseline_fuel = run_baseline(trucks, network, travel_matrix)
"""

from platoon.fuel_physics import calculate_solo_fuel
from platoon.road_network import HubSpokeNetwork
from platoon.travel_time import TravelTimeMatrix


def run_baseline(trucks: list, network: HubSpokeNetwork,
                 travel_matrix: TravelTimeMatrix) -> dict:
    """
    Compute baseline metrics: every truck departs at earliest time, drives solo.

    Args:
        trucks: List of truck dicts with 'id', 'origin', 'destination',
                'earliest_departure', 'latest_departure'.
        network: The hub-and-spoke network (for path finding and distances).
        travel_matrix: Travel time lookup (for route computation).

    Returns:
        Dict with:
            - total_fuel: Total fuel consumed (liters)
            - per_truck: List of per-truck fuel details
            - total_distance_km: Total distance across all trucks
            - truck_count: Number of trucks processed
    """
    total_fuel = 0.0
    total_distance = 0.0
    per_truck = []

    for truck in trucks:
        hub_path = network.get_shortest_path(truck['origin'], truck['destination'])
        if len(hub_path) < 2:
            continue

        # Use earliest departure (no optimization)
        total_time, route_path = travel_matrix.get_full_route_travel_time(
            hub_path, truck['earliest_departure']
        )

        route_distance = network.get_route_distance(route_path)
        fuel = calculate_solo_fuel(route_distance)

        per_truck.append({
            'truck_id': truck['id'],
            'distance_km': route_distance,
            'fuel_liters': fuel,
            'departure_time': truck['earliest_departure'],
            'travel_time_seconds': total_time,
        })

        total_fuel += fuel
        total_distance += route_distance

    return {
        'total_fuel': total_fuel,
        'per_truck': per_truck,
        'total_distance_km': total_distance,
        'truck_count': len(per_truck),
    }
