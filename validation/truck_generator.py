"""
truck_generator.py — Synthetic Truck Demand Generator

Generates realistic truck requests for simulation and stress testing.

Supports:
    - Configurable count, time distributions, speed ranges
    - Corridor-biased origins/destinations (most trucks go Peenya↔Mysuru)
    - Reproducible output via seed parameter

Usage:
    from validation.truck_generator import generate_synthetic_trucks
    trucks = generate_synthetic_trucks(count=1000, seed=42)
"""

import random
from platoon.network_config import HUBS


def generate_synthetic_trucks(count: int = 100,
                               seed: int = None,
                               time_window_seconds: int = 7200,
                               speed_range: tuple = (55, 70),
                               corridor_bias: float = 0.4) -> list:
    """
    Generate a batch of synthetic truck requests.

    Args:
        count: Number of trucks to generate.
        seed: Random seed for reproducibility. None = random.
        time_window_seconds: Departure flexibility window (default 2 hours).
        speed_range: (min_speed_kmh, max_speed_kmh) tuple.
        corridor_bias: Probability [0-1] of generating a full-corridor trip
                       (Peenya→Mysuru or Mysuru→Peenya). Higher = more platoon
                       opportunities. Default 0.4 = 40%.

    Returns:
        List of truck dicts, each with:
            - id, origin, destination
            - earliest_departure, latest_departure (seconds from midnight)
            - speed_kmh
    """
    if seed is not None:
        random.seed(seed)

    hub_names = list(HUBS.keys())
    trucks = []

    for i in range(count):
        # Decide if this is a full-corridor or random trip
        if random.random() < corridor_bias:
            # Full corridor (biased for more platooning)
            if random.random() < 0.5:
                origin, destination = 'Peenya', 'Mysuru'
            else:
                origin, destination = 'Mysuru', 'Peenya'
        else:
            # Random origin/destination
            origin = random.choice(hub_names)
            destination = random.choice([h for h in hub_names if h != origin])

        # Random departure window within 24 hours
        earliest = random.randint(0, 86400 - time_window_seconds)
        latest = earliest + time_window_seconds

        # Random preferred speed
        speed = random.uniform(speed_range[0], speed_range[1])

        trucks.append({
            'id': f'T{i:04d}',
            'origin': origin,
            'destination': destination,
            'earliest_departure': earliest,
            'latest_departure': latest,
            'speed_kmh': round(speed, 1),
        })

    return trucks


def generate_clustered_trucks(count: int = 100,
                                num_clusters: int = 5,
                                seed: int = None,
                                cluster_spread_seconds: int = 600) -> list:
    """
    Generate trucks in temporal clusters (simulating convoy patterns).

    More realistic: trucks tend to depart in waves (morning shift,
    afternoon shift, etc.)

    Args:
        count: Total number of trucks.
        num_clusters: Number of departure time clusters.
        seed: Random seed.
        cluster_spread_seconds: How spread out each cluster is (±seconds).

    Returns:
        List of truck dicts.
    """
    if seed is not None:
        random.seed(seed)

    hub_names = list(HUBS.keys())
    trucks = []

    # Generate cluster centers spread across the day
    cluster_centers = sorted(random.sample(range(3600, 82800), num_clusters))
    trucks_per_cluster = count // num_clusters
    remainder = count % num_clusters

    truck_idx = 0
    for ci, center in enumerate(cluster_centers):
        n = trucks_per_cluster + (1 if ci < remainder else 0)

        for _ in range(n):
            # Random offset within the cluster
            offset = random.randint(-cluster_spread_seconds, cluster_spread_seconds)
            earliest = max(0, center + offset)
            latest = earliest + 3600  # 1-hour window

            # Full-corridor bias within clusters
            if random.random() < 0.6:
                origin, destination = ('Peenya', 'Mysuru') if random.random() < 0.5 else ('Mysuru', 'Peenya')
            else:
                origin = random.choice(hub_names)
                destination = random.choice([h for h in hub_names if h != origin])

            speed = random.uniform(57, 65)

            trucks.append({
                'id': f'T{truck_idx:04d}',
                'origin': origin,
                'destination': destination,
                'earliest_departure': earliest,
                'latest_departure': latest,
                'speed_kmh': round(speed, 1),
            })
            truck_idx += 1

    return trucks
