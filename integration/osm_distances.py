"""
osm_distances.py — Real Road Distances from OpenStreetMap

Uses OSMnx to compute shortest-path driving distances between the
7 corridor hubs (Peenya ↔ Mysuru on NH275).

Strategy: Instead of downloading the entire 100km corridor as one huge
graph, we fetch small graph patches around each hub pair's midpoint.
This is 100× faster and avoids Overpass API timeouts.

Usage:
    from integration.osm_distances import fetch_real_distances
    distances = fetch_real_distances()

    from integration.osm_distances import build_osm_network
    network = build_osm_network()

Caching:
    Results cached to 'cache/osm_distances.json'.
"""

import os
import json
import osmnx as ox
import networkx as nx
from geopy.distance import geodesic

from platoon.network_config import HUBS, ROUTE_SEGMENTS
from platoon.road_network import HubSpokeNetwork


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
CACHE_FILE = os.path.join(CACHE_DIR, 'osm_distances.json')

# Padding factor: fetch graph within (straight_line_dist × PADDING) meters
# of each hub pair's midpoint. 1.5 = 50% buffer for road curvature.
PADDING_FACTOR = 1.5


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def fetch_real_distances(use_cache: bool = True, verbose: bool = True) -> dict:
    """
    Fetch real driving distances between all hub pairs from OpenStreetMap.

    Uses a per-segment approach: for each hub pair, fetches a small
    road graph around the midpoint and computes the shortest path.

    Args:
        use_cache: If True, try loading from cache first.
        verbose: Print progress.

    Returns:
        Dict with keys (start_hub, end_hub) → distance_km.
    """
    # Try cache first
    if use_cache and os.path.exists(CACHE_FILE):
        if verbose:
            print(f"  📦 Loading cached OSM distances from {CACHE_FILE}")
        with open(CACHE_FILE, 'r') as f:
            cached = json.load(f)
        return {tuple(k.split('|')): v for k, v in cached.items()}

    if verbose:
        print("  🌍 Fetching road distances from OpenStreetMap...")
        print(f"     Strategy: per-segment graph patches (fast)")

    distances = {}

    for start_hub, end_hub, hardcoded_km in ROUTE_SEGMENTS:
        start_coords = (HUBS[start_hub]['lat'], HUBS[start_hub]['lon'])
        end_coords = (HUBS[end_hub]['lat'], HUBS[end_hub]['lon'])

        # Midpoint between the two hubs
        mid_lat = (start_coords[0] + end_coords[0]) / 2
        mid_lon = (start_coords[1] + end_coords[1]) / 2

        # Straight-line distance for graph radius
        straight_line_m = geodesic(start_coords, end_coords).meters
        radius_m = straight_line_m * PADDING_FACTOR

        if verbose:
            print(f"\n     {start_hub} → {end_hub}:")
            print(f"       Fetching graph around ({mid_lat:.4f}, {mid_lon:.4f}), "
                  f"radius={radius_m/1000:.1f}km...")

        try:
            # Fetch a driving graph around the midpoint
            G = ox.graph_from_point(
                (mid_lat, mid_lon),
                dist=radius_m,
                network_type='drive',
            )

            if verbose:
                print(f"       Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

            # Find nearest nodes to hub coordinates
            start_node = ox.nearest_nodes(G, start_coords[1], start_coords[0])
            end_node = ox.nearest_nodes(G, end_coords[1], end_coords[0])

            # Shortest path by distance (meters)
            length_m = nx.shortest_path_length(
                G, start_node, end_node, weight='length'
            )
            distance_km = round(length_m / 1000, 2)

        except Exception as e:
            if verbose:
                print(f"       ⚠️  Failed: {e}")
                print(f"       Using hardcoded: {hardcoded_km} km")
            distance_km = hardcoded_km

        distances[(start_hub, end_hub)] = distance_km

        if verbose:
            straight_km = straight_line_m / 1000
            diff_pct = ((distance_km - hardcoded_km) / hardcoded_km) * 100
            print(f"       OSM: {distance_km:.2f} km | "
                  f"Hardcoded: {hardcoded_km} km | "
                  f"Straight: {straight_km:.1f} km | "
                  f"Diff: {diff_pct:+.1f}%")

    # Save to cache
    _save_cache(distances)
    if verbose:
        print(f"\n  💾 Cached to {CACHE_FILE}")

    return distances


def build_osm_network(use_cache: bool = True, verbose: bool = True) -> HubSpokeNetwork:
    """
    Build a HubSpokeNetwork using real OSM driving distances.

    Returns:
        HubSpokeNetwork with 7 hubs and 12 directed segments.
    """
    distances = fetch_real_distances(use_cache=use_cache, verbose=verbose)

    network = HubSpokeNetwork()

    for start_hub, end_hub, _hardcoded_km in ROUTE_SEGMENTS:
        real_km = distances.get((start_hub, end_hub), _hardcoded_km)
        network.add_segment(start_hub, end_hub, real_km)
        network.add_segment(end_hub, start_hub, real_km)

    if verbose:
        total_real = sum(distances.values())
        total_hardcoded = sum(d for _, _, d in ROUTE_SEGMENTS)
        print(f"\n  📊 Total corridor distance:")
        print(f"     OSM:       {total_real:.2f} km")
        print(f"     Hardcoded: {total_hardcoded:.1f} km")
        print(f"     Diff:      {total_real - total_hardcoded:+.2f} km "
              f"({((total_real - total_hardcoded)/total_hardcoded)*100:+.1f}%)")

    return network


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _save_cache(distances: dict):
    """Save distances to JSON cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    serializable = {f"{k[0]}|{k[1]}": v for k, v in distances.items()}
    with open(CACHE_FILE, 'w') as f:
        json.dump(serializable, f, indent=2)


def clear_cache():
    """Delete the cached distances file."""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print(f"  🗑️  Deleted cache: {CACHE_FILE}")
    else:
        print(f"  ℹ️  No cache to delete.")


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_distances(verbose: bool = True) -> dict:
    """Compare OSM distances with hardcoded values."""
    distances = fetch_real_distances(use_cache=True, verbose=False)

    comparisons = []
    for start_hub, end_hub, hardcoded_km in ROUTE_SEGMENTS:
        real_km = distances.get((start_hub, end_hub), hardcoded_km)
        diff_km = real_km - hardcoded_km
        diff_pct = (diff_km / hardcoded_km) * 100

        comparisons.append({
            'segment': f"{start_hub} → {end_hub}",
            'hardcoded_km': hardcoded_km,
            'osm_km': real_km,
            'diff_km': round(diff_km, 2),
            'diff_pct': round(diff_pct, 1),
        })

    if verbose:
        print(f"\n{'='*65}")
        print(f"  DISTANCE COMPARISON: Hardcoded vs OSM")
        print(f"{'='*65}")
        print(f"  {'Segment':<30} {'Hardcoded':>10} {'OSM':>10} {'Diff':>10}")
        print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10}")

        for c in comparisons:
            print(f"  {c['segment']:<30} "
                  f"{c['hardcoded_km']:>8.1f}km "
                  f"{c['osm_km']:>8.2f}km "
                  f"{c['diff_pct']:>+8.1f}%")

        total_h = sum(c['hardcoded_km'] for c in comparisons)
        total_o = sum(c['osm_km'] for c in comparisons)
        print(f"  {'TOTAL':<30} "
              f"{total_h:>8.1f}km "
              f"{total_o:>8.2f}km "
              f"{((total_o-total_h)/total_h)*100:>+8.1f}%")
        print(f"{'='*65}")

    return {'comparisons': comparisons}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    if '--clear-cache' in sys.argv:
        clear_cache()
    elif '--compare' in sys.argv:
        compare_distances()
    else:
        print("╔" + "═"*63 + "╗")
        print("║" + " OSM DISTANCE FETCHER ".center(63) + "║")
        print("╚" + "═"*63 + "╝")
        distances = fetch_real_distances(use_cache=('--no-cache' not in sys.argv))
        print(f"\n  ✅ Fetched distances for {len(distances)} segments")
