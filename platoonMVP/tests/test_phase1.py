"""
test_phase1.py — Validation & Performance Benchmark for Phase 1

Runs the following test suite:
    1. Network construction — hubs, segments, adjacency
    2. Truck insertion & platoon candidate query (overlapping trucks)
    3. No-overlap case (different times → no match)
    4. Edge cases — boundary overlaps, single-segment routes, reverse direction
    5. Remove truck and verify it disappears from queries
    6. Shortest path (BFS) on the hub graph
    7. Performance benchmark — 100 trucks, random routes & times

Run with:
    python -m platoon.tests.test_phase1
"""

import random
import time
import sys

from platoon.network_config import build_default_network, HUBS, ROUTE_SEGMENTS
from platoon.road_network import HubSpokeNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

passed = 0
failed = 0


def check(test_name: str, condition: bool, detail: str = ""):
    """Simple test assertion with PASS/FAIL output."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ PASS: {test_name}")
    else:
        failed += 1
        msg = f"  ❌ FAIL: {test_name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)


def make_route_path(hub_sequence: list, speed_kmh: float = 60.0, network: HubSpokeNetwork = None):
    """
    Convert a list of hub names into a route_path with travel times.

    Args:
        hub_sequence: e.g. ['Peenya', 'Kengeri', 'Bidadi']
        speed_kmh: Assumed constant speed.
        network: The HubSpokeNetwork to look up distances.

    Returns:
        List of (start_hub, end_hub, travel_time_seconds) tuples.
    """
    route = []
    for i in range(len(hub_sequence) - 1):
        start = hub_sequence[i]
        end = hub_sequence[i + 1]
        segment = network.get_segment(start, end)
        travel_time_s = (segment.distance_km / speed_kmh) * 3600  # km → hours → seconds
        route.append((start, end, travel_time_s))
    return route


# ===========================================================================
# TEST 1: Network Construction
# ===========================================================================
def test_network_construction():
    print("\n═══ TEST 1: Network Construction ═══")
    network = build_default_network()

    # 7 hubs
    check("Hub count is 7", len(network.hubs) == 7,
          f"got {len(network.hubs)}: {network.hubs}")

    # All expected hubs present
    for hub_name in HUBS:
        check(f"Hub '{hub_name}' exists", hub_name in network.hubs)

    # 12 directed segments (6 forward + 6 reverse)
    check("Segment count is 12", len(network.segments) == 12,
          f"got {len(network.segments)}")

    # Bidirectional check
    check("Peenya→Kengeri exists",
          ('Peenya', 'Kengeri') in network.segments)
    check("Kengeri→Peenya exists (reverse)",
          ('Kengeri', 'Peenya') in network.segments)

    # Adjacency
    outgoing = network.get_outgoing_segments('Kengeri')
    neighbor_names = {seg.end_hub for seg in outgoing}
    check("Kengeri has outgoing to Peenya and Bidadi",
          neighbor_names == {'Peenya', 'Bidadi'},
          f"got {neighbor_names}")

    return network


# ===========================================================================
# TEST 2: Insert & Query — Overlapping Trucks
# ===========================================================================
def test_insert_and_query(network: HubSpokeNetwork):
    print("\n═══ TEST 2: Insert & Query (Overlapping Trucks) ═══")

    # Truck A: Peenya → Kengeri → Bidadi, departs at t=0
    route_a = make_route_path(['Peenya', 'Kengeri', 'Bidadi'], speed_kmh=60, network=network)
    network.insert_truck_route(route_a, 'T001', departure_time=0)

    # Truck B: Peenya → Kengeri → Bidadi, departs at t=300 (5 min later, should overlap)
    route_b = make_route_path(['Peenya', 'Kengeri', 'Bidadi'], speed_kmh=60, network=network)
    network.insert_truck_route(route_b, 'T002', departure_time=300)

    # Query: a new truck on the same route at t=100 should find both T001 and T002
    route_q = make_route_path(['Peenya', 'Kengeri', 'Bidadi'], speed_kmh=60, network=network)
    candidates = network.query_platoon_candidates(route_q, departure_time=100)

    check("Query finds T001", 'T001' in candidates, f"candidates: {candidates}")
    check("Query finds T002", 'T002' in candidates, f"candidates: {candidates}")
    check("Candidate count is 2", len(candidates) == 2, f"got {len(candidates)}")

    # Verify truck count on segments
    seg_pk = network.get_segment('Peenya', 'Kengeri')
    check("Peenya→Kengeri has 2 trucks", seg_pk.truck_count == 2,
          f"got {seg_pk.truck_count}")


# ===========================================================================
# TEST 3: No-Overlap Case
# ===========================================================================
def test_no_overlap(network: HubSpokeNetwork):
    print("\n═══ TEST 3: No-Overlap Case ═══")

    # Query at a much later time — T001 and T002 should have exited by now
    route_q = make_route_path(['Peenya', 'Kengeri', 'Bidadi'], speed_kmh=60, network=network)

    # T001 departs at 0, Peenya→Kengeri at 60kmh takes 18km/60kmh*3600 = 1080s
    # Then Kengeri→Bidadi takes 25km/60kmh*3600 = 1500s
    # So T001 exits Bidadi at ~2580s, T002 at ~2880s
    # Query at t=5000 should find nobody on Peenya→Kengeri
    candidates = network.query_platoon_candidates(route_q, departure_time=5000)

    check("No overlap at t=5000", len(candidates) == 0,
          f"got {len(candidates)}: {candidates}")


# ===========================================================================
# TEST 4: Edge Cases
# ===========================================================================
def test_edge_cases(network: HubSpokeNetwork):
    print("\n═══ TEST 4: Edge Cases ═══")

    # Single-segment route
    route_single = make_route_path(['Mandya', 'Srirangapatna'], speed_kmh=60, network=network)
    network.insert_truck_route(route_single, 'T010', departure_time=1000)

    candidates = network.query_platoon_candidates(route_single, departure_time=1000)
    check("Single-segment: finds T010", 'T010' in candidates)

    # T001/T002 should NOT appear (they're on Peenya→Bidadi, not Mandya→Srirangapatna)
    check("Single-segment: does NOT find T001",
          'T001' not in candidates,
          f"candidates: {candidates}")

    # Reverse direction: Bidadi → Kengeri → Peenya
    route_rev = make_route_path(['Bidadi', 'Kengeri', 'Peenya'], speed_kmh=60, network=network)
    network.insert_truck_route(route_rev, 'T020', departure_time=0)

    # Query reverse at t=0
    candidates_rev = network.query_platoon_candidates(route_rev, departure_time=100)
    check("Reverse route: finds T020", 'T020' in candidates_rev)
    # T001/T002 are on forward segments (Peenya→Kengeri), not reverse (Kengeri→Peenya)
    check("Reverse route: does NOT find T001 (different direction)",
          'T001' not in candidates_rev,
          f"candidates: {candidates_rev}")


# ===========================================================================
# TEST 5: Remove Truck
# ===========================================================================
def test_remove_truck(network: HubSpokeNetwork):
    print("\n═══ TEST 5: Remove Truck ═══")

    # Verify T001 is currently in the network
    all_trucks = network.get_all_trucks()
    check("T001 exists before removal", 'T001' in all_trucks)

    # Remove T001
    network.remove_truck_route('T001')

    all_trucks_after = network.get_all_trucks()
    check("T001 gone after removal", 'T001' not in all_trucks_after)
    check("T002 still exists", 'T002' in all_trucks_after)


# ===========================================================================
# TEST 6: Shortest Path (BFS)
# ===========================================================================
def test_shortest_path(network: HubSpokeNetwork):
    print("\n═══ TEST 6: Shortest Path (BFS) ═══")

    path = network.get_shortest_path('Peenya', 'Mysuru')
    expected = ['Peenya', 'Kengeri', 'Bidadi', 'Ramanagara', 'Mandya', 'Srirangapatna', 'Mysuru']
    check("Peenya→Mysuru path correct", path == expected,
          f"got {path}")

    path_rev = network.get_shortest_path('Mysuru', 'Peenya')
    expected_rev = list(reversed(expected))
    check("Mysuru→Peenya path correct", path_rev == expected_rev,
          f"got {path_rev}")

    path_same = network.get_shortest_path('Mandya', 'Mandya')
    check("Same hub returns [hub]", path_same == ['Mandya'])

    path_none = network.get_shortest_path('Peenya', 'NonExistent')
    check("Non-existent hub returns []", path_none == [])


# ===========================================================================
# TEST 7: Performance Benchmark — 100 Trucks
# ===========================================================================
def test_performance_benchmark():
    print("\n═══ TEST 7: Performance Benchmark (100 Trucks) ═══")

    # Fresh network for clean benchmark
    network = build_default_network()
    hub_list = list(HUBS.keys())
    random.seed(42)  # Reproducible

    # ---- Insert 100 trucks with random routes and departure times ----
    insert_times = []
    truck_routes = {}  # Store for queries

    for i in range(100):
        # Random origin & destination (at least 2 hubs apart)
        origin = random.choice(hub_list)
        destination = random.choice([h for h in hub_list if h != origin])

        path_hubs = network.get_shortest_path(origin, destination)
        if len(path_hubs) < 2:
            continue  # Skip if no valid path

        route = make_route_path(path_hubs, speed_kmh=random.uniform(50, 80), network=network)
        dep_time = random.uniform(0, 86400)  # Random time in 24h
        truck_id = f'T{i:04d}'

        t0 = time.perf_counter()
        network.insert_truck_route(route, truck_id, dep_time)
        insert_times.append(time.perf_counter() - t0)

        truck_routes[truck_id] = (route, dep_time)

    total_trucks = len(truck_routes)
    avg_insert_ms = (sum(insert_times) / len(insert_times)) * 1000

    print(f"  Inserted {total_trucks} trucks")
    print(f"  Avg insert time: {avg_insert_ms:.3f} ms")

    # ---- Query for platoon candidates ----
    query_times = []
    candidate_counts = []

    for truck_id, (route, dep_time) in truck_routes.items():
        t0 = time.perf_counter()
        candidates = network.query_platoon_candidates(route, dep_time)
        query_times.append(time.perf_counter() - t0)
        # Exclude self from count
        candidate_counts.append(len(candidates - {truck_id}))

    avg_query_ms = (sum(query_times) / len(query_times)) * 1000
    max_query_ms = max(query_times) * 1000
    p95_query_ms = sorted(query_times)[int(len(query_times) * 0.95)] * 1000
    avg_candidates = sum(candidate_counts) / len(candidate_counts)

    print(f"\n  Query Performance ({total_trucks} queries):")
    print(f"    Average:  {avg_query_ms:.3f} ms")
    print(f"    95th pct: {p95_query_ms:.3f} ms")
    print(f"    Maximum:  {max_query_ms:.3f} ms")
    print(f"    Avg platoon candidates per truck: {avg_candidates:.1f}")

    check("Avg query time < 50ms", avg_query_ms < 50,
          f"got {avg_query_ms:.3f} ms")
    check("95th pct query time < 100ms", p95_query_ms < 100,
          f"got {p95_query_ms:.3f} ms")
    check("At least some trucks find platoon candidates",
          any(c > 0 for c in candidate_counts),
          f"max candidates found: {max(candidate_counts)}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  PHASE 1 — Temporal Road Network Test Suite")
    print("=" * 60)

    # Tests 1-6 share a network (to test cumulative state)
    network = test_network_construction()
    test_insert_and_query(network)
    test_no_overlap(network)
    test_edge_cases(network)
    test_remove_truck(network)
    test_shortest_path(network)

    # Test 7 uses a fresh network for clean benchmarking
    test_performance_benchmark()

    # Summary
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"  RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("  🎉 All tests passed! Phase 1 is solid.")
    else:
        print("  ⚠️  Some tests failed. Review output above.")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
