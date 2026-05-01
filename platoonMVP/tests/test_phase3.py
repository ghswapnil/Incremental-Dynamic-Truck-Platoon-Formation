"""
test_phase3.py — Validation for Phase 3: Greedy Platoon Coordinator

Tests:
    1. Single truck (solo, no platoon available)
    2. Two trucks — same route, overlapping time → platoon formed
    3. Two trucks — same route, non-overlapping time → no platoon
    4. Two trucks — different routes (no shared segments) → no platoon
    5. Speed incompatibility → no platoon despite overlap
    6. Platoon size limit (5th truck rejected)
    7. Best option selection (picks highest savings)
    8. Departure time optimization (picks better time in window)
    9. Multi-truck stream (10 trucks, verifies metrics)
   10. Integration: full pipeline with fuel & travel time

Run with:
    python -m platoon.tests.test_phase3
"""

import sys
from platoon.network_config import build_default_network
from platoon.travel_time import TravelTimeMatrix
from platoon.coordinator import PlatoonCoordinator, MIN_OVERLAP_DURATION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

passed = 0
failed = 0


def check(test_name: str, condition: bool, detail: str = ""):
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


def fresh_coordinator():
    """Create a fresh coordinator for each test."""
    network = build_default_network()
    matrix = TravelTimeMatrix.build_synthetic()
    return PlatoonCoordinator(network, matrix)


# ===========================================================================
# TEST 1: Single Truck — Solo
# ===========================================================================
def test_single_truck():
    print("\n═══ TEST 1: Single Truck (Solo) ═══")
    coord = fresh_coordinator()

    truck = {
        'id': 'T001',
        'origin': 'Peenya',
        'destination': 'Mysuru',
        'earliest_departure': 7200,    # 2:00 AM (free-flow)
        'latest_departure': 9000,      # 2:30 AM
    }
    result = coord.process_truck_arrival(truck)

    check("Truck assigned", 'T001' in coord.truck_assignments)
    check("Route path exists", len(result['route_path']) > 0)
    check("No platoon partners (solo)", len(result['platoon_partners']) == 0)
    check("Fuel savings = 0 (solo)", result['fuel_savings'] == 0.0)
    check("Position = 0 (leader/solo)", result['position'] == 0)

    # Verify it's in the network
    all_trucks = coord.network.get_all_trucks()
    check("Truck in network", 'T001' in all_trucks)


# ===========================================================================
# TEST 2: Two Trucks — Overlapping → Platoon Formed
# ===========================================================================
def test_two_trucks_platoon():
    print("\n═══ TEST 2: Two Trucks — Overlapping Time → Platoon ═══")
    coord = fresh_coordinator()

    # Truck A: departs at 2:00 AM
    truck_a = {
        'id': 'T001',
        'origin': 'Peenya',
        'destination': 'Mysuru',
        'earliest_departure': 7200,
        'latest_departure': 7200,  # Fixed time
        'speed_kmh': 60,
    }
    coord.process_truck_arrival(truck_a)

    # Truck B: departs 5 min later, same route — should overlap
    # With MIN_OVERLAP=1200s (20 min), they need enough shared time.
    # Full corridor is 153 min at free-flow — 5 min apart gives ~148 min overlap.
    truck_b = {
        'id': 'T002',
        'origin': 'Peenya',
        'destination': 'Mysuru',
        'earliest_departure': 7500,
        'latest_departure': 7500,
        'speed_kmh': 60,
    }
    result_b = coord.process_truck_arrival(truck_b)

    check("T002 has platoon partners",
          len(result_b['platoon_partners']) > 0,
          f"partners: {result_b['platoon_partners']}")

    check("T002 joined T001's platoon",
          result_b['platoon_id'] == 'T001',
          f"platoon_id: {result_b['platoon_id']}")

    check("T002 fuel savings > 0",
          result_b['fuel_savings'] > 0,
          f"savings: {result_b['fuel_savings']:.4f}L")

    check("Platoon participation = 100%",
          coord.calculate_platoon_participation_rate() == 100.0,
          f"got {coord.calculate_platoon_participation_rate():.1f}%")

    check("System fuel savings > 0",
          coord.calculate_fuel_savings_percentage() > 0,
          f"got {coord.calculate_fuel_savings_percentage():.2f}%")


# ===========================================================================
# TEST 3: Two Trucks — Non-Overlapping → No Platoon
# ===========================================================================
def test_non_overlapping():
    print("\n═══ TEST 3: Two Trucks — Non-Overlapping → No Platoon ═══")
    coord = fresh_coordinator()

    # Truck A: early morning
    truck_a = {
        'id': 'T001',
        'origin': 'Peenya',
        'destination': 'Mysuru',
        'earliest_departure': 0,       # Midnight
        'latest_departure': 0,
    }
    coord.process_truck_arrival(truck_a)

    # Truck B: much later — no overlap on any segment
    # Full corridor at 60km/h: 153km / 60 * 3600 = 9180s ≈ 2.5 hours
    # So T001 finishes by ~9180s. T002 starts at 20000 (5.5 hours later).
    truck_b = {
        'id': 'T002',
        'origin': 'Peenya',
        'destination': 'Mysuru',
        'earliest_departure': 20000,
        'latest_departure': 20000,
    }
    result_b = coord.process_truck_arrival(truck_b)

    check("T002 has NO platoon partners",
          len(result_b['platoon_partners']) == 0,
          f"partners: {result_b['platoon_partners']}")

    check("T002 fuel savings = 0",
          result_b['fuel_savings'] == 0.0)


# ===========================================================================
# TEST 4: Different Routes — No Shared Segments
# ===========================================================================
def test_different_routes():
    print("\n═══ TEST 4: Different Routes — No Shared Segments ═══")
    coord = fresh_coordinator()

    # Truck A: Peenya → Kengeri only
    truck_a = {
        'id': 'T001',
        'origin': 'Peenya',
        'destination': 'Kengeri',
        'earliest_departure': 7200,
        'latest_departure': 7200,
    }
    coord.process_truck_arrival(truck_a)

    # Truck B: Mandya → Mysuru only (completely different segment)
    truck_b = {
        'id': 'T002',
        'origin': 'Mandya',
        'destination': 'Mysuru',
        'earliest_departure': 7200,
        'latest_departure': 7200,
    }
    result_b = coord.process_truck_arrival(truck_b)

    check("No shared segments → no platoon",
          len(result_b['platoon_partners']) == 0)


# ===========================================================================
# TEST 5: Speed Incompatibility
# ===========================================================================
def test_speed_incompatibility():
    print("\n═══ TEST 5: Speed Incompatibility ═══")
    coord = fresh_coordinator()

    # Truck A at 40 km/h
    truck_a = {
        'id': 'T001',
        'origin': 'Peenya',
        'destination': 'Mysuru',
        'earliest_departure': 7200,
        'latest_departure': 7200,
        'speed_kmh': 40,
    }
    coord.process_truck_arrival(truck_a)

    # Truck B at 70 km/h — diff = 30 > MAX_SPEED_DIFF(25)
    truck_b = {
        'id': 'T002',
        'origin': 'Peenya',
        'destination': 'Mysuru',
        'earliest_departure': 7500,
        'latest_departure': 7500,
        'speed_kmh': 70,
    }
    result_b = coord.process_truck_arrival(truck_b)

    check("Speed diff > 25 km/h → no platoon",
          len(result_b['platoon_partners']) == 0,
          f"speed diff = 30 km/h, partners: {result_b['platoon_partners']}")


# ===========================================================================
# TEST 6: Platoon Size Limit
# ===========================================================================
def test_platoon_size_limit():
    print("\n═══ TEST 6: Platoon Size Limit (max 4) ═══")
    coord = fresh_coordinator()

    # Create 5 trucks with near-identical departure times
    for i in range(5):
        truck = {
            'id': f'T{i+1:03d}',
            'origin': 'Peenya',
            'destination': 'Mysuru',
            'earliest_departure': 7200 + (i * 60),  # 1 min apart
            'latest_departure': 7200 + (i * 60),
            'speed_kmh': 60,
        }
        coord.process_truck_arrival(truck)

    # Check platoon sizes
    platoons = coord.get_platoon_summary()
    max_size = max(p['size'] for p in platoons) if platoons else 0

    check("No platoon exceeds size 4",
          max_size <= 4,
          f"largest platoon has {max_size} trucks")

    # At least one platoon should have formed
    multi_platoons = [p for p in platoons if p['size'] > 1]
    check("At least one multi-truck platoon formed",
          len(multi_platoons) > 0,
          f"platoons: {platoons}")


# ===========================================================================
# TEST 7: Best Option Selection
# ===========================================================================
def test_best_option():
    print("\n═══ TEST 7: Best Option Selection ═══")
    coord = fresh_coordinator()

    # Place two trucks at different times on the same route
    truck_a = {
        'id': 'T001',
        'origin': 'Peenya',
        'destination': 'Mysuru',
        'earliest_departure': 7200,
        'latest_departure': 7200,
        'speed_kmh': 60,
    }
    coord.process_truck_arrival(truck_a)

    # New truck with a wide window that includes T001's time
    truck_c = {
        'id': 'T003',
        'origin': 'Peenya',
        'destination': 'Mysuru',
        'earliest_departure': 7000,   # Starts before T001
        'latest_departure': 8000,     # Ends after T001
        'speed_kmh': 60,
    }
    result_c = coord.process_truck_arrival(truck_c)

    check("T003 found platoon opportunity",
          result_c['fuel_savings'] > 0,
          f"savings: {result_c['fuel_savings']:.4f}L")

    # The departure time should be within the window
    check("Departure in valid window",
          7000 <= result_c['departure_time'] <= 8000,
          f"departure: {result_c['departure_time']}")


# ===========================================================================
# TEST 8: Departure Time Optimization
# ===========================================================================
def test_departure_optimization():
    print("\n═══ TEST 8: Departure Time Optimization ═══")
    coord = fresh_coordinator()

    # Place a truck at a specific time
    truck_a = {
        'id': 'T001',
        'origin': 'Peenya',
        'destination': 'Mysuru',
        'earliest_departure': 7500,
        'latest_departure': 7500,
        'speed_kmh': 60,
    }
    coord.process_truck_arrival(truck_a)

    # New truck with a flexible window
    truck_b = {
        'id': 'T002',
        'origin': 'Peenya',
        'destination': 'Mysuru',
        'earliest_departure': 7000,
        'latest_departure': 9000,
        'speed_kmh': 60,
    }
    result_b = coord.process_truck_arrival(truck_b)

    # Should pick a time close to T001 for maximum overlap
    time_diff = abs(result_b['departure_time'] - 7500)
    check("Departure optimized near T001's time",
          time_diff <= 600,  # Within 10 minutes of T001
          f"T001={7500}, T002={result_b['departure_time']}, diff={time_diff}s")

    check("Savings achieved through optimization",
          result_b['fuel_savings'] > 0,
          f"savings: {result_b['fuel_savings']:.4f}L")


# ===========================================================================
# TEST 9: Multi-Truck Stream (10 Trucks)
# ===========================================================================
def test_multi_truck_stream():
    print("\n═══ TEST 9: Multi-Truck Stream (10 Trucks) ═══")
    coord = fresh_coordinator()

    trucks = [
        {'id': f'T{i:03d}', 'origin': 'Peenya', 'destination': 'Mysuru',
         'earliest_departure': 7200 + (i * 180),  # 3 min apart
         'latest_departure': 7200 + (i * 180) + 1800,  # 30 min window
         'speed_kmh': 58 + (i % 5)}  # Speeds: 58-62 km/h
        for i in range(10)
    ]

    for truck in trucks:
        coord.process_truck_arrival(truck)

    # Metrics
    total_trucks = len(coord.truck_assignments)
    savings_pct = coord.calculate_fuel_savings_percentage()
    platoon_rate = coord.calculate_platoon_participation_rate()

    check("All 10 trucks processed", total_trucks == 10)
    check("Fuel savings > 0%", savings_pct > 0,
          f"got {savings_pct:.2f}%")
    check("Some trucks platooned", platoon_rate > 0,
          f"got {platoon_rate:.1f}%")

    # Print summary for visibility
    coord.print_summary()


# ===========================================================================
# TEST 10: Full Integration Pipeline
# ===========================================================================
def test_full_integration():
    print("\n═══ TEST 10: Full Integration Pipeline ═══")
    coord = fresh_coordinator()

    # Simulate a realistic mini-scenario:
    # 3 trucks going Peenya→Mysuru during morning rush
    # 2 trucks going Mysuru→Peenya at the same time
    trucks = [
        {'id': 'FWD1', 'origin': 'Peenya', 'destination': 'Mysuru',
         'earliest_departure': 28800, 'latest_departure': 30600, 'speed_kmh': 60},
        {'id': 'FWD2', 'origin': 'Peenya', 'destination': 'Mysuru',
         'earliest_departure': 29100, 'latest_departure': 30900, 'speed_kmh': 62},
        {'id': 'FWD3', 'origin': 'Peenya', 'destination': 'Mysuru',
         'earliest_departure': 29400, 'latest_departure': 31200, 'speed_kmh': 58},
        {'id': 'REV1', 'origin': 'Mysuru', 'destination': 'Peenya',
         'earliest_departure': 28800, 'latest_departure': 30600, 'speed_kmh': 60},
        {'id': 'REV2', 'origin': 'Mysuru', 'destination': 'Peenya',
         'earliest_departure': 29100, 'latest_departure': 30900, 'speed_kmh': 61},
    ]

    for truck in trucks:
        coord.process_truck_arrival(truck)

    # Forward trucks should platoon with each other, not with reverse trucks
    fwd1 = coord.truck_assignments['FWD1']
    fwd2 = coord.truck_assignments['FWD2']

    # At least FWD2 should be in a platoon with FWD1
    check("FWD2 joined a platoon",
          fwd2['fuel_savings'] > 0,
          f"savings: {fwd2['fuel_savings']:.4f}L")

    # Reverse trucks should NOT be in forward platoons
    rev1 = coord.truck_assignments['REV1']
    rev1_partners = rev1.get('platoon_partners', [])
    forward_in_rev = [p for p in rev1_partners if p.startswith('FWD')]
    check("Reverse trucks NOT in forward platoons",
          len(forward_in_rev) == 0,
          f"REV1 partners: {rev1_partners}")

    # Overall savings
    savings = coord.calculate_fuel_savings_percentage()
    check(f"Overall savings > 0% (got {savings:.2f}%)", savings > 0)

    coord.print_summary()


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  PHASE 3 — Greedy Platoon Coordinator Test Suite")
    print("=" * 60)

    test_single_truck()
    test_two_trucks_platoon()
    test_non_overlapping()
    test_different_routes()
    test_speed_incompatibility()
    test_platoon_size_limit()
    test_best_option()
    test_departure_optimization()
    test_multi_truck_stream()
    test_full_integration()

    # Summary
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"  RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("  🎉 All tests passed! Phase 3 is solid.")
    else:
        print("  ⚠️  Some tests failed. Review output above.")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
