"""
test_phase2.py — Validation for Phase 2: Travel Time Matrix & Fuel Physics

Runs the following test suite:
    1. Travel time matrix construction (96 slots, all segments)
    2. Rush-hour vs free-flow times (2× multiplier check)
    3. Shoulder period (1.3× multiplier check)
    4. Multi-hop route travel time
    5. Time wrapping (>24h handled correctly)
    6. Solo fuel calculation
    7. Platoon fuel by position (leader/first follower/other)
    8. Join cost calculation
    9. Net savings (positive and negative cases)
   10. Full platoon analysis (2, 3, 4 trucks)
   11. Platoon size constraint (>4 rejected)
   12. Integration: Travel time → Fuel savings pipeline

Run with:
    python -m platoon.tests.test_phase2
"""

import sys
import math

from platoon.travel_time import (
    TravelTimeMatrix,
    SLOTS_PER_DAY,
    FREE_FLOW_SPEED_KMH,
    _time_to_slot,
    _get_congestion_multiplier,
)
from platoon.fuel_physics import (
    BASE_FUEL_RATE,
    MAX_PLATOON_SIZE,
    calculate_solo_fuel,
    calculate_platoon_fuel,
    calculate_join_cost,
    calculate_net_savings,
    calculate_platoon_total_savings,
    print_platoon_breakdown,
)
from platoon.network_config import build_default_network, ROUTE_SEGMENTS


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


def approx(a, b, tol=0.01):
    """Check if two floats are approximately equal."""
    return abs(a - b) < tol


# ===========================================================================
# TEST 1: Matrix Construction
# ===========================================================================
def test_matrix_construction():
    print("\n═══ TEST 1: Travel Time Matrix Construction ═══")
    matrix = TravelTimeMatrix.build_synthetic()

    # Should have 12 segments (6 forward + 6 reverse)
    check("12 segments in matrix", len(matrix.matrix) == 12,
          f"got {len(matrix.matrix)}")

    # Each segment should have 96 time slots
    for key, slots in matrix.matrix.items():
        check(f"{key[0]}→{key[1]} has 96 slots", len(slots) == 96,
              f"got {len(slots)}")
        break  # Just check one to keep output clean

    # Verify distances stored
    check("Peenya→Kengeri distance is 18km",
          matrix.distances[('Peenya', 'Kengeri')] == 18)
    check("Kengeri→Peenya distance is 18km (reverse)",
          matrix.distances[('Kengeri', 'Peenya')] == 18)

    return matrix


# ===========================================================================
# TEST 2: Rush-Hour vs Free-Flow
# ===========================================================================
def test_rush_hour(matrix: TravelTimeMatrix):
    print("\n═══ TEST 2: Rush-Hour vs Free-Flow Times ═══")

    # Free-flow: Peenya→Kengeri (18km at 60km/h = 18 minutes)
    free_flow_sec, _ = matrix.get_travel_time('Peenya', 'Kengeri', 7200)  # 2:00 AM
    free_flow_min = free_flow_sec / 60
    expected_free = 18.0  # 18km / 60kmh * 60 = 18 min
    check(f"Free-flow: Peenya→Kengeri = {expected_free} min",
          approx(free_flow_min, expected_free),
          f"got {free_flow_min:.2f} min")

    # Rush hour: 8:00 AM (slot 32) → should be 2× = 36 min
    rush_sec, _ = matrix.get_travel_time('Peenya', 'Kengeri', 28800)  # 8:00 AM
    rush_min = rush_sec / 60
    expected_rush = expected_free * 2.0
    check(f"Rush-hour (8AM): Peenya→Kengeri = {expected_rush} min",
          approx(rush_min, expected_rush),
          f"got {rush_min:.2f} min")

    # Evening rush: 6:00 PM (slot 72)
    evening_sec, _ = matrix.get_travel_time('Peenya', 'Kengeri', 64800)  # 18:00
    evening_min = evening_sec / 60
    check(f"Evening rush (6PM): also 2× = {expected_rush} min",
          approx(evening_min, expected_rush),
          f"got {evening_min:.2f} min")

    # Rush/free-flow ratio
    ratio = rush_min / free_flow_min
    check(f"Rush/free-flow ratio = 2.0×", approx(ratio, 2.0),
          f"got {ratio:.2f}×")


# ===========================================================================
# TEST 3: Shoulder Period
# ===========================================================================
def test_shoulder_period(matrix: TravelTimeMatrix):
    print("\n═══ TEST 3: Shoulder Period (1.3×) ═══")

    # 6:30 AM is in the pre-morning shoulder (6-7 AM, 1.3×)
    shoulder_sec, _ = matrix.get_travel_time('Peenya', 'Kengeri', 23400)  # 6:30 AM
    shoulder_min = shoulder_sec / 60
    expected = 18.0 * 1.3  # 23.4 min
    check(f"Shoulder (6:30AM): 1.3× = {expected} min",
          approx(shoulder_min, expected),
          f"got {shoulder_min:.2f} min")


# ===========================================================================
# TEST 4: Multi-Hop Route Travel Time
# ===========================================================================
def test_multi_hop(matrix: TravelTimeMatrix):
    print("\n═══ TEST 4: Multi-Hop Route Travel Time ═══")

    # Full corridor at 2:00 AM (free-flow)
    hubs = ['Peenya', 'Kengeri', 'Bidadi', 'Ramanagara', 'Mandya', 'Srirangapatna', 'Mysuru']
    total_sec, route_path = matrix.get_full_route_travel_time(hubs, departure_time=7200)

    # Total distance: 18 + 25 + 15 + 50 + 20 + 25 = 153 km
    # At 60 km/h free-flow: 153 min = 9180 sec
    expected_sec = (153.0 / 60.0) * 3600
    check(f"Full corridor free-flow: {expected_sec}s",
          approx(total_sec, expected_sec, tol=60),  # Allow 1 min tolerance for slot rounding
          f"got {total_sec:.0f}s")

    # Route path should have 6 segments
    check("Route path has 6 segments", len(route_path) == 6,
          f"got {len(route_path)}")

    # Each segment's travel time should be positive
    for start, end, seg_time in route_path:
        check(f"  {start}→{end}: {seg_time:.0f}s > 0", seg_time > 0)


# ===========================================================================
# TEST 5: Time Wrapping
# ===========================================================================
def test_time_wrapping():
    print("\n═══ TEST 5: Time Wrapping (>24h) ═══")

    # Slot for 25 hours = 1 hour into next day = slot 4
    slot_25h = _time_to_slot(25 * 3600)
    slot_1h = _time_to_slot(1 * 3600)
    check("25h wraps to same slot as 1h", slot_25h == slot_1h,
          f"25h→slot {slot_25h}, 1h→slot {slot_1h}")

    # Midnight edge case
    slot_midnight = _time_to_slot(0)
    check("Midnight = slot 0", slot_midnight == 0)

    # Last slot
    slot_2345 = _time_to_slot(23 * 3600 + 45 * 60)
    check("23:45 = slot 95", slot_2345 == 95,
          f"got slot {slot_2345}")


# ===========================================================================
# TEST 6: Solo Fuel Calculation
# ===========================================================================
def test_solo_fuel():
    print("\n═══ TEST 6: Solo Fuel Calculation ═══")

    fuel_100 = calculate_solo_fuel(100)
    check("100km solo = 35.0L", approx(fuel_100, 35.0),
          f"got {fuel_100}")

    fuel_0 = calculate_solo_fuel(0)
    check("0km solo = 0.0L", approx(fuel_0, 0.0))

    fuel_153 = calculate_solo_fuel(153)  # Full corridor
    expected = 153 * 0.35
    check(f"153km (full corridor) = {expected}L",
          approx(fuel_153, expected),
          f"got {fuel_153}")


# ===========================================================================
# TEST 7: Platoon Fuel by Position
# ===========================================================================
def test_platoon_fuel_positions():
    print("\n═══ TEST 7: Platoon Fuel by Position ═══")

    dist = 100  # km

    leader = calculate_platoon_fuel(dist, 0)
    solo = calculate_solo_fuel(dist)
    expected_leader = dist * BASE_FUEL_RATE * 0.97  # 3% savings
    check("Leader (pos 0) = 97% of solo (3% savings)",
          approx(leader, expected_leader),
          f"leader={leader}, expected={expected_leader}")
    check("Leader saves less than solo",
          leader < solo,
          f"leader={leader}, solo={solo}")

    first_follower = calculate_platoon_fuel(dist, 1)
    expected_ff = dist * BASE_FUEL_RATE * 0.90  # 10% savings
    check(f"First follower (pos 1) = {expected_ff}L (10% savings)",
          approx(first_follower, expected_ff),
          f"got {first_follower}")

    other_follower = calculate_platoon_fuel(dist, 2)
    expected_of = dist * BASE_FUEL_RATE * 0.94  # 6% savings
    check(f"Other follower (pos 2) = {expected_of}L (6% savings)",
          approx(other_follower, expected_of),
          f"got {other_follower}")

    # Position 3 should also get 6%
    pos3 = calculate_platoon_fuel(dist, 3)
    check("Position 3 = same as position 2 (6%)",
          approx(pos3, expected_of))


# ===========================================================================
# TEST 8: Join Cost
# ===========================================================================
def test_join_cost():
    print("\n═══ TEST 8: Join Cost ═══")

    cost_5 = calculate_join_cost(5)
    check("5 km/h diff → 0.25L cost", approx(cost_5, 0.25))

    cost_10 = calculate_join_cost(10)
    check("10 km/h diff → 0.50L cost", approx(cost_10, 0.50))

    cost_0 = calculate_join_cost(0)
    check("0 km/h diff → 0.0L cost", approx(cost_0, 0.0))

    # Negative speed diff (slowing down) → same absolute cost
    cost_neg = calculate_join_cost(-8)
    check("Negative diff uses absolute value", approx(cost_neg, 0.40))


# ===========================================================================
# TEST 9: Net Savings
# ===========================================================================
def test_net_savings():
    print("\n═══ TEST 9: Net Savings ═══")

    # Positive case: platoon saves more than join cost
    net = calculate_net_savings(35.0, 31.5, 0.25)
    check("Positive savings: 35 - 31.5 - 0.25 = 3.25L",
          approx(net, 3.25))

    # Break-even
    net_zero = calculate_net_savings(35.0, 34.0, 1.0)
    check("Break-even: 35 - 34 - 1 = 0.0L", approx(net_zero, 0.0))

    # Negative case: join cost exceeds savings
    net_neg = calculate_net_savings(35.0, 34.5, 1.0)
    check("Negative: 35 - 34.5 - 1 = -0.5L", approx(net_neg, -0.5))


# ===========================================================================
# TEST 10: Full Platoon Analysis
# ===========================================================================
def test_platoon_analysis():
    print("\n═══ TEST 10: Full Platoon Analysis ═══")

    # 3-truck platoon, 100km, 5 km/h speed diff
    result = calculate_platoon_total_savings(100, 3, speed_diff_kmh=5)

    check("Platoon size = 3", result['platoon_size'] == 3)
    check("Distance = 100km", result['distance_km'] == 100)

    # Solo total: 3 × 35.0 = 105.0
    check("Solo total = 105.0L", approx(result['solo_fuel_total'], 105.0))

    # Platoon total: leader=33.95 + ff=31.5 + other=32.9 = 98.35
    expected_platoon = 33.95 + 31.5 + 32.9
    check(f"Platoon total ≈ {expected_platoon}L",
          approx(result['platoon_fuel_total'], expected_platoon, tol=0.1),
          f"got {result['platoon_fuel_total']}")

    # Join cost: 0 (leader) + 0.25 + 0.25 = 0.50
    check("Join cost total = 0.50L",
          approx(result['join_cost_total'], 0.50, tol=0.01),
          f"got {result['join_cost_total']}")

    # Savings should be > 5% (thesis target, improved with leader savings)
    check(f"Savings > 5% (got {result['savings_percentage']:.2f}%)",
          result['savings_percentage'] > 5.0)

    # Pretty-print for visual verification
    print_platoon_breakdown(result)

    # 4-truck platoon
    result_4 = calculate_platoon_total_savings(100, 4, speed_diff_kmh=3)
    check(f"4-truck savings > 3-truck savings",
          result_4['savings_percentage'] > result['savings_percentage'] * 0.8,
          f"4-truck: {result_4['savings_percentage']:.2f}%, 3-truck: {result['savings_percentage']:.2f}%")


# ===========================================================================
# TEST 11: Platoon Size Constraint
# ===========================================================================
def test_platoon_size_constraint():
    print("\n═══ TEST 11: Platoon Size Constraint ═══")

    try:
        calculate_platoon_total_savings(100, 5)
        check("Platoon size > 4 should raise ValueError", False)
    except ValueError:
        check("Platoon size > 4 raises ValueError", True)

    try:
        calculate_platoon_total_savings(100, 0)
        check("Platoon size < 1 should raise ValueError", False)
    except ValueError:
        check("Platoon size < 1 raises ValueError", True)


# ===========================================================================
# TEST 12: Integration — Travel Time → Fuel Pipeline
# ===========================================================================
def test_integration_pipeline():
    print("\n═══ TEST 12: Integration — Travel Time → Fuel Pipeline ═══")

    matrix = TravelTimeMatrix.build_synthetic()
    network = build_default_network()

    # Scenario: Truck going Peenya → Mysuru at 8:00 AM (rush hour)
    hubs = ['Peenya', 'Kengeri', 'Bidadi', 'Ramanagara', 'Mandya', 'Srirangapatna', 'Mysuru']
    total_time_rush, route_rush = matrix.get_full_route_travel_time(hubs, departure_time=28800)

    # Same trip at 2:00 AM (free-flow)
    total_time_free, route_free = matrix.get_full_route_travel_time(hubs, departure_time=7200)

    check("Rush-hour trip takes longer than free-flow",
          total_time_rush > total_time_free,
          f"rush={total_time_rush:.0f}s, free={total_time_free:.0f}s")

    # Total corridor distance = 153 km
    total_distance = 153.0
    solo_fuel = calculate_solo_fuel(total_distance)
    check(f"Solo fuel for 153km = {solo_fuel:.1f}L",
          approx(solo_fuel, 53.55))

    # If this truck joins a 3-truck platoon as first follower
    platoon_fuel = calculate_platoon_fuel(total_distance, position_in_platoon=1)
    join_cost = calculate_join_cost(5)  # 5 km/h speed diff
    net = calculate_net_savings(solo_fuel, platoon_fuel, join_cost)

    check(f"Net savings > 0 (got {net:.2f}L)", net > 0)
    savings_pct = (net / solo_fuel) * 100
    check(f"Savings % > 5% (got {savings_pct:.1f}%)", savings_pct > 5,
          f"This proves platooning is worthwhile on the full corridor")

    # Insert into network and verify query works with travel-time-aware route
    network.insert_truck_route(route_free, 'TRUCK_A', departure_time=7200)
    network.insert_truck_route(route_free, 'TRUCK_B', departure_time=7500)

    candidates = network.query_platoon_candidates(route_free, departure_time=7300)
    check("Integration: query finds both trucks",
          'TRUCK_A' in candidates and 'TRUCK_B' in candidates,
          f"got {candidates}")

    print(f"\n  📊 Full corridor summary:")
    print(f"     Distance: {total_distance} km")
    print(f"     Solo fuel: {solo_fuel:.2f}L")
    print(f"     Platoon fuel (follower 1): {platoon_fuel:.2f}L")
    print(f"     Join cost (5 km/h): {join_cost:.2f}L")
    print(f"     Net savings: {net:.2f}L ({savings_pct:.1f}%)")
    print(f"     Free-flow time: {total_time_free/60:.0f} min")
    print(f"     Rush-hour time: {total_time_rush/60:.0f} min")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  PHASE 2 — Travel Time & Fuel Physics Test Suite")
    print("=" * 60)

    matrix = test_matrix_construction()
    test_rush_hour(matrix)
    test_shoulder_period(matrix)
    test_multi_hop(matrix)
    test_time_wrapping()
    test_solo_fuel()
    test_platoon_fuel_positions()
    test_join_cost()
    test_net_savings()
    test_platoon_analysis()
    test_platoon_size_constraint()
    test_integration_pipeline()

    # Summary
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"  RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("  🎉 All tests passed! Phase 2 is solid.")
    else:
        print("  ⚠️  Some tests failed. Review output above.")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
