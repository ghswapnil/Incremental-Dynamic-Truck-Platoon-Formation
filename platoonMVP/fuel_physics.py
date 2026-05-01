"""
fuel_physics.py — Fuel Consumption & Platoon Savings Model

Implements the aerodynamic draft fuel savings model for truck platooning:
  - Leader:          3% fuel savings (reduced pressure differential behind)
  - First follower: 10% fuel savings (strongest slipstream effect)
  - Other followers:  6% fuel savings (diminishing returns)

The leader's 3% benefit comes from the follower truck filling the low-pressure
wake region behind the leader, reducing aerodynamic drag (Liang et al., 2016).

The speed-change penalty models the fuel cost of the *follower* adjusting
speed to match the platoon leader's cruising speed. Leaders never pay this.

Usage:
    from platoon.fuel_physics import (
        calculate_solo_fuel,
        calculate_platoon_fuel,
        calculate_join_cost,
        calculate_net_savings,
        calculate_platoon_total_savings,
    )
"""


# ---------------------------------------------------------------------------
# Constants (from thesis proposal)
# ---------------------------------------------------------------------------

# Base fuel consumption rate for a solo truck (liters per km)
BASE_FUEL_RATE = 0.35

# Fuel savings by position in platoon
LEADER_SAVINGS = 0.03        # 3% — reduced wake drag from follower presence
FIRST_FOLLOWER_SAVINGS = 0.10  # 10% — strongest slipstream effect
OTHER_FOLLOWER_SAVINGS = 0.06  # 6% — diminishing returns for positions 2+

# Penalty for speed adjustment to join a platoon
# Units: liters per km/h of speed difference
SPEED_CHANGE_PENALTY = 0.05

# Maximum platoon size (hard constraint)
MAX_PLATOON_SIZE = 4


# ---------------------------------------------------------------------------
# Core fuel functions
# ---------------------------------------------------------------------------

def calculate_solo_fuel(distance_km: float) -> float:
    """
    Calculate fuel consumption for a solo (non-platooned) truck.

    Args:
        distance_km: Distance traveled in kilometres.

    Returns:
        Fuel consumed in litres.

    Example:
        >>> calculate_solo_fuel(100)
        35.0
    """
    return distance_km * BASE_FUEL_RATE


def calculate_platoon_fuel(distance_km: float, position_in_platoon: int) -> float:
    """
    Calculate fuel consumption for a truck in a platoon.

    Args:
        distance_km: Distance traveled while platooning, in km.
        position_in_platoon: 0 = leader, 1 = first follower, 2+ = other followers.

    Returns:
        Fuel consumed in litres (always ≤ solo fuel for the same distance).

    Example:
        >>> calculate_platoon_fuel(100, 0)   # Leader: 3% savings
        33.95
        >>> calculate_platoon_fuel(100, 1)   # First follower: 10% savings
        31.5
        >>> calculate_platoon_fuel(100, 2)   # Other follower: 6% savings
        32.9
    """
    if position_in_platoon == 0:
        savings_rate = LEADER_SAVINGS
    elif position_in_platoon == 1:
        savings_rate = FIRST_FOLLOWER_SAVINGS
    else:
        savings_rate = OTHER_FOLLOWER_SAVINGS

    return distance_km * BASE_FUEL_RATE * (1 - savings_rate)


def calculate_join_cost(speed_diff_kmh: float) -> float:
    """
    Calculate the fuel penalty for adjusting speed to join a platoon.

    Models the energy cost of accelerating or decelerating to match
    the platoon's cruising speed.

    Args:
        speed_diff_kmh: Absolute speed difference in km/h.

    Returns:
        Fuel penalty in litres (one-time cost, not per-km).

    Example:
        >>> calculate_join_cost(5)    # Small adjustment
        0.25
        >>> calculate_join_cost(10)   # Maximum allowed (from Phase 3 rules)
        0.5
    """
    return abs(speed_diff_kmh) * SPEED_CHANGE_PENALTY


def calculate_net_savings(base_fuel: float, platoon_fuel: float, join_cost: float) -> float:
    """
    Calculate net fuel savings from platooning.

    Args:
        base_fuel: Fuel that would be consumed driving solo.
        platoon_fuel: Fuel consumed while in the platoon.
        join_cost: One-time fuel penalty for speed adjustment.

    Returns:
        Net savings in litres. Positive = platooning saves fuel.
        Can be negative if join cost exceeds platoon benefits.

    Example:
        >>> calculate_net_savings(35.0, 31.5, 0.25)
        3.25
    """
    return base_fuel - platoon_fuel - join_cost


# ---------------------------------------------------------------------------
# Higher-level platoon analysis
# ---------------------------------------------------------------------------

def calculate_platoon_total_savings(distance_km: float, platoon_size: int,
                                     speed_diff_kmh: float = 0.0) -> dict:
    """
    Calculate comprehensive fuel metrics for an entire platoon.

    Assumes the platoon stays together for the full distance_km.
    Positions: 0=leader, 1=first follower, 2+=other followers.

    Args:
        distance_km: Shared platooning distance in km.
        platoon_size: Number of trucks in the platoon (1-4).
        speed_diff_kmh: Speed adjustment needed for followers (same for all).

    Returns:
        Dictionary with:
            - solo_fuel_total: Total fuel if all drove solo
            - platoon_fuel_total: Total fuel while platooning
            - join_cost_total: Total speed-adjustment penalty
            - net_savings_total: Total litres saved
            - savings_percentage: Percentage of fuel saved
            - per_truck: List of per-truck breakdown dicts

    Example:
        >>> result = calculate_platoon_total_savings(100, 3, speed_diff_kmh=5)
        >>> print(f"Savings: {result['savings_percentage']:.1f}%")
        Savings: 7.0%
    """
    if platoon_size < 1:
        raise ValueError("Platoon size must be at least 1")
    if platoon_size > MAX_PLATOON_SIZE:
        raise ValueError(
            f"Platoon size {platoon_size} exceeds maximum of {MAX_PLATOON_SIZE}"
        )

    per_truck = []
    solo_total = 0.0
    platoon_total = 0.0
    join_total = 0.0

    for position in range(platoon_size):
        solo = calculate_solo_fuel(distance_km)
        platoon = calculate_platoon_fuel(distance_km, position)

        # Only followers pay the join cost (leader sets the speed)
        join = calculate_join_cost(speed_diff_kmh) if position > 0 else 0.0

        net = calculate_net_savings(solo, platoon, join)

        per_truck.append({
            'position': position,
            'role': 'leader' if position == 0 else f'follower_{position}',
            'solo_fuel': round(solo, 4),
            'platoon_fuel': round(platoon, 4),
            'join_cost': round(join, 4),
            'net_savings': round(net, 4),
        })

        solo_total += solo
        platoon_total += platoon
        join_total += join

    net_total = solo_total - platoon_total - join_total
    pct = (net_total / solo_total * 100) if solo_total > 0 else 0.0

    return {
        'platoon_size': platoon_size,
        'distance_km': distance_km,
        'solo_fuel_total': round(solo_total, 4),
        'platoon_fuel_total': round(platoon_total, 4),
        'join_cost_total': round(join_total, 4),
        'net_savings_total': round(net_total, 4),
        'savings_percentage': round(pct, 2),
        'per_truck': per_truck,
    }


def print_platoon_breakdown(result: dict):
    """
    Pretty-print the output of calculate_platoon_total_savings().
    """
    print(f"\n{'='*60}")
    print(f"  PLATOON FUEL ANALYSIS")
    print(f"  {result['platoon_size']} trucks × {result['distance_km']} km")
    print(f"{'='*60}")

    print(f"\n  {'Position':<12} {'Solo':>8} {'Platoon':>8} {'Join':>8} {'Savings':>8}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for truck in result['per_truck']:
        print(f"  {truck['role']:<12} "
              f"{truck['solo_fuel']:>7.2f}L "
              f"{truck['platoon_fuel']:>7.2f}L "
              f"{truck['join_cost']:>7.2f}L "
              f"{truck['net_savings']:>7.2f}L")

    print(f"\n  {'TOTALS':<12} "
          f"{result['solo_fuel_total']:>7.2f}L "
          f"{result['platoon_fuel_total']:>7.2f}L "
          f"{result['join_cost_total']:>7.2f}L "
          f"{result['net_savings_total']:>7.2f}L")
    print(f"\n  💰 Net Savings: {result['savings_percentage']:.2f}%")
    print(f"{'='*60}")
