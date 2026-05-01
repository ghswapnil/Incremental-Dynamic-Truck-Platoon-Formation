"""
run_validation.py — Phase 4 Master Runner

Executes the full validation suite:
    1. Baseline calculation (solo driving)
    2. Monte Carlo validation (10 runs × 100 trucks)
    3. Scale test (1,000 trucks, single run)
    4. Final thesis metrics extraction

Run with:
    python -m validation.run_validation
"""

import time
import sys
import numpy as np

from platoon.network_config import build_default_network
from platoon.travel_time import TravelTimeMatrix
from platoon.coordinator import PlatoonCoordinator
from validation.baseline import run_baseline
from validation.monte_carlo import monte_carlo_validation
from validation.truck_generator import (
    generate_synthetic_trucks,
    generate_clustered_trucks,
)


def run_scale_test(truck_count: int = 1000, seed: int = 42) -> dict:
    """
    Single-run stress test at scale to measure performance.

    Args:
        truck_count: Number of trucks to simulate.
        seed: Random seed.

    Returns:
        Dict with fuel savings, platoon rate, timing, query performance.
    """
    print(f"\n{'='*65}")
    print(f"  SCALE TEST — {truck_count} Trucks")
    print(f"{'='*65}")

    trucks = generate_synthetic_trucks(count=truck_count, seed=seed, corridor_bias=0.6)

    network = build_default_network()
    matrix = TravelTimeMatrix.build_synthetic()

    # Baseline
    baseline = run_baseline(trucks, network, matrix)
    baseline_fuel = baseline['total_fuel']
    print(f"  Baseline (solo) fuel: {baseline_fuel:.2f} L")

    # Run platooning
    network = build_default_network()  # Fresh network
    coord = PlatoonCoordinator(network, matrix)

    start = time.perf_counter()
    processed = 0

    for truck in trucks:
        coord.process_truck_arrival(truck)
        processed += 1
        if processed % 200 == 0:
            elapsed = time.perf_counter() - start
            print(f"  ... processed {processed}/{truck_count} "
                  f"({elapsed:.2f}s elapsed)")

    total_time = time.perf_counter() - start

    # Metrics
    actual_fuel = coord.calculate_total_fuel()
    savings_pct = ((baseline_fuel - actual_fuel) / baseline_fuel) * 100
    platoon_rate = coord.calculate_platoon_participation_rate()
    platoons = coord.get_platoon_summary()
    multi_platoons = [p for p in platoons if p['size'] > 1]

    query_times_ms = [t * 1000 for t in coord.query_time_log]
    avg_query_ms = float(np.mean(query_times_ms)) if query_times_ms else 0
    p95_query_ms = float(np.percentile(query_times_ms, 95)) if query_times_ms else 0
    max_query_ms = float(np.max(query_times_ms)) if query_times_ms else 0

    avg_per_truck_ms = (total_time / truck_count) * 1000

    result = {
        'truck_count': truck_count,
        'baseline_fuel': baseline_fuel,
        'actual_fuel': actual_fuel,
        'savings_pct': savings_pct,
        'platoon_rate': platoon_rate,
        'platoons_formed': len(multi_platoons),
        'total_time_seconds': total_time,
        'avg_per_truck_ms': avg_per_truck_ms,
        'avg_query_ms': avg_query_ms,
        'p95_query_ms': p95_query_ms,
        'max_query_ms': max_query_ms,
    }

    print(f"\n  Results:")
    print(f"    Total time:         {total_time:.2f}s")
    print(f"    Per-truck avg:      {avg_per_truck_ms:.2f} ms")
    print(f"    Fuel savings:       {savings_pct:.2f}%")
    print(f"    Platoon rate:       {platoon_rate:.1f}%")
    print(f"    Platoons formed:    {len(multi_platoons)}")
    print(f"    Avg query time:     {avg_query_ms:.3f} ms")
    print(f"    P95 query time:     {p95_query_ms:.3f} ms")

    # Performance target
    target_time = 10.0  # seconds
    print(f"\n  ⏱️  Target: < {target_time}s total → ", end="")
    if total_time < target_time:
        print(f"✅ PASS ({total_time:.2f}s)")
    else:
        print(f"❌ FAIL ({total_time:.2f}s)")

    print(f"{'='*65}")
    return result


def generate_thesis_results(mc_result: dict, scale_result: dict):
    """
    Print the final publishable thesis metrics.
    """
    print(f"\n\n{'█'*65}")
    print(f"{'█'*3}{'FINAL THESIS RESULTS':^59}{'█'*3}")
    print(f"{'█'*65}")

    print(f"""
  ╔═══════════════════════════════════════════════════════════╗
  ║  1. FUEL SAVINGS                                        ║
  ║     Monte Carlo (100 trucks, 10 runs):                  ║
  ║       Average: {mc_result['avg_savings_pct']:5.2f}% ± {mc_result['std_savings_pct']:.2f}%{' '*24}║
  ║       Range:   [{mc_result['min_savings_pct']:.2f}%, {mc_result['max_savings_pct']:.2f}%]{' '*26}║
  ║     Target: > 3%  →  {'✅ MET' if mc_result['avg_savings_pct'] > 3 else '❌ NOT MET'}{' '*31}║
  ╠═══════════════════════════════════════════════════════════╣
  ║  2. PLATOON FORMATION RATE                              ║
  ║     Average: {mc_result['avg_platoon_rate']:5.1f}% ± {mc_result['std_platoon_rate']:.1f}%{' '*29}║
  ║     Target: > 40%  →  {'✅ MET' if mc_result['avg_platoon_rate'] > 40 else '❌ NOT MET'}{' '*30}║
  ╠═══════════════════════════════════════════════════════════╣
  ║  3. QUERY PERFORMANCE ({scale_result['truck_count']} trucks)                     ║
  ║     Avg query:  {scale_result['avg_query_ms']:6.3f} ms{' '*32}║
  ║     P95 query:  {scale_result['p95_query_ms']:6.3f} ms{' '*32}║
  ║     Total time: {scale_result['total_time_seconds']:6.2f} s{' '*33}║
  ║     Target: avg < 50ms  →  {'✅ MET' if scale_result['avg_query_ms'] < 50 else '❌ NOT MET'}{' '*24}║
  ╠═══════════════════════════════════════════════════════════╣
  ║  4. ROBUSTNESS (Order Independence)                     ║
  ║     Std deviation: {mc_result['std_savings_pct']:.2f}%{' '*33}║
  ║     Target: < 2%  →  {'✅ MET' if mc_result['std_savings_pct'] < 2 else '❌ NOT MET'}{' '*31}║
  ╚═══════════════════════════════════════════════════════════╝""")

    # Overall verdict
    all_pass = (
        mc_result['avg_savings_pct'] > 3.0 and
        mc_result['avg_platoon_rate'] > 40.0 and
        scale_result['avg_query_ms'] < 50.0 and
        mc_result['std_savings_pct'] < 2.0
    )

    if all_pass:
        print(f"\n  🎉 ALL THESIS TARGETS MET!")
    else:
        print(f"\n  ⚠️  Some targets not met — review parameters.")

    print(f"\n{'█'*65}\n")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    print("╔" + "═"*63 + "╗")
    print("║" + " PHASE 4: VALIDATION & MONTE CARLO STRESS TEST ".center(63) + "║")
    print("╚" + "═"*63 + "╝")

    # --- Step 1: Clustered demand (primary — most realistic) ---
    # Real freight moves in waves (morning shift, afternoon shift, etc.)
    print("\n📊 Step 1: Clustered Demand Monte Carlo (PRIMARY)")
    trucks_clustered = generate_clustered_trucks(
        count=200, num_clusters=5, seed=42, cluster_spread_seconds=900
    )
    mc_primary = monte_carlo_validation(trucks_clustered, num_runs=10, seed=42)

    # --- Step 2: Uniform demand (secondary — worst case) ---
    print("\n📊 Step 2: Uniform Demand Monte Carlo (SECONDARY)")
    trucks_uniform = generate_synthetic_trucks(count=100, seed=42, corridor_bias=0.6)
    mc_uniform = monte_carlo_validation(trucks_uniform, num_runs=5, seed=100)

    # --- Step 3: Scale test (1,000 trucks, clustered) ---
    print("\n📊 Step 3: Scale Test (1,000 trucks)")
    scale_result = run_scale_test(truck_count=1000, seed=42)

    # --- Step 4: Final thesis metrics (use primary clustered result) ---
    generate_thesis_results(mc_primary, scale_result)

