"""
monte_carlo.py — Monte Carlo Arrival Order Validation

Shuffles the truck arrival order across multiple runs and measures
the consistency of fuel savings. This is the key defense against
the "order-dependency" critique of greedy algorithms.

Defense argument:
    "While the algorithm is greedy and order-dependent, Monte Carlo
     analysis over N random arrival sequences shows savings are
     statistically consistent at X% ± Y%, proving the approach is
     robust to arrival timing variations."

Usage:
    from validation.monte_carlo import monte_carlo_validation
    results = monte_carlo_validation(trucks, num_runs=10, seed=42)
"""

import random
import time
import numpy as np

from platoon.network_config import build_default_network
from platoon.travel_time import TravelTimeMatrix
from platoon.coordinator import PlatoonCoordinator
from validation.baseline import run_baseline


def monte_carlo_validation(trucks: list, num_runs: int = 10,
                            seed: int = 42, verbose: bool = True) -> dict:
    """
    Run the platooning simulation with shuffled arrival orders.

    For each run:
      1. Shuffle the truck list (different arrival order)
      2. Create a fresh coordinator
      3. Process all trucks
      4. Record fuel savings, platoon rate, query performance

    Args:
        trucks: List of truck dicts.
        num_runs: Number of Monte Carlo iterations.
        seed: Base random seed (each run uses seed + run_index).
        verbose: Print progress and per-run results.

    Returns:
        Dict with:
            - baseline_fuel: Reference solo fuel total
            - runs: List of per-run result dicts
            - avg_savings_pct, std_savings_pct: Mean ± std of savings
            - avg_platoon_rate, std_platoon_rate: Mean ± std of participation
            - avg_query_ms, p95_query_ms: Query latency stats
            - total_time_seconds: Wall-clock time for all runs
    """
    # Build baseline once (same for all runs)
    network_template = build_default_network()
    matrix = TravelTimeMatrix.build_synthetic()
    baseline = run_baseline(trucks, network_template, matrix)
    baseline_fuel = baseline['total_fuel']

    if verbose:
        print(f"\n{'='*65}")
        print(f"  MONTE CARLO VALIDATION — {num_runs} runs × {len(trucks)} trucks")
        print(f"{'='*65}")
        print(f"  Baseline (solo) fuel: {baseline_fuel:.2f} L")
        print()

    runs = []
    all_query_times = []
    total_start = time.perf_counter()

    for run_idx in range(num_runs):
        run_seed = seed + run_idx
        random.seed(run_seed)

        # Shuffle arrival order
        shuffled = random.sample(trucks, len(trucks))

        # Fresh network and coordinator for each run
        network = build_default_network()
        coord = PlatoonCoordinator(network, matrix)

        run_start = time.perf_counter()

        for truck in shuffled:
            coord.process_truck_arrival(truck)

        run_time = time.perf_counter() - run_start

        # Extract metrics
        actual_fuel = coord.calculate_total_fuel()
        savings_pct = ((baseline_fuel - actual_fuel) / baseline_fuel) * 100
        platoon_rate = coord.calculate_platoon_participation_rate()
        platoons = coord.get_platoon_summary()
        multi_platoons = [p for p in platoons if p['size'] > 1]

        run_result = {
            'run': run_idx + 1,
            'seed': run_seed,
            'actual_fuel': actual_fuel,
            'savings_pct': savings_pct,
            'fuel_saved_liters': baseline_fuel - actual_fuel,
            'platoon_rate': platoon_rate,
            'platoons_formed': len(multi_platoons),
            'avg_platoon_size': (
                np.mean([p['size'] for p in multi_platoons])
                if multi_platoons else 0
            ),
            'run_time_seconds': run_time,
            'query_times': list(coord.query_time_log),
        }
        runs.append(run_result)
        all_query_times.extend(coord.query_time_log)

        if verbose:
            print(f"  Run {run_idx+1:2d}/{num_runs}: "
                  f"savings={savings_pct:5.2f}%  "
                  f"platoon_rate={platoon_rate:5.1f}%  "
                  f"platoons={len(multi_platoons):2d}  "
                  f"time={run_time:.3f}s")

    total_time = time.perf_counter() - total_start

    # Statistical summary
    savings_values = [r['savings_pct'] for r in runs]
    platoon_rates = [r['platoon_rate'] for r in runs]
    query_times_ms = [t * 1000 for t in all_query_times]

    result = {
        'baseline_fuel': baseline_fuel,
        'truck_count': len(trucks),
        'num_runs': num_runs,
        'runs': runs,
        'avg_savings_pct': float(np.mean(savings_values)),
        'std_savings_pct': float(np.std(savings_values)),
        'min_savings_pct': float(np.min(savings_values)),
        'max_savings_pct': float(np.max(savings_values)),
        'avg_platoon_rate': float(np.mean(platoon_rates)),
        'std_platoon_rate': float(np.std(platoon_rates)),
        'avg_query_ms': float(np.mean(query_times_ms)) if query_times_ms else 0,
        'p95_query_ms': float(np.percentile(query_times_ms, 95)) if query_times_ms else 0,
        'max_query_ms': float(np.max(query_times_ms)) if query_times_ms else 0,
        'total_time_seconds': total_time,
    }

    if verbose:
        _print_monte_carlo_summary(result)

    return result


def _print_monte_carlo_summary(result: dict):
    """Pretty-print Monte Carlo results."""
    print(f"\n{'─'*65}")
    print(f"  MONTE CARLO SUMMARY")
    print(f"{'─'*65}")
    print(f"  Trucks: {result['truck_count']}  |  "
          f"Runs: {result['num_runs']}  |  "
          f"Total time: {result['total_time_seconds']:.2f}s")

    print(f"\n  📊 Fuel Savings:")
    print(f"     Average:  {result['avg_savings_pct']:.2f}% "
          f"± {result['std_savings_pct']:.2f}%")
    print(f"     Range:    [{result['min_savings_pct']:.2f}%, "
          f"{result['max_savings_pct']:.2f}%]")

    print(f"\n  🚛 Platoon Participation:")
    print(f"     Average:  {result['avg_platoon_rate']:.1f}% "
          f"± {result['std_platoon_rate']:.1f}%")

    print(f"\n  ⚡ Query Performance:")
    print(f"     Average:  {result['avg_query_ms']:.3f} ms")
    print(f"     95th pct: {result['p95_query_ms']:.3f} ms")
    print(f"     Maximum:  {result['max_query_ms']:.3f} ms")

    # Thesis target checks
    print(f"\n  {'─'*40}")
    print(f"  THESIS TARGET CHECKS:")
    targets = [
        ("Fuel savings > 3%", result['avg_savings_pct'] > 3.0),
        ("Platoon rate > 40%", result['avg_platoon_rate'] > 40.0),
        ("Avg query < 50ms", result['avg_query_ms'] < 50.0),
        ("Std dev < 2% (consistency)", result['std_savings_pct'] < 2.0),
    ]
    for name, passed in targets:
        icon = "✅" if passed else "❌"
        print(f"    {icon} {name}")

    print(f"{'='*65}")
