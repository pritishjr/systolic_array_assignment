#!/usr/bin/env python3
"""Driver script that runs all FIXED_LAYERS through the simulator and writes
results.json for Levels 4-5 of the autograder.

Usage:
    python run_all.py
"""

import json
import math
import sys

import numpy as np

from systolic_sim import SystolicSimulator

# ======================================================================
# FIXED_LAYERS (Section VIII) – (M, K, N)
# ======================================================================
FIXED_LAYERS: dict[str, tuple[int, int, int]] = {
    "resnet_stem":   (1, 3 * 7 * 7, 64),   # (1, 147, 64)
    "resnet_block1": (1, 64, 64),
    "resnet_block3": (1, 256, 512),
    "resnet_fc":     (1, 2048, 1000),
}

# Use a large SRAM size so the capacity check never fires during sweeps.
# The project spec does not constrain sram_size for Levels 4-5.
LARGE_SRAM = 2 ** 30  # 1 GiB – far more than any tile ever needs.


def _make_matrices(M: int, K: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic random (M,K) and (K,N) float64 matrices."""
    rng = np.random.default_rng(42)
    A = rng.uniform(-1, 1, size=(M, K))
    B = rng.uniform(-1, 1, size=(K, N))
    return A, B


# ======================================================================
# Level 4 – Dataflow comparison (16x16 array, sram_bw=128, double_buffer)
# ======================================================================
def build_level4() -> dict:
    traffic_by_dataflow: dict[str, dict[str, int]] = {}
    cycles_by_dataflow: dict[str, dict[str, int]] = {}
    best_dataflow_by_traffic: dict[str, str] = {}

    for layer_name, (M, K, N) in FIXED_LAYERS.items():
        A, B = _make_matrices(M, K, N)
        traffic: dict[str, int] = {}
        cycles: dict[str, int] = {}

        for df in ("WS", "OS", "RS"):
            sim = SystolicSimulator(16, 16, sram_size=LARGE_SRAM, sram_bw=128)
            _, cyc, _ = sim.matmul(A, B, dataflow=df, double_buffer=True)
            stats = sim.get_memory_stats()
            traffic[df] = stats["sram_reads"] + stats["sram_writes"]
            cycles[df] = cyc

        traffic_by_dataflow[layer_name] = traffic
        cycles_by_dataflow[layer_name] = cycles
        best_dataflow_by_traffic[layer_name] = min(traffic, key=traffic.get)

    return {
        "traffic_by_dataflow": traffic_by_dataflow,
        "cycles_by_dataflow": cycles_by_dataflow,
        "best_dataflow_by_traffic": best_dataflow_by_traffic,
    }


# ======================================================================
# Level 5 – Roofline sweep & design-space exploration
# ======================================================================
ARRAY_SIZES = [8, 16, 32, 64, 128]
SRAM_BWS    = [32, 64, 128, 256, 512]


def build_level5() -> dict:
    roofline_points: list[dict] = []
    # Track total cycles per config across all layers.
    config_total_cycles: dict[tuple[int, int], int] = {}

    for layer_name, (M, K, N) in FIXED_LAYERS.items():
        A, B = _make_matrices(M, K, N)
        macs = M * K * N

        for arr in ARRAY_SIZES:
            for bw in SRAM_BWS:
                try:
                    sim = SystolicSimulator(arr, arr, sram_size=LARGE_SRAM, sram_bw=bw)
                    _, cyc, _ = sim.matmul(A, B, dataflow="WS", double_buffer=True)
                except MemoryError:
                    continue  # skip infeasible configurations

                stats = sim.get_memory_stats()
                bytes_moved = 8 * (stats["sram_reads"] + stats["sram_writes"])
                ai = macs / bytes_moved   # MACs / byte
                perf = macs / cyc         # MACs / cycle

                roofline_points.append({
                    "layer": layer_name,
                    "array": arr,
                    "sram_bw": bw,
                    "AI": ai,
                    "P": perf,
                })

                key = (arr, bw)
                config_total_cycles[key] = config_total_cycles.get(key, 0) + cyc

    # A config is only valid for best_config if it ran *all* layers.
    n_layers = len(FIXED_LAYERS)
    # Count how many layers each config completed.
    config_layer_count: dict[tuple[int, int], int] = {}
    for pt in roofline_points:
        key = (pt["array"], pt["sram_bw"])
        config_layer_count[key] = config_layer_count.get(key, 0) + 1

    valid_configs = {k: v for k, v in config_total_cycles.items()
                     if config_layer_count.get(k, 0) == n_layers}

    best_key = min(valid_configs, key=valid_configs.get)

    # Ridge point: array=128, sram_bw=128.
    peak_compute = 128 * 128   # MACs / cycle
    ridge_bw = 128             # bytes / cycle
    ridge_ai = peak_compute / ridge_bw

    return {
        "roofline_points": roofline_points,
        "ridge_point_AI": ridge_ai,
        "best_config": {"array": best_key[0], "sram_bw": best_key[1]},
    }


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    print("Building Level 4 results …")
    level4 = build_level4()

    print("Building Level 5 results …")
    level5 = build_level5()

    results = {"level4": level4, "level5": level5}

    out_path = "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
