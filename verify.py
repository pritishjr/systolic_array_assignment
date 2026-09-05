#!/usr/bin/env python3
"""Verification script that checks systolic_sim.py against every worked
example and closed-form value in Section V of the project specification.

Run:  python verify.py
"""

import sys
import math
import numpy as np

from systolic_sim import SystolicSimulator

PASS = 0
FAIL = 0


def check(name, got, expected, tol=None):
    global PASS, FAIL
    if tol is None:
        ok = (got == expected)
    else:
        ok = abs(got - expected) <= tol
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAIL += 1
        print(f"  [{tag}] {name}: got {got}, expected {expected}")
    else:
        PASS += 1
        print(f"  [{tag}] {name}")


def section_sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ======================================================================
# 1. Running example from Section V: M=12, K=4, N=8, R=4, C=4
# ======================================================================
def test_running_example():
    section_sep("Running Example: M=12, K=4, N=8, R=4, C=4")
    rng = np.random.default_rng(42)
    A = rng.uniform(-1, 1, (12, 4))
    B = rng.uniform(-1, 1, (4, 8))

    # -- Tiling checks --
    R, C, M, K, N = 4, 4, 12, 4, 8
    T_M = math.ceil(M / R)
    T_N = math.ceil(N / C)
    check("T_M", T_M, 3)
    check("T_N", T_N, 2)
    check("Total tiles", T_M * T_N, 6)

    # -- Per-tile cycles (all tiles full 4x4) --
    tile_cycles = (R - 1) + (C - 1) + K + 1
    check("tile_cycles", tile_cycles, 11)

    # -- WS reads / writes (Section V-D example) --
    for df, exp_reads in [("WS", 128), ("OS", 192), ("RS", 144)]:
        sim = SystolicSimulator(R, C, sram_size=2**30, sram_bw=32)
        result, cycles, util = sim.matmul(A, B, dataflow=df, double_buffer=False)
        stats = sim.get_memory_stats()

        check(f"{df} sram_reads", stats["sram_reads"], exp_reads)
        check(f"{df} sram_writes", stats["sram_writes"], 96)
        check(f"{df} result matches A@B",
              bool(np.allclose(result, A @ B, rtol=1e-9, atol=1e-9)), True)

    # -- WS total_cycles with sram_bw=32 --
    print()
    for df, db, exp_cycles in [
        ("WS", False, 98), ("WS", True, 74),
        ("OS", False, 114), ("OS", True, 74),
        ("RS", False, 102), ("RS", True, 74),
    ]:
        sim = SystolicSimulator(R, C, sram_size=2**30, sram_bw=32)
        _, cyc, _ = sim.matmul(A, B, dataflow=df, double_buffer=db)
        check(f"{df} double_buffer={db} total_cycles", cyc, exp_cycles)

    # -- Stall details for WS, sram_bw=32 --
    print()
    sim = SystolicSimulator(R, C, sram_size=2**30, sram_bw=32)
    sim.matmul(A, B, dataflow="WS", double_buffer=True)
    stats = sim.get_memory_stats()
    # Only tile 0 stalls (8 cycles), rest hidden. Total stall = 8.
    check("WS DB stall_cycles", stats["stall_cycles"], 8)
    # peak_sram_bw_used = 8*32 = 256 (tile 0 loads 32 elements)
    check("WS DB peak_sram_bw_used", stats["peak_sram_bw_used"], 256)

    # -- PE activity map --
    pe_map = sim.get_pe_activity_map()
    check("PE activity shape", pe_map.shape, (4, 4))
    # All tiles are 4x4, K=4, 6 tiles -> each PE active for 6*4=24
    check("PE activity all equal 24", bool(np.all(pe_map == 24)), True)
    check("PE activity sum == M*K*N", int(pe_map.sum()), 12 * 4 * 8)

    # -- Utilization --
    active = 12 * 4 * 8  # = 384
    for df, db, exp_cycles in [("WS", False, 98), ("WS", True, 74)]:
        sim = SystolicSimulator(R, C, sram_size=2**30, sram_bw=32)
        _, cyc, util = sim.matmul(A, B, dataflow=df, double_buffer=db)
        exp_util = active / (exp_cycles * R * C)
        check(f"{df} DB={db} utilization", util, exp_util, tol=1e-6)


# ======================================================================
# 2. Section VIII test shapes – numerical correctness
# ======================================================================
def test_section_viii_shapes():
    section_sep("Section VIII Test Shapes – Numerical Correctness")
    rng = np.random.default_rng(42)
    shapes = [
        ("Exact multiple", (16, 16, 16), (4, 4)),
        ("Ragged", (10, 7, 13), (4, 4)),
        ("Single tile", (3, 5, 3), (8, 8)),
        ("Rectangular", (20, 20, 20), (4, 8)),
    ]
    for name, (M, K, N), (R, C) in shapes:
        A = rng.uniform(-1, 1, (M, K))
        B = rng.uniform(-1, 1, (K, N))
        for df in ("WS", "OS", "RS"):
            sim = SystolicSimulator(R, C, sram_size=2**30, sram_bw=128)
            result, cyc, util = sim.matmul(A, B, dataflow=df, double_buffer=True)
            ok = np.allclose(result, A @ B, rtol=1e-9, atol=1e-9)
            check(f"{name} {df} numerical", bool(ok), True)


# ======================================================================
# 3. Ragged-tile test: (10,7,13) on 4x4 array
# ======================================================================
def test_ragged():
    section_sep("Ragged Tile: M=10, K=7, N=13, R=4, C=4")
    M, K, N, R, C = 10, 7, 13, 4, 4
    T_M = math.ceil(M / R)  # 3
    T_N = math.ceil(N / C)  # 4
    row_sizes = [min(R, M - i * R) for i in range(T_M)]  # [4, 4, 2]
    col_sizes = [min(C, N - j * C) for j in range(T_N)]  # [4, 4, 4, 1]

    check("row_sizes", row_sizes, [4, 4, 2])
    check("col_sizes", col_sizes, [4, 4, 4, 1])

    # WS reads = Σ_j K*c_j + Σ_i Σ_j r_i*K
    ws_B = sum(K * cj for cj in col_sizes)             # 7*(4+4+4+1)=91
    ws_A = sum(ri * K for ri in row_sizes) * T_N        # (4+4+2)*7*4 = 280
    ws_reads = ws_B + ws_A                               # 371
    check("WS reads formula", ws_reads, 371)

    rng = np.random.default_rng(42)
    A = rng.uniform(-1, 1, (M, K))
    B = rng.uniform(-1, 1, (K, N))

    sim = SystolicSimulator(R, C, sram_size=2**30, sram_bw=128)
    _, _, _ = sim.matmul(A, B, dataflow="WS", double_buffer=False)
    stats = sim.get_memory_stats()
    check("WS sram_reads (simulated)", stats["sram_reads"], ws_reads)

    # Writes = Σ r_i * c_j
    writes = sum(ri * cj for ri in row_sizes for cj in col_sizes)
    check("sram_writes", stats["sram_writes"], writes)

    # PE activity sum should equal Σ r_i*c_j*K
    active = sum(ri * cj * K for ri in row_sizes for cj in col_sizes)
    pe_map = sim.get_pe_activity_map()
    check("PE activity sum", int(pe_map.sum()), active)


# ======================================================================
# 4. Capacity check (MemoryError)
# ======================================================================
def test_capacity_check():
    section_sep("Capacity Check (MemoryError)")
    R, C, K = 4, 4, 100
    bytes_tile = 8 * (R * K + K * C + R * C)  # 8*(400+400+16) = 6528
    # Set sram_size just below -> should raise
    sim_fail = SystolicSimulator(R, C, sram_size=bytes_tile - 1, sram_bw=128)
    A = np.ones((8, K))
    B = np.ones((K, 8))
    try:
        sim_fail.matmul(A, B)
        check("MemoryError raised", False, True)
    except MemoryError:
        check("MemoryError raised", True, True)

    # Set sram_size exactly equal -> should pass
    sim_ok = SystolicSimulator(R, C, sram_size=bytes_tile, sram_bw=128)
    try:
        sim_ok.matmul(A, B)
        check("No MemoryError when fits", True, True)
    except MemoryError:
        check("No MemoryError when fits", False, True)


# ======================================================================
# 5. ValueError for bad dataflow
# ======================================================================
def test_bad_dataflow():
    section_sep("ValueError for Invalid Dataflow")
    sim = SystolicSimulator(4, 4)
    A = np.ones((4, 4))
    B = np.ones((4, 4))
    try:
        sim.matmul(A, B, dataflow="XY")
        check("ValueError raised", False, True)
    except ValueError:
        check("ValueError raised", True, True)


# ======================================================================
# 6. Single-tile case: (3,5,3) on 8x8 array
# ======================================================================
def test_single_tile():
    section_sep("Single Tile: M=3, K=5, N=3, R=8, C=8")
    M, K, N, R, C = 3, 5, 3, 8, 8
    rng = np.random.default_rng(42)
    A = rng.uniform(-1, 1, (M, K))
    B = rng.uniform(-1, 1, (K, N))

    # Only one tile: r=3, c=3
    tile_cycles = (3 - 1) + (3 - 1) + 5 + 1  # 10
    # All dataflows identical for single tile
    delta_0 = 3 * 5 + 5 * 3  # 30 elements = 240 bytes
    # sram_bw=128: L_0 = ceil(240/128) = 2
    L_0 = math.ceil(240 / 128)  # 2
    total = L_0 + tile_cycles  # 12
    reads = 30
    writes = 3 * 3  # 9

    for df in ("WS", "OS", "RS"):
        sim = SystolicSimulator(R, C, sram_size=2**30, sram_bw=128)
        _, cyc, util = sim.matmul(A, B, dataflow=df, double_buffer=False)
        stats = sim.get_memory_stats()

        check(f"{df} cycles", cyc, total)
        check(f"{df} reads", stats["sram_reads"], reads)
        check(f"{df} writes", stats["sram_writes"], writes)

        active = 3 * 3 * 5  # 45
        exp_util = active / (total * R * C)
        check(f"{df} utilization", util, exp_util, tol=1e-6)


# ======================================================================
# 7. Double-buffer vs no-double-buffer consistency
# ======================================================================
def test_db_vs_nodb():
    section_sep("Double-Buffer vs No-Double-Buffer Consistency")
    rng = np.random.default_rng(42)
    A = rng.uniform(-1, 1, (20, 20))
    B = rng.uniform(-1, 1, (20, 20))

    for df in ("WS", "OS", "RS"):
        sim_nodb = SystolicSimulator(4, 8, sram_size=2**30, sram_bw=128)
        _, cyc_nodb, _ = sim_nodb.matmul(A, B, dataflow=df, double_buffer=False)
        stats_nodb = sim_nodb.get_memory_stats()

        sim_db = SystolicSimulator(4, 8, sram_size=2**30, sram_bw=128)
        _, cyc_db, _ = sim_db.matmul(A, B, dataflow=df, double_buffer=True)
        stats_db = sim_db.get_memory_stats()

        # Reads/writes should be identical (traffic doesn't depend on DB)
        check(f"{df} reads same", stats_nodb["sram_reads"], stats_db["sram_reads"])
        check(f"{df} writes same", stats_nodb["sram_writes"], stats_db["sram_writes"])
        # DB should never increase cycles
        check(f"{df} DB cycles <= noDB", cyc_db <= cyc_nodb, True)


# ======================================================================
# Main
# ======================================================================
def main():
    test_running_example()
    test_section_viii_shapes()
    test_ragged()
    test_capacity_check()
    test_bad_dataflow()
    test_single_tile()
    test_db_vs_nodb()

    section_sep("SUMMARY")
    total = PASS + FAIL
    print(f"  {PASS}/{total} checks passed, {FAIL} failed.")
    if FAIL > 0:
        print("  *** SOME CHECKS FAILED ***")
        sys.exit(1)
    else:
        print("  All checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
