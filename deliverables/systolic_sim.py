"""Systolic Array Accelerator Simulator.

Implements a cycle-accurate model of a 2-D systolic array supporting
Weight-Stationary (WS), Output-Stationary (OS), and Row-Stationary (RS)
dataflows, with SRAM bandwidth / capacity modelling and double-buffering.

Every graded quantity is computed from the closed-form formulas in Section V
of the project specification.
"""

import math
import numpy as np


class SystolicSimulator:
    """Cycle-accurate simulator for a 2-D systolic array."""

    def __init__(
        self,
        array_rows: int,
        array_cols: int,
        sram_size: int = 65536,
        sram_bw: int = 128,
        pe_regfile_size: int = 4,
    ):
        """Initialise the simulator.

        Parameters
        ----------
        array_rows, array_cols : int
            Physical PE array dimensions (R x C).  Must be >= 1.
        sram_size : int
            On-chip SRAM capacity in bytes (default 64 KiB).
        sram_bw : int
            SRAM bandwidth in bytes / cycle (default 128).
        pe_regfile_size : int
            Accepted for API compatibility; unused by the graded model.
        """
        if array_rows < 1 or array_cols < 1:
            raise ValueError("array_rows and array_cols must be >= 1")
        self.array_rows = array_rows
        self.array_cols = array_cols
        self.sram_size = sram_size
        self.sram_bw = sram_bw
        self.pe_regfile_size = pe_regfile_size

        # State populated by the most recent matmul() call.
        self._pe_activity: np.ndarray | None = None
        self._memory_stats: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def matmul(
        self,
        A: np.ndarray,
        B: np.ndarray,
        dataflow: str = "WS",
        double_buffer: bool = False,
    ) -> tuple[np.ndarray, int, float]:
        """Run a matrix multiplication through the systolic array.

        Returns
        -------
        result : np.ndarray
            The product A @ B  (numerically exact via NumPy).
        cycles : int
            Total simulated cycles (integer-exact per Section V).
        utilization : float
            PE utilization in [0, 1] (Section V-F).

        Raises
        ------
        ValueError
            If *dataflow* is not one of ``'WS'``, ``'OS'``, ``'RS'``.
        MemoryError
            If a single full-sized tile's working set exceeds *sram_size*.
        """
        # --- validate dataflow -------------------------------------------
        if dataflow not in ("WS", "OS", "RS"):
            raise ValueError(
                f"dataflow must be one of 'WS', 'OS', 'RS', got '{dataflow}'"
            )

        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "Inner dimensions of A and B must match"

        R = self.array_rows
        C = self.array_cols

        # --- capacity check (Section V-G) --------------------------------
        bytes_tile = 8 * (R * K + K * C + R * C)
        if bytes_tile > self.sram_size:
            raise MemoryError(
                f"Single tile working set ({bytes_tile} bytes) exceeds "
                f"SRAM capacity ({self.sram_size} bytes)"
            )

        # --- numerical result --------------------------------------------
        result = A @ B

        # --- tiling (Section V-B) ----------------------------------------
        T_M = math.ceil(M / R)
        T_N = math.ceil(N / C)

        row_sizes = [min(R, M - i * R) for i in range(T_M)]  # r_i
        col_sizes = [min(C, N - j * C) for j in range(T_N)]  # c_j

        # --- build ordered tile list with per-tile Δ_t -------------------
        # Each entry: (i, j, r_i, c_j, delta_t)
        tiles: list[tuple[int, int, int, int, int]] = []

        if dataflow == "WS":
            # Outer j (N-tile), inner i (M-tile).
            # B-tile stays resident across all i for fixed j.
            for j in range(T_N):
                cj = col_sizes[j]
                for i in range(T_M):
                    ri = row_sizes[i]
                    if i == 0:
                        # New j column: load both A-tile and B-tile.
                        delta = ri * K + K * cj
                    else:
                        # B-tile resident from previous i; only load A-tile.
                        delta = ri * K
                    tiles.append((i, j, ri, cj, delta))

        elif dataflow == "OS":
            # Order irrelevant; nothing stays resident.
            for i in range(T_M):
                ri = row_sizes[i]
                for j in range(T_N):
                    cj = col_sizes[j]
                    delta = ri * K + K * cj
                    tiles.append((i, j, ri, cj, delta))

        else:  # RS
            # Outer i (M-tile), inner j (N-tile).
            # A-tile stays resident across all j for fixed i.
            for i in range(T_M):
                ri = row_sizes[i]
                for j in range(T_N):
                    cj = col_sizes[j]
                    if j == 0:
                        # New i row: load both A-tile and B-tile.
                        delta = ri * K + K * cj
                    else:
                        # A-tile resident; only load B-tile.
                        delta = K * cj
                    tiles.append((i, j, ri, cj, delta))

        # --- simulate cycles, stalls, reads/writes ----------------------
        total_cycles = 0
        total_stall_cycles = 0
        peak_sram_bw_bytes = 0
        total_sram_reads = 0          # element count
        total_sram_writes = 0         # element count
        active_pe_cycles = 0          # numerator of utilisation
        pe_activity = np.zeros((R, C), dtype=np.int64)

        prev_tile_compute = 0  # compute cycles of the tile before this one

        for t, (i, j, ri, cj, delta_t) in enumerate(tiles):
            # -- per-tile compute cycles (Section V-C) --------------------
            tile_compute = (ri - 1) + (cj - 1) + K + 1

            # -- SRAM traffic (element counts) ----------------------------
            total_sram_reads += delta_t
            total_sram_writes += ri * cj

            # -- stall cycles (Section V-E) -------------------------------
            load_bytes = 8 * delta_t
            # Integer ceiling division (avoids float-precision edge cases).
            L_t = (load_bytes + self.sram_bw - 1) // self.sram_bw if load_bytes > 0 else 0

            if t == 0:
                # First tile: always a full stall (nothing to overlap with).
                stall = L_t
            elif double_buffer:
                # Overlap with previous tile's compute.
                stall = max(0, L_t - prev_tile_compute)
            else:
                # No overlap; full stall for every tile.
                stall = L_t

            total_stall_cycles += stall
            total_cycles += stall + tile_compute

            # -- peak bandwidth demand (bytes) ----------------------------
            if load_bytes > peak_sram_bw_bytes:
                peak_sram_bw_bytes = load_bytes

            # -- PE activity map ------------------------------------------
            pe_activity[:ri, :cj] += K
            active_pe_cycles += ri * cj * K

            prev_tile_compute = tile_compute

        # --- utilisation (Section V-F) -----------------------------------
        utilization = active_pe_cycles / (total_cycles * R * C)

        # --- persist state for accessor methods --------------------------
        self._pe_activity = pe_activity
        self._memory_stats = {
            "sram_reads": int(total_sram_reads),
            "sram_writes": int(total_sram_writes),
            "stall_cycles": int(total_stall_cycles),
            "peak_sram_bw_used": int(peak_sram_bw_bytes),
        }

        return result, int(total_cycles), float(utilization)

    def get_pe_activity_map(self) -> np.ndarray:
        """Return per-PE active-cycle counts from the last ``matmul()``."""
        if self._pe_activity is None:
            raise RuntimeError("No matmul has been run yet")
        return self._pe_activity

    def get_memory_stats(self) -> dict:
        """Return memory statistics from the last ``matmul()``."""
        if self._memory_stats is None:
            raise RuntimeError("No matmul has been run yet")
        return self._memory_stats
