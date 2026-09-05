# Systolic Array Accelerator Simulator

## Overview

A cycle-accurate software simulator of a systolic array that models matrix
multiplication through a 2-D PE grid, including memory bandwidth constraints
and three dataflow strategies (WS, OS, RS).

## Requirements

- **Python 3.10+**
- **NumPy** (the only external dependency)

```bash
pip install numpy
```

## Files

| File | Purpose |
|---|---|
| `systolic_sim.py` | Core simulator – the single module imported by the autograder. |
| `run_all.py` | Driver that generates `results.json` for Levels 4–5. |
| `results.json` | Generated numeric answers (created by `run_all.py`). |
| `README.md` | This file. |

## Quick Start

```bash
# Generate results.json
python run_all.py
```

## API Usage

```python
import numpy as np
from systolic_sim import SystolicSimulator

sim = SystolicSimulator(array_rows=4, array_cols=4,
                        sram_size=65536, sram_bw=128)

A = np.random.randn(12, 4)
B = np.random.randn(4, 8)

result, cycles, utilization = sim.matmul(A, B, dataflow='WS',
                                          double_buffer=True)
pe_map   = sim.get_pe_activity_map()
mem_stats = sim.get_memory_stats()

print(f"Cycles: {cycles}, Utilization: {utilization:.4f}")
print(f"SRAM reads: {mem_stats['sram_reads']}, writes: {mem_stats['sram_writes']}")
```

## Supported Dataflows

- **WS (Weight-Stationary):** B-tile stays in the array while A-tiles stream through.
- **OS (Output-Stationary):** No operand reuse; both tiles are re-fetched each time.
- **RS (Row-Stationary):** A-tile stays in the array while B-tiles stream through.
