#!/usr/bin/env python3
"""Independent direct-cell float64 CFAR oracle and seeded statistical review."""
import json
import math
import subprocess
import sys
import time

started = time.monotonic()
run = subprocess.run([sys.argv[1]], check=True, text=True, capture_output=True)
report = json.loads(run.stdout)
assert report['status'] == 'PASS' and report['family_alpha'] == .001
assert report['schema_version'] == 1 and len(report['cases']) == 6
mask = (1 << 64) - 1

def powers(state, size):
    values = []
    for _ in range(size):
        state = (state + 0x9e3779b97f4a7c15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & mask
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & mask
        z ^= z >> 31
        values.append(-math.log(((z >> 12) + .5) * 2.0**-52))
    return state, values

summaries = []
max_error = 0.0
for k, case in enumerate(report['cases']):
    tr, tc, gr, gc = ((1, 2, 0, 0) if k < 3 else (2, 1, 1, 1))
    p = (0.1, 0.01, 0.001)[k % 3]
    trials = (50000, 50000, 200000)[k % 3]
    rows, cols = 2*(tr+gr)+1, 2*(tc+gc)+1
    center_row, center_col = rows//2, cols//2
    center = center_row*cols + center_col
    indices = [y*cols+x for y in range(rows) for x in range(cols)
               if abs(y-center_row) > gr or abs(x-center_col) > gc]
    count = len(indices)
    assert case['index'] == k and case['trials'] == trials and case['pfa'] == p
    assert (case['rows'], case['columns'], case['training_count']) == (rows, cols, count)
    assert (case['training_rows'], case['training_columns'], case['guard_rows'], case['guard_columns']) == (tr, tc, gr, gc)
    state = 0x5347303400000000 + k
    assert int(case['seed']) == state and len(case['samples']) == 8
    # Direct equation, independently summed selected cells; no summed-area table.
    factor = p**(-1.0/count) - 1.0
    detections = 0
    for trial in range(trials):
        state, grid = powers(state, rows*cols)
        threshold = factor * math.fsum(grid[i] for i in indices)
        detections += grid[center] > threshold
        if trial < 8:
            sample = case['samples'][trial]
            assert all(math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-12)
                       for a, b in zip(grid, sample['power'], strict=True))
            error = abs(threshold-sample['threshold'])/max(1.0, abs(threshold))
            max_error = max(max_error, error)
            assert error <= 2e-12, (k, trial, error)
    assert detections == case['detections'], (k, detections, case['detections'])
    bound = math.sqrt(2*trials*p*(1-p)*math.log(12000)) + (2/3)*math.log(12000)
    assert math.isclose(case['count_bound'], bound, rel_tol=2e-14)
    assert abs(detections-trials*p) <= bound
    summaries.append({'case': k, 'trials': trials, 'training_count': count,
                      'pfa': p, 'detections': detections,
                      'observed_rate': detections/trials, 'count_bound': bound})
print(json.dumps({'status': 'PASS', 'family_alpha': .001,
                  'method': 'two-sided Bernstein plus union bound over six predeclared cases',
                  'precision': 'float64 powers/threshold; production long-double accumulation',
                  'scope': 'iid unit-mean exponential power; one center CUT per independent grid',
                  'threshold_max_normalized_error': max_error, 'cases': summaries,
                  'elapsed_seconds': time.monotonic()-started}, indent=2))
