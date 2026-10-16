#!/usr/bin/env python3
"""Independent stdlib direct DFT and nested-loop 2D CFAR reference."""
import cmath
import json
import math
import subprocess
import sys

def main():
    result = subprocess.run([sys.argv[1], '--oracle'], check=True, text=True, capture_output=True)
    data = json.loads(result.stdout)
    samples = [complex(*v) for v in data['fft_input']]
    expected = [sum(v * cmath.exp(-2j * math.pi * k * n / len(samples))
                    for n, v in enumerate(samples)) for k in range(len(samples))]
    fft_errors = {}
    for name, tolerance in [('float64', 2e-11), ('float32', 2e-4)]:
        errors = [abs(complex(*v) - ref) for v, ref in zip(data[name], expected)]
        assert max(errors) < tolerance, (name, max(errors))
        fft_errors[name] = max(errors)
    rows, cols = data['rows'], data['columns']
    tr, tc = data['training_rows'], data['training_columns']
    gr, gc = data['guard_rows'], data['guard_columns']
    ry, rx = tr + gr, tc + gc
    count = (2 * ry + 1) * (2 * rx + 1) - (2 * gr + 1) * (2 * gc + 1)
    alpha = count * math.expm1(-math.log(data['pfa']) / count)
    largest = 0
    for y in range(rows):
        for x in range(cols):
            actual = data['threshold'][y * cols + x]
            if y < ry or y >= rows - ry or x < rx or x >= cols - rx:
                assert actual == -1
                continue
            training = [data['power'][(y + dy) * cols + x + dx]
                        for dy in range(-ry, ry + 1) for dx in range(-rx, rx + 1)
                        if abs(dy) > gr or abs(dx) > gc]
            assert len(training) == count
            ref = alpha * math.fsum(training) / count
            error = abs(actual - ref)
            largest = max(error, largest)
            assert error < 1e-11 * max(1., abs(ref))
    print(json.dumps({'status': 'PASS', 'oracle': 'stdlib direct DFT/nested rectangle',
                      'fft_max_complex_error': fft_errors, 'cfar_max_absolute_error': largest,
                      'cfar_training_count': count, 'grid_shape': [rows, cols]}))

if __name__ == '__main__':
    main()
