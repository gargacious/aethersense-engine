#!/usr/bin/env python3
"""Independent Gauss-Jordan quadratic-form check and posthoc aggregate evaluation."""
import json
import math
import subprocess
import sys

def inverse(matrix):
    augmented = [list(row) + [float(i == j) for j in range(4)] for i, row in enumerate(matrix)]
    for column in range(4):
        pivot = max(range(column, 4), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        assert abs(divisor) > 1e-15
        augmented[column] = [v / divisor for v in augmented[column]]
        for row in range(4):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [v - factor * q for v, q in zip(augmented[row], augmented[column])]
    return [row[4:] for row in augmented]

def main():
    result = subprocess.run([sys.argv[1], '--oracle'], check=True, text=True, capture_output=True)
    report = json.loads(result.stdout)
    nees, position_error, maximum_error = [], 0., 0.
    for sample in report.pop('records'):
        covariance = [sample['covariance'][i:i + 4] for i in range(0, 16, 4)]
        inv = inverse(covariance)
        error = [a - b for a, b in zip(sample['state'], sample['truth'])]
        value = math.fsum(error[i] * inv[i][j] * error[j] for i in range(4) for j in range(4))
        maximum_error = max(maximum_error, abs(value - sample['nees']))
        assert abs(value - sample['nees']) < 1e-10 * max(1., abs(value))
        nees.append(value)
        position_error += sum(v * v for v in error[:2])
    assert len(nees) == report['completed_original_tracks']
    assert abs(math.fsum(nees) / len(nees) - report['final_state_mean_nees']) < 1e-12
    assert abs(math.sqrt(position_error / len(nees)) - report['final_position_rmse_m']) < 1e-12
    report['oracle'] = 'independent Python stdlib Gauss-Jordan inverse vs C++ Cholesky solve'
    report['maximum_nees_absolute_error'] = maximum_error
    report['retention_fraction'] = report['completed_original_tracks'] / report['seeds']
    report['limitations'] = ['NEES/RMSE conditional on original-track survival',
                             'deterministic CV truth has no acceleration process noise',
                             'gating/association and survival selection prevent chi-square calibration claims',
                             'crossing identity switches observed; no retention or identity acceptance claim']
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
