#!/usr/bin/env python3
"""Independent fixed-model Kalman consistency reconstruction (SW16b)."""
import argparse
import hashlib
import json
import math
import selectors
import subprocess
import time
from pathlib import Path

PROFILE = 'model_matched_cv_v1'
TRIALS = 1024
STEPS = 20
MAX_BYTES = 16 * 1024 * 1024
DT = 0.1
LIMITATIONS = [
    'Fixed seeded Gaussian model and known associations; no gating or survival selection.',
    'Kernel consistency diagnostic, not production multi-object or physical-sensor calibration.',
    'One final-step sample per trial; correlated time samples are not pooled.',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def keys(value, expected, label):
    require(type(value) is dict and set(value) == set(expected), label + ': invalid keys')


def vector(value, length, label):
    require(type(value) is list and len(value) == length and all(number(v) for v in value),
            label + ': invalid finite vector')
    return value


def loads_strict(raw):
    require(len(raw) <= MAX_BYTES, 'fixture size bound')

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError('nonfinite JSON constant: ' + value)

    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def transpose(a):
    return [list(row) for row in zip(*a)]


def multiply(a, b):
    return [[math.fsum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def add(a, b, scale=1):
    return [[x + scale * y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def identity(n):
    return [[float(i == j) for j in range(n)] for i in range(n)]


def spd(a, label):
    n = len(a)
    require(all(number(v) for row in a for v in row), label + ': nonfinite covariance')
    for i in range(n):
        for j in range(n):
            require(abs(a[i][j] - a[j][i]) <= 1e-12 * (1 + max(abs(a[i][j]), abs(a[j][i]))),
                    label + ': asymmetric covariance')
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            value = a[i][j] - math.fsum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                require(number(value) and value > 0, label + ': non-SPD covariance')
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value / lower[j][j]


def inverse(a):
    """Pivoted Gauss-Jordan, independent of the producer's Cholesky quadratic form."""
    spd(a, 'inverse input')
    n = len(a)
    rows = [list(row) + eye for row, eye in zip(a, identity(n))]
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(rows[i][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        divisor = rows[col][col]
        require(number(divisor) and divisor != 0, 'singular inverse')
        rows[col] = [v / divisor for v in rows[col]]
        for row in range(n):
            if row != col:
                scale = rows[row][col]
                rows[row] = [x - scale * y for x, y in zip(rows[row], rows[col])]
    result = [row[n:] for row in rows]
    require(all(number(v) for row in result for v in row), 'nonfinite inverse')
    return result


def quadratic(error, covariance):
    column = [[v] for v in error]
    result = multiply(transpose(column), multiply(inverse(covariance), column))[0][0]
    require(number(result) and result >= 0, 'invalid quadratic form')
    return result


def consistency_bounds(dimension):
    require(type(dimension) is int and dimension in (2, 4), 'invalid statistic dimension')
    alpha = 0.001
    t = math.log(4 / alpha)
    degrees = TRIALS * dimension
    spread = 2 * math.sqrt(degrees * t)
    return dict(N=TRIALS, dimension=dimension, degrees_of_freedom=degrees,
                family_alpha=alpha, t=t, lower=(degrees - spread) / TRIALS,
                upper=(degrees + spread + 2 * t) / TRIALS)


def evaluate_statistics(nees_values, nis_values):
    checks = {}
    for name, values, dimension in [('nees', nees_values, 4), ('nis', nis_values, 2)]:
        require(type(values) in (list, tuple) and len(values) == TRIALS
                and all(number(v) and v >= 0 for v in values), name + ': invalid sample population')
        mean = math.fsum(values) / TRIALS
        bounds = consistency_bounds(dimension)
        checks[name] = dict(bounds, mean=mean, passed=bounds['lower'] <= mean <= bounds['upper'])
    checks['passed'] = all(checks[name]['passed'] for name in ('nees', 'nis'))
    return checks


def validate_fixture(payload):
    keys(payload, ('schema_version', 'profile', 'trials'), 'fixture')
    require(type(payload['schema_version']) is int and payload['schema_version'] == 1
            and payload['profile'] == PROFILE, 'fixture version/profile')
    trials = payload['trials']
    require(type(trials) is list and len(trials) == TRIALS, 'incorrect trial count')
    # Validate structure first so damaged seeds/steps fail before numerical work.
    for index, trial in enumerate(trials):
        keys(trial, ('seed', 'initial_truth', 'steps', 'final_truth', 'state', 'covariance', 'nees', 'nis'), 'trial')
        require(type(trial['seed']) is int and trial['seed'] == 30000 + index, 'seed order')
        for name in ('initial_truth', 'final_truth', 'state'):
            vector(trial[name], 4, name)
        vector(trial['covariance'], 16, 'covariance')
        spd([trial['covariance'][i:i + 4] for i in range(0, 16, 4)], 'reported final')
        require(all(number(trial[name]) and trial[name] >= 0 for name in ('nees', 'nis')), 'invalid reported statistic')
        require(type(trial['steps']) is list and len(trial['steps']) == STEPS, 'incorrect step count')
        for step_index, step in enumerate(trial['steps'], 1):
            keys(step, ('step_index', 'acceleration', 'measurement_noise', 'observation'), 'step')
            require(type(step['step_index']) is int and step['step_index'] == step_index, 'step order')
            for name in ('acceleration', 'measurement_noise', 'observation'):
                vector(step[name], 2, name)
    f = [[1., 0., DT, 0.], [0., 1., 0., DT], [0., 0., 1., 0.], [0., 0., 0., 1.]]
    g = [[0.5 * DT * DT, 0.], [0., 0.5 * DT * DT], [DT, 0.], [0., DT]]
    q = multiply(g, transpose(g))
    h = [[1., 0., 0., 0.], [0., 1., 0., 0.]]
    r = [[0.25, 0.], [0., 0.25]]
    max_errors = dict(truth=0., observation=0., state=0., covariance=0., nees=0., nis=0.)

    def compare(actual, expected, name, tolerance=1e-9):
        require(number(expected), name + ': nonfinite reference')
        error = abs(actual - expected)
        max_errors[name] = max(max_errors[name], error)
        require(error <= tolerance + tolerance * max(abs(actual), abs(expected)), name + ': numerical mismatch')

    nees_values, nis_values = [], []
    for trial in trials:
        truth = [[v] for v in trial['initial_truth']]
        state = [[v] for v in [10., 30., 0.8, -0.3]]
        covariance = [[0.25, 0., 0., 0.], [0., 0.25, 0., 0.], [0., 0., 25., 0.], [0., 0., 0., 25.]]
        for step in trial['steps']:
            truth = add(multiply(f, truth), multiply(g, [[v] for v in step['acceleration']]))
            observation = add(multiply(h, truth), [[v] for v in step['measurement_noise']])
            for actual, expected in zip(step['observation'], observation):
                compare(actual, expected[0], 'observation')
            prior_state = multiply(f, state)
            prior_covariance = add(multiply(multiply(f, covariance), transpose(f)), q)
            spd(prior_covariance, 'predicted')
            innovation = add(observation, multiply(h, prior_state), -1)
            innovation_covariance = add(multiply(multiply(h, prior_covariance), transpose(h)), r)
            nis = quadratic([v[0] for v in innovation], innovation_covariance)
            gain = multiply(multiply(prior_covariance, transpose(h)), inverse(innovation_covariance))
            state = add(prior_state, multiply(gain, innovation))
            # Algebraic covariance update deliberately differs from the shared Joseph kernel.
            covariance = multiply(add(identity(4), multiply(gain, h), -1), prior_covariance)
            spd(covariance, 'posterior')
        for actual, expected in zip(trial['final_truth'], truth):
            compare(actual, expected[0], 'truth')
        for actual, expected in zip(trial['state'], state):
            compare(actual, expected[0], 'state')
        for actual, expected in zip(trial['covariance'], [v for row in covariance for v in row]):
            compare(actual, expected, 'covariance')
        nees = quadratic([x[0] - y[0] for x, y in zip(state, truth)], covariance)
        compare(trial['nees'], nees, 'nees', 1e-8)
        compare(trial['nis'], nis, 'nis', 1e-8)
        nees_values.append(nees)
        nis_values.append(nis)
    statistics = evaluate_statistics(nees_values, nis_values)
    return dict(schema_version=1, profile=PROFILE, status='PASS' if statistics['passed'] else 'FAIL',
                trial_count=len(trials), steps_per_trial=STEPS, statistics=statistics,
                max_numerical_errors=max_errors, limitations=LIMITATIONS)


def read_fixture(binary):
    """Bound both subprocess streams while reading, with a fixed execution deadline."""
    chunks = []
    size = 0
    with subprocess.Popen([binary], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
        try:
            deadline = time.monotonic() + 120
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                selector.register(child.stderr, selectors.EVENT_READ)
                while selector.get_map():
                    require(time.monotonic() < deadline, 'fixture producer timeout')
                    for key, _ in selector.select(min(1., max(0., deadline - time.monotonic()))):
                        chunk = key.fileobj.read1(65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        size += len(chunk)
                        require(size <= MAX_BYTES, 'fixture output size bound')
                        if key.fileobj is child.stdout:
                            chunks.append(chunk)
            code = child.wait(timeout=max(0.001, deadline - time.monotonic()))
            require(code == 0, 'fixture producer failed: exit ' + str(code))
        except BaseException:
            child.kill()
            child.wait()
            raise
    return b''.join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--output')
    parser.add_argument('--fixture-output')
    args = parser.parse_args()
    result = dict(schema_version=1, profile=PROFILE, status='FAIL', limitations=LIMITATIONS)
    try:
        raw = read_fixture(args.binary)
        digest = hashlib.sha256(raw).hexdigest()
        result['fixture_sha256'] = digest
        result.update(validate_fixture(loads_strict(raw)))
        if args.fixture_output:
            Path(args.fixture_output).write_bytes(raw)
    except (ValueError, OSError, subprocess.SubprocessError, OverflowError, RecursionError) as error:
        result.update(status='FAIL', error=str(error)[:2048])
    rendered = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        Path(args.output).write_text(rendered)
    print(rendered, end='')
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
