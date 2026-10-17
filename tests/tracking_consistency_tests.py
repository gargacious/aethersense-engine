#!/usr/bin/env python3
"""Independent SW16b population, reconstruction and statistical-bound regressions."""
import argparse
import copy
import json
import math
import subprocess
import unittest

import tracking_consistency_oracle as oracle


class ConsistencyContractTests(unittest.TestCase):
    payload = None

    def fixture(self):
        # Only the first trial is mutated. Preserve independent input for each case
        # without repeatedly copying the remaining fixed 1023 trials.
        result = dict(self.payload)
        result['trials'] = list(self.payload['trials'])
        result['trials'][0] = copy.deepcopy(result['trials'][0])
        return result

    def rejected(self, mutation):
        payload = self.fixture()
        mutation(payload)
        with self.assertRaises(ValueError):
            oracle.validate_fixture(payload)

    def test_actual_unconditional_fixture(self):
        report = oracle.validate_fixture(self.payload)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['trial_count'], 1024)
        for name, dimension in [('nees', 4), ('nis', 2)]:
            self.assertEqual(report['statistics'][name]['N'], 1024)
            self.assertEqual(report['statistics'][name]['dimension'], dimension)
            self.assertTrue(report['statistics'][name]['passed'])

    def test_missing_duplicate_and_reordered_trials(self):
        mutations = [
            lambda p: p['trials'].pop(),
            lambda p: p['trials'].__setitem__(1, p['trials'][0]),
            lambda p: p['trials'].__setitem__(slice(0, 2), p['trials'][1::-1]),
            lambda p: p['trials'][0].__setitem__('seed', True),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.rejected(mutate)

    def test_missing_duplicate_and_reordered_steps(self):
        mutations = [
            lambda p: p['trials'][0]['steps'].pop(),
            lambda p: p['trials'][0]['steps'].__setitem__(1, p['trials'][0]['steps'][0]),
            lambda p: p['trials'][0]['steps'].__setitem__(
                slice(0, 2), p['trials'][0]['steps'][1::-1]),
            lambda p: p['trials'][0]['steps'][0].__setitem__('step_index', True),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.rejected(mutate)

    def test_nonfinite_and_wrong_types(self):
        for field in ('initial_truth', 'final_truth', 'state', 'covariance'):
            for invalid in (float('nan'), float('inf'), True, '1'):
                with self.subTest(field=field, invalid=invalid):
                    self.rejected(lambda p, f=field, v=invalid:
                                  p['trials'][0][f].__setitem__(0, v))
        for field in ('nees', 'nis'):
            for invalid in (float('nan'), float('-inf'), True):
                with self.subTest(field=field, invalid=invalid):
                    self.rejected(lambda p, f=field, v=invalid:
                                  p['trials'][0].__setitem__(f, v))

    def test_asymmetric_indefinite_and_singular_covariance(self):
        for index, value in ((1, 1.0), (0, -1.0)):
            self.rejected(lambda p, i=index, v=value:
                          p['trials'][0]['covariance'].__setitem__(i, v))
        self.rejected(lambda p: p['trials'][0].__setitem__('covariance', [0.0] * 16))

    def test_reconstructed_state_truth_and_scalar_mismatch(self):
        for field in ('initial_truth', 'final_truth', 'state'):
            self.rejected(lambda p, f=field: p['trials'][0][f].__setitem__(
                0, p['trials'][0][f][0] + 1.0))
        for field in ('nees', 'nis'):
            self.rejected(lambda p, f=field: p['trials'][0].__setitem__(
                f, p['trials'][0][f] + 1.0))

    def test_model_increments_and_observation_mismatch(self):
        for field in ('acceleration', 'measurement_noise', 'observation'):
            self.rejected(lambda p, f=field: p['trials'][0]['steps'][0][f].__setitem__(
                0, p['trials'][0]['steps'][0][f][0] + 1.0))

    def test_unknown_fields_and_dimensions(self):
        self.rejected(lambda p: p.__setitem__('unreviewed', True))
        self.rejected(lambda p: p['trials'][0].__setitem__('unreviewed', True))
        self.rejected(lambda p: p['trials'][0]['steps'][0].__setitem__('unreviewed', True))
        self.rejected(lambda p: p['trials'][0]['state'].pop())
        self.rejected(lambda p: p['trials'][0]['steps'][0]['observation'].append(0))

    def test_strict_json_and_size_boundary(self):
        for raw in (b'{"schema_version":1,"schema_version":1}',
                    b'{"value":NaN}', b'{"value":Infinity}',
                    b' ' * (16 * 1024 * 1024 + 1)):
            with self.assertRaises(ValueError):
                oracle.loads_strict(raw)

    def test_independent_frozen_bounds(self):
        for dimension in (2, 4):
            actual = oracle.consistency_bounds(dimension)
            k = 1024 * dimension
            t = math.log(4000)
            self.assertEqual(actual['N'], 1024)
            self.assertEqual(actual['degrees_of_freedom'], k)
            self.assertEqual(actual['family_alpha'], 0.001)
            self.assertAlmostEqual(actual['t'], t, places=14)
            self.assertAlmostEqual(actual['lower'], (k - 2 * math.sqrt(k * t)) / 1024,
                                   places=14)
            self.assertAlmostEqual(actual['upper'],
                                   (k + 2 * math.sqrt(k * t) + 2 * t) / 1024, places=14)

    def test_inclusive_bounds_and_each_outside_tail(self):
        for name, dimension in [('nees', 4), ('nis', 2)]:
            bounds = oracle.consistency_bounds(dimension)
            for edge in ('lower', 'upper'):
                samples = {'nees': [4.0] * 1024, 'nis': [2.0] * 1024}
                samples[name] = [bounds[edge]] * 1024
                result = oracle.evaluate_statistics(samples['nees'], samples['nis'])
                self.assertTrue(result['passed'], (name, edge, result))
                # A representable distance outside avoids platform fsum rounding
                # ambiguity while remaining much smaller than statistical margins.
                delta = -1e-10 if edge == 'lower' else 1e-10
                samples[name] = [bounds[edge] + delta] * 1024
                result = oracle.evaluate_statistics(samples['nees'], samples['nis'])
                self.assertFalse(result['passed'], (name, edge, result))
                self.assertFalse(result[name]['passed'])

    def test_missing_samples_never_improve_statistics(self):
        for which in (0, 1):
            for values in ([], [4.0] * 1023, [4.0] * 1025,
                           [float('nan')] + [4.0] * 1023,
                           [-1.0] + [4.0] * 1023,
                           [True] + [4.0] * 1023):
                samples = [[4.0] * 1024, [2.0] * 1024]
                samples[which] = values
                with self.assertRaises(ValueError):
                    oracle.evaluate_statistics(*samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True)
    args = parser.parse_args()
    completed = subprocess.run([args.binary], capture_output=True, check=True, timeout=60)
    if len(completed.stdout) > 16 * 1024 * 1024:
        raise ValueError('Fixture exceeds 16 MiB')
    ConsistencyContractTests.payload = json.loads(completed.stdout)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ConsistencyContractTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
