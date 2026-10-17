#!/usr/bin/env python3
"""Independent malformed-ledger, denominator and frozen-budget regressions."""
import argparse
import copy
import json
import math
import subprocess
import unittest

import tracking_regression_oracle as oracle


class RegressionContractTests(unittest.TestCase):
    payload = None
    budget = None

    def fixture(self):
        return copy.deepcopy(self.payload)

    def test_actual_fixture_reconciles(self):
        metrics = oracle.reduce_fixture(self.payload)
        oracle.verify_report(self.payload, metrics)
        oracle.validate_budget(self.budget)
        checks = oracle.evaluate_budget(metrics, self.budget)
        self.assertEqual(len(checks), 10)
        self.assertTrue(all(check['passed'] for check in checks))
        self.assertEqual(metrics['eligible_post_warmup_frames'], 40448)
        self.assertEqual(metrics['final_associated_count'] + metrics['final_unassociated_count'], 256)

    def test_missing_final_association_is_not_zero_error(self):
        payload = self.fixture()
        trial = payload['regression_fixture']['trials'][0]
        for track in trial['frames'][-1]['tracks']:
            track['matched_observation_ids'] = []
        for track in trial['final_tracks']:
            track['matched_observations'] = []
        metrics = oracle.reduce_fixture(payload)
        squared_errors = []
        for sample in payload['regression_fixture']['trials']:
            for track in sample['final_tracks']:
                if track['matched_observations']:
                    squared_errors.append(sum((track['state'][i] - sample['final_truth'][i]) ** 2
                                              for i in (0, 1)))
        self.assertEqual(metrics['final_associated_count'], len(squared_errors))
        self.assertEqual(metrics['final_unassociated_count'], 256 - len(squared_errors))
        self.assertAlmostEqual(metrics['final_associated_position_rmse_m'],
                               math.sqrt(math.fsum(squared_errors) / len(squared_errors)), places=12)
        self.assertEqual(metrics['eligible_post_warmup_frames'], 40448)
        self.assertEqual(metrics['associated_post_warmup_frames'],
                         oracle.reduce_fixture(self.payload)['associated_post_warmup_frames'] - 1)
        with self.assertRaises(ValueError):
            oracle.verify_report(payload, metrics)

    def test_empty_population_keeps_denominators_and_nulls(self):
        payload = self.fixture()
        for trial in payload['regression_fixture']['trials']:
            for frame in trial['frames']:
                frame['tracks'] = []
            trial['final_tracks'] = []
        metrics = oracle.reduce_fixture(payload)
        self.assertEqual(metrics['eligible_post_warmup_frames'], 40448)
        self.assertEqual(metrics['associated_post_warmup_frames'], 0)
        self.assertEqual(metrics['confirmed_coverage_all_eligible_frames'], 0)
        self.assertEqual(metrics['final_associated_count'], 0)
        self.assertEqual(metrics['final_unassociated_count'], 256)
        self.assertEqual(metrics['original_never_confirmed_count'], 256)
        self.assertEqual(metrics['original_track_losses'], 256)
        for key in ('confirmed_observation_coverage_after_warmup',
                    'final_associated_position_rmse_m', 'mean_original_confirmation_delay_s'):
            self.assertIsNone(metrics[key], key)
        with self.assertRaises(ValueError):
            oracle.evaluate_budget(metrics, self.budget)
        # A present null is valid; an omitted field must not masquerade as null.
        for key, value in metrics.items():
            if key.startswith('crossing_'):
                payload['crossing'][key.removeprefix('crossing_')] = value
            else:
                payload[key] = value
        oracle.verify_report(payload, metrics)
        del payload['mean_original_confirmation_delay_s']
        with self.assertRaises(ValueError):
            oracle.verify_report(payload, metrics)

    def test_never_confirmed_is_not_immediate_confirmation(self):
        payload = self.fixture()
        for trial in payload['regression_fixture']['trials']:
            for frame in trial['frames']:
                for track in frame['tracks']:
                    if track['track_id'] == 1:
                        track['confirmed'] = False
            for track in trial['final_tracks']:
                if track['track_id'] == 1:
                    track['lifecycle'] = 'tentative'
        metrics = oracle.reduce_fixture(payload)
        self.assertEqual(metrics['original_confirmed_count'], 0)
        self.assertEqual(metrics['original_never_confirmed_count'], 256)
        self.assertIsNone(metrics['mean_original_confirmation_delay_s'])

    def test_report_mutations_are_rejected(self):
        metrics = oracle.reduce_fixture(self.payload)
        for key in ('eligible_post_warmup_frames', 'associated_post_warmup_frames',
                    'confirmed_associated_post_warmup_frames', 'final_associated_count',
                    'final_unassociated_count', 'original_confirmed_count',
                    'original_never_confirmed_count', 'original_track_losses',
                    'observation_track_identity_transitions'):
            with self.subTest(field=key):
                payload = self.fixture()
                payload[key] += 1
                with self.assertRaises(ValueError):
                    oracle.verify_report(payload, metrics)
        for value in (None, True, '0.2', float('nan'), float('inf'), -100):
            with self.subTest(value=repr(value)):
                payload = self.fixture()
                payload['mean_original_confirmation_delay_s'] = value
                with self.assertRaises(ValueError):
                    oracle.verify_report(payload, metrics)
        for key, value in (('seeds', 256.0), ('frames_per_seed', 160.0),
                           ('schema_version', True), ('status', 'FAIL'),
                           ('final_state_mean_nees', float('nan')),
                           ('final_nees_p95', float('inf'))):
            with self.subTest(field=key, value=value):
                payload = self.fixture()
                payload[key] = value
                with self.assertRaises(ValueError):
                    oracle.verify_report(payload, metrics)
        for value in (False, 1, None):
            with self.subTest(deterministic_repeat=value):
                payload = self.fixture()
                payload['crossing']['deterministic_repeat'] = value
                with self.assertRaises(ValueError):
                    oracle.verify_report(payload, metrics)

    def test_malformed_ledgers_are_rejected(self):
        def first(payload):
            return payload['regression_fixture']['trials'][0]

        mutations = {
            'missing seed': lambda p: p['regression_fixture']['trials'].pop(),
            'duplicate seed': lambda p: p['regression_fixture']['trials'][1].update(seed=9000),
            'wrong seed type': lambda p: first(p).update(seed=True),
            'missing frame': lambda p: first(p)['frames'].pop(),
            'frame order': lambda p: first(p)['frames'][1].update(frame_index=0),
            'unknown frame field': lambda p: first(p)['frames'][0].update(extra=1),
            'unknown track field': lambda p: first(p)['frames'][0]['tracks'][0].update(extra=1),
            'boolean track id': lambda p: first(p)['frames'][0]['tracks'][0].update(track_id=True),
            'nonboolean confirmed': lambda p: first(p)['frames'][0]['tracks'][0].update(confirmed=1),
            'duplicate track': lambda p: first(p)['frames'][0]['tracks'].append(
                copy.deepcopy(first(p)['frames'][0]['tracks'][0])),
            'duplicate match': lambda p: first(p)['frames'][0]['tracks'][0].update(
                matched_observation_ids=[0, 0]),
            'wrong observation frame': lambda p: first(p)['frames'][0]['tracks'][0].update(
                matched_observation_ids=[1]),
            'missing final identity': lambda p: first(p).update(final_tracks=[]),
            'noninteger final match': lambda p: first(p)['final_tracks'][0].update(
                matched_observations=[159.0]),
            'nonfinite truth': lambda p: first(p)['final_truth'].__setitem__(0, float('nan')),
            'nonfinite state': lambda p: first(p)['final_tracks'][0]['state'].__setitem__(0, float('inf')),
            'wrong crossing seed': lambda p: p['regression_fixture']['crossing'].update(seed=457),
            'missing crossing frame': lambda p: p['regression_fixture']['crossing']['frames'].pop(),
        }
        for name, mutation in mutations.items():
            with self.subTest(mutation=name):
                payload = self.fixture()
                mutation(payload)
                with self.assertRaises(ValueError):
                    oracle.reduce_fixture(payload)

    def test_budget_edges_are_inclusive_and_exceedance_fails(self):
        metrics = oracle.reduce_fixture(self.payload)
        for limit, bound in self.budget['limits'].items():
            metric, direction = limit.rsplit('_', 1)
            with self.subTest(limit=limit):
                changed = dict(metrics)
                changed[metric] = bound
                checks = oracle.evaluate_budget(changed, self.budget)
                self.assertTrue(all(check['passed'] for check in checks))
                delta = 1 if isinstance(bound, int) else 1e-6
                changed[metric] = bound + (delta if direction == 'max' else -delta)
                checks = oracle.evaluate_budget(changed, self.budget)
                self.assertFalse(all(check['passed'] for check in checks))
                failed = [check for check in checks if not check['passed']]
                self.assertEqual(len(failed), 1)
                self.assertEqual(failed[0]['metric'], metric)

    def test_invalid_budgets_are_rejected(self):
        mutations = [lambda b: b.update(schema_version=True),
                     lambda b: b.update(profile='unknown'),
                     lambda b: b.update(extra=1),
                     lambda b: b['limits'].pop('original_track_losses_max'),
                     lambda b: b['limits'].update(extra_max=1)]
        mutations.extend(lambda b, value=value: b['limits'].update(original_track_losses_max=value)
                         for value in (None, True, '2', float('nan'), float('inf'), -1))
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                budget = copy.deepcopy(self.budget)
                mutation(budget)
                with self.assertRaises(ValueError):
                    oracle.validate_budget(budget)

    def test_raw_json_duplicates_and_nonfinite_literals_are_rejected(self):
        for raw in ('{"schema_version": 1, "schema_version": 2}',
                    '{"limits":{"original_track_losses_max":2,"original_track_losses_max":99}}',
                    '{"regression_fixture":{"profile":"cv_tracking_regression_v1",'
                    '"profile":"other"}}',
                    '{"value": NaN}', '{"value": Infinity}', '{"value": -Infinity}'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    oracle.loads_strict(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--budget', required=True)
    args = parser.parse_args()
    completed = subprocess.run([args.binary, '--regression-fixture'], check=True,
                               text=True, capture_output=True, timeout=60)
    RegressionContractTests.payload = json.loads(completed.stdout)
    with open(args.budget, encoding='utf-8') as source:
        RegressionContractTests.budget = json.load(source)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RegressionContractTests)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if outcome.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
