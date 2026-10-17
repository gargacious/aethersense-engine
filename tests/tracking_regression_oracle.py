#!/usr/bin/env python3
"""Independent reduction of the fixed SW16a tracking decision ledger."""
import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

PROFILE = 'cv_tracking_regression_v1'
MAX_BYTES = 16 * 1024 * 1024
LIMITS = {
    'original_track_losses_max': ('original_track_losses', '<=', 256, True),
    'observation_track_identity_transitions_max': ('observation_track_identity_transitions', '<=', 40704, True),
    'confirmed_coverage_all_eligible_frames_min': ('confirmed_coverage_all_eligible_frames', '>=', 1, False),
    'final_associated_count_min': ('final_associated_count', '>=', 256, True),
    'final_associated_position_rmse_m_max': ('final_associated_position_rmse_m', '<=', None, False),
    'original_confirmed_count_min': ('original_confirmed_count', '>=', 256, True),
    'mean_original_confirmation_delay_s_max': ('mean_original_confirmation_delay_s', '<=', 15.9, False),
    'crossing_associations_min': ('crossing_associations', '>=', 300, True),
    'crossing_identity_switches_max': ('crossing_identity_switches', '<=', 300, True),
    'crossing_majority_association_purity_min': ('crossing_majority_association_purity', '>=', 1, False),
}


def loads_strict(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)

    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected), label + ': invalid keys')


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def ratio(a, b):
    return a / b if b else None


def frames(value, count, objects):
    require(isinstance(value, list) and len(value) == count, 'incorrect frame count')
    for index, frame in enumerate(value):
        keys(frame, ('frame_index', 'tracks'), 'frame')
        require(type(frame['frame_index']) is int and frame['frame_index'] == index, 'frame order')
        tracks = frame['tracks']
        require(isinstance(tracks, list) and len(tracks) <= 8, 'track bound')
        seen, matched = set(), set()
        for track in tracks:
            keys(track, ('track_id', 'confirmed', 'matched_observation_ids'), 'compact track')
            tid = track['track_id']
            require(integer(tid, 1) and tid not in seen, 'duplicate/invalid track ID')
            seen.add(tid)
            require(type(track['confirmed']) is bool, 'confirmed must be boolean')
            observations = track['matched_observation_ids']
            require(isinstance(observations, list) and len(observations) <= 1, 'one-to-one match bound')
            for oid in observations:
                require(integer(oid) and index * objects <= oid < (index + 1) * objects,
                        'observation ID/frame mismatch')
                require(oid not in matched, 'duplicate observation ID')
                matched.add(oid)
        yield index, tracks


def finite_numbers(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_numbers(item)
    elif isinstance(value, list):
        for item in value:
            finite_numbers(item)
    elif type(value) in (int, float):
        require(math.isfinite(value), 'nonfinite payload number')


def reduce_fixture(payload):
    require(isinstance(payload, dict), 'payload must be an object')
    finite_numbers(payload)
    fixture = payload.get('regression_fixture')
    keys(fixture, ('schema_version', 'profile', 'trials', 'crossing'), 'fixture')
    require(type(fixture['schema_version']) is int and fixture['schema_version'] == 1
            and fixture['profile'] == PROFILE, 'fixture version/profile')
    trials = fixture['trials']
    require(isinstance(trials, list) and len(trials) == 256, 'incorrect seed count')
    metrics = dict(eligible_post_warmup_frames=256 * 158, associated_post_warmup_frames=0,
                   confirmed_associated_post_warmup_frames=0, final_associated_count=0,
                   original_confirmed_count=0, completed_original_tracks=0,
                   lost_original_with_final_confirmed_replacement=0,
                   observation_track_identity_transitions=0)
    errors, delays = [], []
    for offset, trial in enumerate(trials):
        keys(trial, ('seed', 'frames', 'final_truth', 'final_tracks'), 'trial')
        require(type(trial['seed']) is int and trial['seed'] == 9000 + offset, 'seed order')
        last_id, confirmation = None, None
        final_compact = None
        for index, tracks in frames(trial['frames'], 160, 1):
            final_compact = tracks
            for track in tracks:
                if track['track_id'] == 1 and track['confirmed'] and confirmation is None:
                    confirmation = index * 0.1
                if track['matched_observation_ids']:
                    if last_id is not None and last_id != track['track_id']:
                        metrics['observation_track_identity_transitions'] += 1
                    last_id = track['track_id']
                    if index >= 2:
                        metrics['associated_post_warmup_frames'] += 1
                        metrics['confirmed_associated_post_warmup_frames'] += int(track['confirmed'])
        if confirmation is not None:
            delays.append(confirmation)
        truth = trial['final_truth']
        require(isinstance(truth, list) and len(truth) == 4 and all(number(v) for v in truth), 'finite truth')
        expected_truth = [10 + 0.8 * 15.9, 30 - 0.3 * 15.9, 0.8, -0.3]
        require(all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                    for a, b in zip(truth, expected_truth)), 'fixed truth mismatch')
        final = trial['final_tracks']
        require(isinstance(final, list) and len(final) <= 8, 'final track bound')
        compact_by_id = {t['track_id']: t for t in final_compact}
        final_ids = set()
        for track in final:
            require(isinstance(track, dict), 'final track must be object')
            tid = track.get('track_id')
            require(integer(tid, 1) and tid in compact_by_id and tid not in final_ids, 'final track ID mismatch')
            final_ids.add(tid)
            require(type(track.get('schema_version')) is int and track['schema_version'] == 1, 'final track version')
            state, covariance = track.get('state'), track.get('covariance')
            require(isinstance(state, list) and len(state) == 4 and all(number(v) for v in state), 'finite final state')
            require(isinstance(covariance, list) and len(covariance) == 16
                    and all(number(v) for v in covariance), 'finite final covariance')
            matches = track.get('matched_observations')
            require(isinstance(matches, list) and all(integer(v) for v in matches)
                    and matches == compact_by_id[tid]['matched_observation_ids'], 'final match mismatch')
            if compact_by_id[tid]['matched_observation_ids']:
                errors.append(math.fsum((state[i] - truth[i]) ** 2 for i in range(2)))
        require(final_ids == set(compact_by_id), 'final track population mismatch')
        metrics['completed_original_tracks'] += int(1 in final_ids)
        if 1 not in final_ids:
            metrics['lost_original_with_final_confirmed_replacement'] += int(any(
                t['confirmed'] and t['matched_observation_ids'] for t in final_compact))
    metrics['original_confirmed_count'] = len(delays)
    metrics['original_never_confirmed_count'] = 256 - len(delays)
    metrics['mean_original_confirmation_delay_s'] = ratio(math.fsum(delays), len(delays))
    metrics['original_track_losses'] = 256 - metrics['completed_original_tracks']
    metrics['final_associated_count'] = len(errors)
    metrics['final_unassociated_count'] = 256 - len(errors)
    metrics['final_associated_position_rmse_m'] = math.sqrt(math.fsum(errors) / len(errors)) if errors else None
    confirmed = metrics['confirmed_associated_post_warmup_frames']
    metrics['confirmed_coverage_all_eligible_frames'] = confirmed / metrics['eligible_post_warmup_frames']
    metrics['confirmed_observation_coverage_after_warmup'] = ratio(confirmed, metrics['associated_post_warmup_frames'])
    crossing = fixture['crossing']
    keys(crossing, ('seed', 'frames'), 'crossing')
    require(type(crossing['seed']) is int and crossing['seed'] == 456, 'crossing seed')
    identities, votes = {}, {}
    associations, switches = 0, 0
    for _, tracks in frames(crossing['frames'], 150, 2):
        for track in tracks:
            tid = track['track_id']
            for oid in track['matched_observation_ids']:
                identity = oid % 2
                switches += int(tid in identities and identities[tid] != identity)
                identities[tid] = identity
                votes.setdefault(tid, [0, 0])[identity] += 1
                associations += 1
    metrics.update(crossing_associations=associations, crossing_identity_switches=switches,
                   crossing_majority_association_purity=ratio(sum(max(v) for v in votes.values()), associations))
    return metrics


def verify_report(payload, metrics):
    finite_numbers(payload)
    require(payload.get('status') == 'PASS_WITH_LIMITATIONS', 'report status')
    require(type(payload.get('schema_version')) is int and payload['schema_version'] == 2, 'report version')
    require(type(payload.get('seeds')) is int and payload['seeds'] == 256
            and type(payload.get('frames_per_seed')) is int and payload['frames_per_seed'] == 160, 'report experiment size')
    require('final_associated_position_rmse_all_seeds_m' not in payload
            and 'mean_confirmation_delay_s' not in payload, 'obsolete report field')
    crossing = payload.get('crossing')
    require(isinstance(crossing, dict), 'missing crossing report')
    require(crossing.get('deterministic_repeat') is True, 'crossing determinism failed')
    for key, expected in metrics.items():
        container = crossing if key.startswith('crossing_') else payload
        field = key.removeprefix('crossing_') if key.startswith('crossing_') else key
        require(field in container, 'missing report metric: ' + key)
        actual = container[field]
        if expected is None:
            require(actual is None, 'report mismatch: ' + key)
        elif type(expected) is int:
            require(type(actual) is int and actual == expected, 'report mismatch: ' + key)
        else:
            require(number(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), 'report mismatch: ' + key)


def validate_budget(budget):
    keys(budget, ('schema_version', 'profile', 'limits'), 'budget')
    require(type(budget['schema_version']) is int and budget['schema_version'] == 1 and budget['profile'] == PROFILE, 'budget version/profile')
    keys(budget['limits'], LIMITS, 'budget limits')
    for key, (_, _, maximum, whole) in LIMITS.items():
        value = budget['limits'][key]
        require(number(value) and value >= 0 and (not whole or type(value) is int)
                and (maximum is None or value <= maximum), 'invalid budget limit: ' + key)
    return budget


def evaluate_budget(metrics, budget):
    validate_budget(budget)
    checks = []
    for key, (metric, operator, _, _) in LIMITS.items():
        actual, bound = metrics.get(metric), budget['limits'][key]
        require(number(actual), 'missing/nonfinite metric: ' + metric)
        checks.append(dict(metric=metric, actual=actual, bound=bound, operator=operator,
                           passed=actual <= bound if operator == '<=' else actual >= bound))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--budget', required=True)
    parser.add_argument('--output')
    parser.add_argument('--fixture-output')
    args = parser.parse_args()
    result = dict(schema_version=1, profile=PROFILE, status='FAIL')
    try:
        budget_bytes = Path(args.budget).read_bytes()
        require(len(budget_bytes) <= 65536, 'budget size bound')
        budget = validate_budget(loads_strict(budget_bytes))
        result['budget_sha256'] = hashlib.sha256(budget_bytes).hexdigest()
        # File-backed subprocess output avoids holding an unchecked stream in memory.
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            child = subprocess.run([args.binary, '--regression-fixture'], stdout=output, stderr=errors, timeout=120)
            require(child.returncode == 0, 'fixture producer failed: exit ' + str(child.returncode))
            require(output.tell() <= MAX_BYTES, 'fixture size bound')
            output.seek(0)
            raw = output.read(MAX_BYTES + 1)
        result['fixture_sha256'] = hashlib.sha256(raw).hexdigest()
        payload = loads_strict(raw)
        metrics = reduce_fixture(payload)
        verify_report(payload, metrics)
        checks = evaluate_budget(metrics, budget)
        result.update(metrics=metrics, checks=checks,
                      status='PASS' if all(check['passed'] for check in checks) else 'FAIL',
                      limitations=['Deterministic fixture regression alarms, not statistical calibration or hardware accuracy.'])
        if args.fixture_output:
            Path(args.fixture_output).write_bytes(raw)
    except (ValueError, OSError, subprocess.SubprocessError, OverflowError) as error:
        result['error'] = str(error)[:2048]
    rendered = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        Path(args.output).write_text(rendered)
    print(rendered, end='')
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
