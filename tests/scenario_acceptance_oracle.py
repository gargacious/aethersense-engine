#!/usr/bin/env python3
"""Independent analytic trajectories/window counts and posthoc evaluation; no calibration claim."""
import cmath
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

binary=Path(sys.argv[1]).resolve()
completed=subprocess.run([str(binary)],text=True,capture_output=True,check=True)
report=json.loads(completed.stdout)
assert report['status']=='passed'

def trajectory(t,stationary,linear,rate):
    origin=complex(-25,35)
    velocity=complex(1.2,-.15)
    if t<stationary:
        return origin,0j
    elapsed=t-stationary
    if elapsed<linear:
        return origin+velocity*elapsed,velocity
    elapsed-=linear
    half=rate*elapsed/2
    # Centered sinc integral avoids production's sin/cos-difference divided by rate.
    sinc=math.sin(half)/half if half else 1.0
    integral=elapsed*sinc*cmath.exp(1j*half)
    return origin+velocity*linear+velocity*integral,velocity*cmath.exp(2j*half)

maximum_truth_error=0.0
for sample in report['motion_samples']:
    position,velocity=trajectory(sample['t_s'],sample['stationary_s'],sample['linear_s'],sample['turn_rate_rad_s'])
    truth=sample['truth']
    actual=[truth['x_m'],truth['y_m'],truth['vx_mps'],truth['vy_mps']]
    expected=[position.real,position.imag,velocity.real,velocity.imag]
    for found,wanted in zip(actual,expected):
        error=abs(found-wanted)
        maximum_truth_error=max(maximum_truth_error,error)
        assert error<=1e-10+1e-12*abs(wanted),(sample,found,wanted)

# Sylvester leading principal minors use permutation determinants, independently
# from the native Cholesky check. No probabilistic NEES coverage is inferred.
def determinant(matrix):
    total=0.0
    for permutation in itertools.permutations(range(len(matrix))):
        inversions=sum(permutation[i]>permutation[j] for i in range(len(matrix)) for j in range(i+1,len(matrix)))
        term=(-1.0 if inversions%2 else 1.0)
        for i,column in enumerate(permutation):term*=matrix[i][column]
        total+=term
    return total

frames=report['scenario_run']['frames']
assert len(frames)==60
squared=0.0
matched=0
missing_original=0
transitions=0
previous=0
covariance_checks=0
for expected_sequence,frame in enumerate(frames):
    sequence=frame['sequence_no']
    assert sequence==expected_sequence
    position,velocity=trajectory(sequence*.1,.5,.5,.2)
    truth=frame['evaluation_truth']
    for found,wanted in zip([truth['x_m'],truth['y_m'],truth['vx_mps'],truth['vy_mps']], [position.real,position.imag,velocity.real,velocity.imag]):
        assert abs(found-wanted)<=1e-10+1e-12*abs(wanted)
    occluded=10<=sequence<15
    burst=20<=sequence<23
    noisy=30<=sequence<35
    observations=frame['observations']
    assert len(observations)==(0 if occluded else 1)+(10 if burst else 0)
    if not occluded:
        assert observations[0]['covariance_m2']==[16 if noisy else 1,0,0,16 if noisy else 1]
    assert sum(o['confidence']==.3 for o in observations)==(10 if burst else 0)
    for track in frame['tracks']:
        assert all(math.isfinite(value) for value in track['state'])
        p=track['covariance']
        assert len(p)==16 and all(math.isfinite(v) for v in p)
        for i in range(4):
            for j in range(4):assert abs(p[i*4+j]-p[j*4+i])<1e-9
        for size in range(1,5):assert determinant([[p[i*4+j] for j in range(size)] for i in range(size)])>0
        covariance_checks+=1
    missing_original+=not any(t['track_id']==1 for t in frame['tracks'])
    if frame['tracks']:
        nearest=min(frame['tracks'],key=lambda track:math.hypot(track['state'][0]-position.real,track['state'][1]-position.imag))
        distance=math.hypot(nearest['state'][0]-position.real,nearest['state'][1]-position.imag)
        if distance<=10:
            matched+=1;squared+=distance*distance
            if previous and previous!=nearest['track_id']:transitions+=1
            previous=nearest['track_id']
run=report['scenario_run']
assert run['matched_evaluation_frames']==matched
assert run['original_id_missing_frames']==missing_original
assert run['nearest_track_identity_transitions']==transitions
rmse=math.sqrt(squared/matched) if matched else None
assert rmse is not None and abs(run['conditional_nearest_position_rmse_m']-rmse)<1e-10
last=-1
accepted=[]
rejected=[]
for source_frame in [0,2,1,2,4,3,5]:
    if source_frame<=last:rejected.append(source_frame)
    else:accepted.append(source_frame);last=source_frame
assert accepted==[0,2,4,5] and rejected==[1,2,3]
assert report['delivery']['accepted']==len(accepted) and report['delivery']['rejected']==len(rejected)
assert report['delivery']['rejected_source_frames']==rejected
assert report['delivery']['capacity']==8 and report['delivery']['production_delivery_scheduler'] is False
print(json.dumps({'schema_version':1,'status':'passed','motion_samples':len(report['motion_samples']),
 'maximum_analytic_truth_error':maximum_truth_error,'truth_tolerance':'abs1e-10+rel1e-12',
 'disturbance_frames':len(frames),'independent_covariance_checks':covariance_checks,
 'conditional_nearest_position_rmse_m':rmse,'matched_evaluation_frames':matched,
 'original_id_missing_frames':missing_original,'nearest_track_identity_transitions':transitions,
 'delayed_policy_accepted':accepted,'delayed_policy_rejected':rejected,
 'scope':'finite deterministic fixture, posthoc conditional evaluation; not statistical calibration or production delayed delivery'},indent=2))
