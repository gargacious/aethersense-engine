#!/usr/bin/env python3
"""Independent matrix prediction/update for the seed9003 two-candidate cascade."""
import json
import math
import subprocess
import sys

def transpose(a):
    return list(map(list, zip(*a)))

def mul(a, b):
    return [[math.fsum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]

def main():
    fixture = json.loads(subprocess.run([sys.argv[1], '--cascade-fixture'], check=True,
                                       text=True, capture_output=True).stdout)
    f = [[1.,0.,.1,0.], [0.,1.,0.,.1], [0.,0.,1.,0.], [0.,0.,0.,1.]]
    z = [fixture['observation']['x_m'], fixture['observation']['y_m']]
    candidates = []
    predicted = {}
    for order, track in enumerate(fixture['before']):
        p = [track['covariance'][i:i+4] for i in range(0, 16, 4)]
        p = mul(mul(f, p), transpose(f))
        for axis in (0, 1):
            p[axis][axis] += .25 * .1**4
            p[axis][axis+2] += .5 * .1**3
            p[axis+2][axis] += .5 * .1**3
            p[axis+2][axis+2] += .1**2
        x = [row[0] for row in mul(f, [[v] for v in track['state']])]
        s = [[p[0][0]+.25,p[0][1]], [p[1][0],p[1][1]+.25]]
        det = s[0][0]*s[1][1]-s[0][1]*s[1][0]
        inv = [[s[1][1]/det,-s[0][1]/det],[-s[1][0]/det,s[0][0]/det]]
        residual = [a-b for a,b in zip(z,x)]
        distance = math.fsum(residual[i]*inv[i][j]*residual[j] for i in range(2) for j in range(2))
        confirmed = track['lifecycle'] != 'tentative'
        if distance < 9.21034:
            candidates.append((not confirmed, distance, order, track['track_id']))
        predicted[track['track_id']] = (x,p,inv,residual)
    selected = sorted(candidates)[0][3]
    pure_nearest = min(candidates, key=lambda item: item[1])[3]
    assert selected == 1 and pure_nearest == 2
    actual = next(t for t in fixture['after'] if t['track_id'] == selected)
    assert actual['matched_observations'] == [22]
    x,p,inv,residual = predicted[selected]
    k = mul([row[:2] for row in p], inv)
    expected_x = [v+math.fsum(a*b for a,b in zip(row,residual)) for v,row in zip(x,k)]
    # Algebraic covariance update (I-KH)P, independent of production Joseph form.
    ikh = [[float(i==j)-(k[i][j] if j<2 else 0.) for j in range(4)] for i in range(4)]
    expected_p = mul(ikh,p)
    state_error = max(abs(a-b) for a,b in zip(expected_x,actual['state']))
    covariance_error = max(abs(expected_p[i][j]-actual['covariance'][i*4+j]) for i in range(4) for j in range(4))
    assert state_error < 1e-10 and covariance_error < 1e-10
    print(json.dumps({'status':'PASS', 'fixture_seed':9003, 'frame':22,
                      'cascade_selected':selected, 'pure_distance_selected':pure_nearest,
                      'candidate_distances':{str(v[3]):v[1] for v in candidates},
                      'max_state_absolute_error':state_error,
                      'max_covariance_absolute_error':covariance_error}))

if __name__ == '__main__':
    main()
