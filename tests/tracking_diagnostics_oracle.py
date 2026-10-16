#!/usr/bin/env python3
"""Independent generic matrix inverse oracle; no NumPy or estimator truth feedback."""
import json, math, subprocess, sys
fixture=json.loads(subprocess.check_output([sys.argv[1], '--oracle'],text=True))
def inverse(matrix):
    n=len(matrix); a=[list(row)+[float(i==j) for j in range(n)] for i,row in enumerate(matrix)]
    for i in range(n):
        pivot=max(range(i,n),key=lambda row:abs(a[row][i])); a[i],a[pivot]=a[pivot],a[i]
        scale=a[i][i]; assert scale>0
        a[i]=[x/scale for x in a[i]]
        for k in range(n):
            if k!=i:
                factor=a[k][i]; a[k]=[x-factor*y for x,y in zip(a[k],a[i])]
    return [row[n:] for row in a]
def distance(residual,S):
    inv=inverse(S)
    return sum(residual[i]*inv[i][j]*residual[j] for i in range(2) for j in range(2))
def near(a,b): assert math.isclose(a,b,rel_tol=2e-13,abs_tol=2e-13),(a,b)
c=fixture['correlated']; near(c['d2'],distance([2,-3],[[4.5,1.5],[1.5,5]]))
match=fixture['report']['matches'][0]
# Independent CV prediction from birth P=diag(1,1,25,25), dt=.1, Qpos=.25*dt^4.
S=1+25*.1**2+.25*.1**4+1
for a,b in zip(match['innovation_covariance_m2'],[S,0,0,S]):near(a,b)
for a,b in zip(match['innovation_m'],[1,.5]):near(a,b)
near(match['mahalanobis_squared'],distance([1,.5],[[S,0],[0,S]]))
assert fixture['report']['gate_pass_pairs']==fixture['report']['selected_associations']==1
print('PASS independent generic-inverse oracle: correlated innovation and actual pre-update Tracker report; abs/rel 2e-13')
