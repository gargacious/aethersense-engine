#!/usr/bin/env python3
"""Independent stdlib direct DFT and rectangular-cell CFAR oracle; no production FFT calls."""
import cmath
import json
import math
import struct
import subprocess
import sys

fixture = json.loads(subprocess.check_output([sys.argv[1], "--oracle"], text=True))
errors = {}
for case in fixture["cases"]:
    config, signal = case["config"], case["signal"]
    options = config["iq_processing"]
    is_float = options["precision"] == "float32"
    q = (lambda x: struct.unpack("f", struct.pack("f", x))[0]) if is_float else float
    cq = lambda z: complex(q(z.real), q(z.imag))
    add = lambda a, b: cq(a + b)
    sub = lambda a, b: cq(a - b)
    scale = lambda a, b: complex(q(a.real * b), q(a.imag * b))
    divide = lambda a, b: complex(q(a.real / b), q(a.imag / b))
    power = lambda a: q(q(a.real * a.real) + q(a.imag * a.imag))
    n, m = config["fft_size"], config["chirps"]
    pi = q(math.pi)

    def window(k, size):
        angle = q(q(q(2 * pi) * k) / (size - 1))
        return q(.5 - q(.5 * q(math.cos(angle))))

    def dft(values):
        # Direct summation is independent of radix-2 butterflies/twiddle recurrence.
        return [cq(sum((z * cmath.exp(-2j * math.pi * k * j / len(values))
                        for j, z in enumerate(values)), 0j))
                for k in range(len(values))]

    raw = [cq(complex(*z)) for z in fixture["raw"]]
    ranges = []
    gain = q(q(n - 1) / q(2 * n))
    for ch in range(2):
        rows = []
        for j in range(m):
            row = raw[(ch*m+j)*n:(ch*m+j+1)*n]
            dc = 0j
            for z in row:
                dc = add(dc, z)
            dc = divide(dc, n)
            preprocessed = [scale(sub(z, dc), window(k, n)) for k, z in enumerate(row)]
            rows.append([divide(z, q(n * gain)) for z in dft(preprocessed)])
        ranges.append(rows)
    profile = []
    for k in range(n):
        total = 0.
        for j in range(m):
            total = q(total + q(power(ranges[0][j][k]) / m))
        profile.append(total)
    doppler = []
    dg = q(q(m-1) / q(2*m))
    for ch in range(2):
        columns = []
        for k in range(n):
            col = [scale(ranges[ch][j][k], window(j, m)) for j in range(m)]
            columns.append([divide(z, q(m * dg)) for z in dft(col)])
        doppler.append(columns)
    rd = [power(doppler[0][k][(row + m//2) % m]) for row in range(m) for k in range(n)]
    expected = {"i": [z.real for z in raw[:n]], "q": [z.imag for z in raw[:n]],
                "spectrum": [abs(z) for z in ranges[0][0]],
                "range_profile_power": profile, "range_doppler_power": rd}
    tolerance = 6e-6 if is_float else 2e-13
    maxima = {}
    for field, values in expected.items():
        observed = signal[field]
        assert len(values) == len(observed), field
        error = max(abs(a-b)/(1+abs(b)) for a, b in zip(observed, values))
        maxima[field] = error
        assert error < tolerance, (options["precision"], field, error)
    # Phase at near-null bins is ill-conditioned; compare reconstructed complex
    # amplitude, giving a fixed absolute error rather than weakening angular checks.
    phase_error = max(abs(amplitude*cmath.exp(1j*phase)-z) for amplitude,phase,z in
                      zip(signal["spectrum"],signal["phase"],ranges[0][0]))
    assert phase_error < tolerance, (options["precision"], "complex phase", phase_error)
    maxima["phase_reconstructed_complex"] = phase_error
    # Independent direct-cell CFAR summation on actual map: no prefix-table reuse.
    box = options["cfar_2d"]
    gy, gx = box["guard_rows"], box["guard_columns"]
    ry, rx = gy+box["training_rows"], gx+box["training_columns"]
    thresholds = [-1.] * (n*m)
    for y in range(ry, m-ry):
        for x in range(rx, n-rx):
            cells = [signal["range_doppler_power"][yy*n+xx]
                     for yy in range(y-ry, y+ry+1) for xx in range(x-rx, x+rx+1)
                     if abs(yy-y) > gy or abs(xx-x) > gx]
            alpha = len(cells)*math.expm1(-math.log(box["pfa"])/len(cells))
            thresholds[y*n+x] = alpha*math.fsum(cells)/len(cells)
    assert max(abs(a-b)/(1+abs(b)) for a,b in
               zip(signal["range_doppler_threshold_power"], thresholds)) < 2e-13
    # The same-map candidates are independently enumerated and IDs define order.
    peaks = []
    for y in range(1, m-1):
        for x in range(1, n//2):
            index=y*n+x
            value=signal["range_doppler_power"][index]
            if thresholds[index] < 0 or value <= thresholds[index]:
                continue
            neighbors=[(y+dy)*n+x+dx for dy in (-1,0,1) for dx in (-1,0,1) if dx or dy]
            if all(signal["range_doppler_power"][i] < value or
                   (signal["range_doppler_power"][i] == value and i > index) for i in neighbors):
                peaks.append(index)
    assert peaks, "Oracle fixture must exercise detection"
    assert [o["observation_id"] for o in case["observations"]] == [3*65536+p for p in peaks]
    for observation, index in zip(case["observations"], peaks):
        y,x=divmod(index,n)
        assert observation["range_m"] == x*signal["range_bin_m"]
        assert observation["radial_velocity_mps"] == (y-m//2)*signal["velocity_bin_mps"]
        j=(y+m//2)%m
        cross=doppler[1][x][j]*doppler[0][x][j].conjugate()
        sx=max(-1.,min(1.,cmath.phase(cross)/math.pi))
        assert abs(observation["x_m"]-observation["range_m"]*sx) < (1e-3 if is_float else 1e-9)
    errors[options["precision"]] = maxima
print(json.dumps({"status":"PASS", "oracle":"independent direct DFT / direct-cell CFAR / local maxima", "normalized_max_errors":errors}, indent=2))
