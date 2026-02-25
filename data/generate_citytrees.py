#!/usr/bin/env python3
"""
Generate synthetic cityTrees-style SLAM datasets.

The robot drives on a city grid (block length ~30-50m), making 90-degree turns.
Landmarks (trees) are scattered randomly along the streets.
Range-bearing observations are generated within sensor range.

Usage:
    python3 generate_citytrees.py <n_poses> <output.txt> [seed=42]

Examples:
    python3 generate_citytrees.py 800   cityTrees800_gen.txt
    python3 generate_citytrees.py 5000  cityTrees5k.txt
    python3 generate_citytrees.py 10000 cityTrees10k_gen.txt
    python3 generate_citytrees.py 50000 cityTrees50k.txt
"""
import sys
import math
import random

# --- Generation parameters (tuned to match cityTrees800) ---
DELTA        = 1.0          # time step [s]
NU_MEAN      = 1.0          # mean forward speed [m/step]
NU_STD       = 0.03         # speed noise std
OMEGA_STD    = 0.008        # straight-driving heading noise std [rad/step]
BLOCK_MIN    = 5            # minimum block length [steps]
BLOCK_MAX    = 75           # maximum block length [steps]

SENSOR_RANGE = 8.5          # max observable range [m]
SENSOR_BEARING_STD = 0.0    # bearing noise (0 = noiseless, matching existing data)
SENSOR_RANGE_STD   = 0.0    # range noise

# Landmark density: ~55 landmarks per 18000 m² (cityTrees800 scale)
LANDMARK_DENSITY = 3.0e-3   # landmarks per m²  (≈55 per 18125 m²)
LANDMARK_STRIP_WIDTH = 20   # place trees within this distance of path

TURN_NU_SCALE = 1.12        # speed boost during turn (matches data)

# Cap observations per landmark to keep edge count manageable.
# C(k,2) pairs: k=5 → 10, k=10 → 45, k=20 → 190.
# With ~16k landmarks and k=5, total edges ≈ 80k — fast solve.
# With ~16k landmarks and k=10, total edges ≈ 360k — manageable.
MAX_OBS_PER_LM = 10         # max times any single landmark is observed


def state_transition(x, y, theta, nu, omega, delta=1.0):
    if abs(omega) < 1e-10:
        nx = x + nu * math.cos(theta) * delta
        ny = y + nu * math.sin(theta) * delta
        nt = theta
    else:
        nx = x + nu / omega * (math.sin(theta + omega * delta) - math.sin(theta))
        ny = y + nu / omega * (-math.cos(theta + omega * delta) + math.cos(theta))
        nt = theta + omega * delta
    return nx, ny, nt


def generate(n_poses, seed=42):
    rng = random.Random(seed)

    # --- Phase 1: plan grid trajectory (turns at block ends) ---
    # Robot heads in 4 compass directions: 0, π/2, π, 3π/2
    # Start heading: π (west, matching cityTrees800 which turns ~180° at step 1)
    heading = math.pi
    turns = []        # list of (step, delta_heading) = where 90-degree turns happen
    step = 1          # step 0 is always straight, first turn planned from step 1

    while step < n_poses:
        block_len = rng.randint(BLOCK_MIN, BLOCK_MAX)
        step += block_len
        if step >= n_poses:
            break
        # Choose ±90° turn
        turn_dir = rng.choice([-1, 1])
        turns.append((step, turn_dir * math.pi / 2))
        step += 1  # the turn itself takes 1 step

    turn_set = {s: dh for s, dh in turns}

    # --- Phase 2: generate u (control inputs) ---
    us = []    # (nu, omega) for each step 1..n_poses-1
    heading = math.pi

    for step in range(1, n_poses):
        if step in turn_set:
            dh = turn_set[step]
            omega = dh / DELTA   # complete turn in 1 step
            nu = NU_MEAN * TURN_NU_SCALE + rng.gauss(0, NU_STD)
        else:
            omega = rng.gauss(0, OMEGA_STD)
            nu = NU_MEAN + rng.gauss(0, NU_STD)
            nu = max(0.85, min(1.65, nu))
        us.append((nu, omega))

    # --- Phase 3a: integrate ground-truth trajectory (used for obs generation) ---
    poses_gt = [(0.0, 0.0, 0.0)]
    for nu, omega in us:
        x, y, theta = poses_gt[-1]
        nx, ny, nt = state_transition(x, y, theta, nu, omega, DELTA)
        nt = (nt + math.pi) % (2 * math.pi) - math.pi
        poses_gt.append((nx, ny, nt))

    # --- Phase 3b: integrate dead-reckoning trajectory (noisy initial estimates) ---
    # Motion noise model: stds nn=0.19, no=0.001, on=0.13, oo=0.2
    # sigma_nu  ≈ sqrt(nn² * |nu| + no² * |omega|) ≈ 0.19 * sqrt(|nu|) per step
    # sigma_omega ≈ sqrt(on² * |nu| + oo² * |omega|) ≈ sqrt(0.13² + 0.2² * |omega|) per step
    NN, NO, ON, OO = 0.19, 0.001, 0.13, 0.2
    poses = [(0.0, 0.0, 0.0)]
    for nu, omega in us:
        x, y, theta = poses[-1]
        sigma_nu    = math.sqrt(NN**2 * abs(nu) + NO**2 * abs(omega) + 1e-9)
        sigma_omega = math.sqrt(ON**2 * abs(nu) + OO**2 * abs(omega) + 1e-9)
        nu_noisy    = nu    + rng.gauss(0, sigma_nu)
        omega_noisy = omega + rng.gauss(0, sigma_omega)
        nx, ny, nt = state_transition(x, y, theta, nu_noisy, omega_noisy, DELTA)
        nt = (nt + math.pi) % (2 * math.pi) - math.pi
        poses.append((nx, ny, nt))

    # --- Phase 4: place landmarks along trajectory ---
    # Find bounding box of ground-truth path
    xs = [p[0] for p in poses_gt]
    ys = [p[1] for p in poses_gt]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_min -= LANDMARK_STRIP_WIDTH
    x_max += LANDMARK_STRIP_WIDTH
    y_min -= LANDMARK_STRIP_WIDTH
    y_max += LANDMARK_STRIP_WIDTH

    area = (x_max - x_min) * (y_max - y_min)
    n_landmarks_target = max(10, int(area * LANDMARK_DENSITY))

    # Place landmarks randomly in the bounding box, then filter to keep only
    # those within LANDMARK_STRIP_WIDTH of any pose (visible ones)
    landmarks = {}   # landmark_id -> (mx, my)
    next_id = 6      # start from 6 matching existing datasets

    attempts = 0
    while len(landmarks) < n_landmarks_target and attempts < n_landmarks_target * 50:
        attempts += 1
        mx = rng.uniform(x_min, x_max)
        my = rng.uniform(y_min, y_max)

        # Check if this landmark is ever within sensor range of any ground-truth pose
        visible = False
        for px, py, _ in poses_gt[::max(1, len(poses_gt)//500)]:  # sample poses for speed
            if (mx - px)**2 + (my - py)**2 <= SENSOR_RANGE**2:
                visible = True
                break
        if not visible:
            continue

        landmarks[next_id] = (mx, my)
        next_id += 1

    # --- Phase 5: generate observations ---
    # Use ground-truth poses for observations (clean sensor model)
    # z step landmark_id range bearing
    observations = []   # (step, landmark_id, range, bearing)
    lm_obs_count = {lid: 0 for lid in landmarks}  # cap per-landmark obs count

    for step, (px, py, ptheta) in enumerate(poses_gt):
        for lid, (mx, my) in landmarks.items():
            if lm_obs_count[lid] >= MAX_OBS_PER_LM:
                continue
            dx = mx - px
            dy = my - py
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > SENSOR_RANGE:
                continue
            bearing = math.atan2(dy, dx) - ptheta
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi

            rng_obs = dist + rng.gauss(0, SENSOR_RANGE_STD) if SENSOR_RANGE_STD > 0 else dist
            b_obs   = bearing + rng.gauss(0, SENSOR_BEARING_STD) if SENSOR_BEARING_STD > 0 else bearing
            observations.append((step, lid, rng_obs, b_obs))
            lm_obs_count[lid] += 1

    return poses, us, observations


def write_log(output_path, poses, us, observations):
    n_poses = len(poses)
    obs_per_step = {}
    for step, lid, r, b in observations:
        if step not in obs_per_step:
            obs_per_step[step] = []
        obs_per_step[step].append((lid, r, b))

    with open(output_path, 'w') as f:
        f.write(f'delta {DELTA}\n')
        for step in range(n_poses):
            nu, omega = us[step - 1] if step > 0 else (0.0, 0.0)
            f.write(f'u {step} {nu} {omega}\n')
            px, py, ptheta = poses[step]
            f.write(f'x {step} {px} {py} {ptheta}\n')
            for lid, r, b in obs_per_step.get(step, []):
                f.write(f'z {step} {lid} {r} {b}\n')

    n_obs = len(observations)
    lm_ids = set(o[1] for o in observations)
    print(f'Generated {output_path}')
    print(f'  Poses: {n_poses}, Landmarks: {len(lm_ids)}, Observations: {n_obs}, dim: {n_poses*3}')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'Usage: python3 {sys.argv[0]} <n_poses> <output.txt> [seed=42]')
        sys.exit(1)
    n_poses = int(sys.argv[1])
    output  = sys.argv[2]
    seed    = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    poses, us, obs = generate(n_poses, seed=seed)
    write_log(output, poses, us, obs)
