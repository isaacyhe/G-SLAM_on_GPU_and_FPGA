#!/usr/bin/env python3
"""
Convert SLAM++ format to G-SLAM log format.

SLAM++ input:
  ODOMETRY from to dx dy dtheta info...
  LANDMARK pose_id landmark_id range bearing info...

G-SLAM log output:
  delta 1.0
  u step nu omega
  x step x y theta
  z step landmark_id range bearing
"""
import sys
import math

def state_transition(x, y, theta, nu, omega, delta=1.0):
    if abs(omega) < 1e-10:
        nx = x + nu * math.cos(theta) * delta
        ny = y + nu * math.sin(theta) * delta
        nt = theta + omega * delta
    else:
        nx = x + nu / omega * (math.sin(theta + omega * delta) - math.sin(theta))
        ny = y + nu / omega * (-math.cos(theta + omega * delta) + math.cos(theta))
        nt = theta + omega * delta
    return nx, ny, nt

def convert(input_path, output_path, delta=1.0):
    odometry = {}   # from_id -> (dx, dy, dtheta)
    landmarks = []  # list of (pose_id, landmark_id, range, bearing)

    with open(input_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'ODOMETRY':
                # ODOMETRY from to dx dy dtheta info...
                from_id = int(parts[1])
                dx, dy, dtheta = float(parts[3]), float(parts[4]), float(parts[5])
                odometry[from_id] = (dx, dy, dtheta)
            elif parts[0] == 'LANDMARK':
                # LANDMARK pose_id landmark_id range bearing info...
                pose_id = int(parts[1])
                landmark_id = int(parts[2])
                rng = float(parts[3])
                bearing = float(parts[4])
                landmarks.append((pose_id, landmark_id, rng, bearing))

    # Build sorted list of pose IDs (0-based consecutive)
    n_poses = max(odometry.keys()) + 2  # max from_id + 1 (to_id) + 1

    # Forward-integrate poses
    poses = {}   # step -> (x, y, theta)
    us = {}      # step -> (nu, omega)

    poses[0] = (0.0, 0.0, 0.0)
    for step in range(n_poses - 1):
        if step not in odometry:
            # Gap — use identity
            poses[step + 1] = poses[step]
            us[step + 1] = (0.0, 0.0)
            continue
        dx, dy, dtheta = odometry[step]
        nu = math.sqrt(dx * dx + dy * dy)
        omega = dtheta
        x, y, theta = poses[step]
        nx, ny, nt = state_transition(x, y, theta, nu, omega, delta)
        poses[step + 1] = (nx, ny, nt)
        us[step + 1] = (nu, omega)

    with open(output_path, 'w') as out:
        out.write(f'delta {delta}\n')
        for step in range(n_poses):
            nu, omega = us.get(step, (0.0, 0.0))
            out.write(f'u {step} {nu} {omega}\n')
            x, y, theta = poses[step]
            out.write(f'x {step} {x} {y} {theta}\n')
        for pose_id, landmark_id, rng, bearing in landmarks:
            out.write(f'z {pose_id} {landmark_id} {rng} {bearing}\n')

    print(f'Converted {input_path} -> {output_path}')
    print(f'  Poses: {n_poses}, Observations: {len(landmarks)}, dim: {n_poses*3}')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'Usage: python3 {sys.argv[0]} input.txt output.log.txt [delta=1.0]')
        sys.exit(1)
    delta = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    convert(sys.argv[1], sys.argv[2], delta)
