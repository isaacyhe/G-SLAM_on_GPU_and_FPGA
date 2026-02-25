"""
pose_dump_draw.py — drop-in replacement for GSLAM_draw.py.

When GSLAM runs with toDraw>=1 it calls draw(...) via embedded Python.
This version writes final poses to a file instead of showing a plot.

The output file path is read from the environment variable POSE_DUMP_FILE.
Each line: <step> <x> <y> <theta>
"""

import os

def draw(hat_xs_steps, hat_xs, zlist_steps, zlist_landmark_ids, zs, ms_indexes, mses):
    out_path = os.environ.get("POSE_DUMP_FILE", "/tmp/pose_dump.txt")
    n = len(hat_xs_steps)
    with open(out_path, "w") as f:
        for i in range(n):
            step  = hat_xs_steps[i]
            x     = hat_xs[i * 3]
            y     = hat_xs[i * 3 + 1]
            theta = hat_xs[i * 3 + 2]
            f.write(f"{step} {x:.17g} {y:.17g} {theta:.17g}\n")
