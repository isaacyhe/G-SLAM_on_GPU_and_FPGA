#!/usr/bin/env python3
"""
compare_poses.py — Run all 6 PCG variants on a dataset and compare final poses.

Usage:
    python3 scripts/compare_poses.py [datafile] [rounds]

Defaults:
    datafile = data/cityTrees10k.log.txt
    rounds   = 20

GPU variants use embedded Python draw (GSLAM_draw.py shim).
CPU/FPGA variants write poses via POSE_DUMP_FILE env var directly.

Reference: cpu_pcg_fp64
Reports per-variant RMSE for x, y, theta vs reference.
"""

import os
import sys
import shutil
import subprocess
import math

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIM_SRC  = os.path.join(REPO_ROOT, "scripts", "pose_dump_draw.py")

# SYCL runtime libs needed for FPGA variants
SYCL_LIB_PATH = "/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/umf/1.0/lib"

VARIANTS = [
    # (label,           src_dir,          binary_name,      needs_shim, needs_sycl)
    ("cpu_pcg_fp64",    "cpu_pcg_fp64",   "GSLAM",          False,      False),
    ("cpu_pcg_fp32",    "cpu_pcg_fp32",   "GSLAM",          False,      False),
    ("gpu_pcg_fp64",    "gpu_pcg_fp64",   "GSLAM.cuda",     True,       False),
    ("gpu_pcg_fp32",    "gpu_pcg_fp32",   "GSLAM.cuda",     True,       False),
    ("fpga_pcg_fp64",   "fpga_pcg_fp64",  "GSLAM.cpu_test", False,      True),
    ("fpga_pcg_fp32",   "fpga_pcg_fp32",  "GSLAM.cpu_test", False,      True),
]

REFERENCE = "cpu_pcg_fp64"


def run_variant(label, src_dir, binary_name, needs_shim, needs_sycl,
                datafile, rounds, dump_dir):
    src_path    = os.path.join(REPO_ROOT, "src", src_dir)
    binary_path = os.path.join(src_path, binary_name)
    dump_file   = os.path.join(dump_dir, f"{label}.txt")

    if not os.path.isfile(binary_path):
        print(f"  [{label}] SKIP — binary not found: {binary_path}")
        return None

    # GPU variants use embedded Python draw mechanism
    if needs_shim:
        shutil.copy2(SHIM_SRC, os.path.join(src_path, "GSLAM_draw.py"))

    env = os.environ.copy()
    env["POSE_DUMP_FILE"] = dump_file

    if needs_sycl:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = SYCL_LIB_PATH + (":" + existing if existing else "")

    cmd = [binary_path, "1", "0.1", str(rounds), "1", datafile]
    print(f"  [{label}] running {rounds} rounds ...", flush=True)
    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
        cwd=src_path, env=env,
        timeout=600
    )
    if result.returncode != 0:
        print(f"  [{label}] FAILED (exit {result.returncode})")
        print(result.stderr[-2000:] if result.stderr else "")
        return None

    # Print last few lines of stdout (convergence info)
    stdout_lines = result.stdout.strip().splitlines()
    for line in stdout_lines[-5:]:
        print(f"    {line}")

    if not os.path.isfile(dump_file):
        print(f"  [{label}] WARN — no pose dump written (draw not called?)")
        return None

    return dump_file


def load_poses(path):
    """Returns list of (step, x, y, theta)."""
    poses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            step, x, y, theta = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            poses.append((step, x, y, theta))
    return poses


def rmse(ref_poses, cmp_poses):
    """Compute per-component and combined RMSE. Aligns by step index."""
    ref_by_step = {s: (x, y, t) for s, x, y, t in ref_poses}
    cmp_by_step = {s: (x, y, t) for s, x, y, t in cmp_poses}

    steps = sorted(set(ref_by_step) & set(cmp_by_step))
    if not steps:
        return None

    sx2 = sy2 = st2 = 0.0
    n = len(steps)
    for s in steps:
        rx, ry, rt = ref_by_step[s]
        cx, cy, ct = cmp_by_step[s]
        sx2 += (rx - cx) ** 2
        sy2 += (ry - cy) ** 2
        # Wrap theta difference to [-pi, pi]
        dth = (rt - ct + math.pi) % (2 * math.pi) - math.pi
        st2 += dth ** 2

    rmse_x = math.sqrt(sx2 / n)
    rmse_y = math.sqrt(sy2 / n)
    rmse_t = math.sqrt(st2 / n)
    rmse_combined = math.sqrt((sx2 + sy2 + st2) / (3 * n))
    return rmse_x, rmse_y, rmse_t, rmse_combined, n


def main():
    datafile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "data", "cityTrees10k.log.txt")
    rounds   = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    # Resolve to absolute path so subprocess sees it regardless of cwd
    datafile = os.path.abspath(datafile)

    if not os.path.isfile(datafile):
        print(f"ERROR: data file not found: {datafile}")
        sys.exit(1)

    dump_dir = os.path.join(REPO_ROOT, "scripts", "_pose_dumps")
    os.makedirs(dump_dir, exist_ok=True)

    print(f"Dataset : {datafile}")
    print(f"Rounds  : {rounds}")
    print()

    dump_files = {}
    for label, src_dir, binary_name, needs_shim, needs_sycl in VARIANTS:
        path = run_variant(label, src_dir, binary_name, needs_shim, needs_sycl,
                           datafile, rounds, dump_dir)
        if path:
            dump_files[label] = path
        print()

    # Load reference
    if REFERENCE not in dump_files:
        print(f"ERROR: reference variant '{REFERENCE}' did not produce output.")
        sys.exit(1)

    ref_poses = load_poses(dump_files[REFERENCE])
    print(f"Reference: {REFERENCE} — {len(ref_poses)} poses")
    print()

    # Print comparison table
    header = f"{'Variant':<20}  {'RMSE_x':>10}  {'RMSE_y':>10}  {'RMSE_θ':>10}  {'RMSE_comb':>10}  {'Poses':>6}"
    print(header)
    print("-" * len(header))

    for label, _, _, _, _ in VARIANTS:
        if label == REFERENCE:
            print(f"{label:<20}  {'(reference)':>10}")
            continue
        if label not in dump_files:
            print(f"{label:<20}  {'SKIPPED':>10}")
            continue
        cmp_poses = load_poses(dump_files[label])
        r = rmse(ref_poses, cmp_poses)
        if r is None:
            print(f"{label:<20}  {'NO OVERLAP':>10}")
        else:
            rx, ry, rt, rc, n = r
            print(f"{label:<20}  {rx:>10.4f}  {ry:>10.4f}  {rt:>10.4f}  {rc:>10.4f}  {n:>6}")

    print()
    print(f"Dump files in: {dump_dir}")


if __name__ == "__main__":
    main()
