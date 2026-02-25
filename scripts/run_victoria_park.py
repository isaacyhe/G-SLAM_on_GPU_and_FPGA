#!/usr/bin/env python3
"""
run_victoria_park.py — Run all PCG and GJ variants on victoria-park datasets,
generate trajectory comparison plots, and report convergence rounds.

Usage:
    python3 scripts/run_victoria_park.py

PCG variants run on full victoria-park.log.txt (7120 poses)
GJ  variants run on victoria-park-100.log.txt  (100 poses)
"""

import os
import sys
import shutil
import subprocess
import math
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIM_SRC  = os.path.join(REPO_ROOT, "scripts", "pose_dump_draw.py")
SYCL_LIB  = "/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/umf/1.0/lib"

PCG_DATA = os.path.join(REPO_ROOT, "data", "victoria-park.log.txt")
GJ_DATA  = os.path.join(REPO_ROOT, "data", "victoria-park-100.log.txt")
DUMP_DIR = os.path.join(REPO_ROOT, "scripts", "_pose_dumps_vp")

PCG_VARIANTS = [
    # (label,           src_dir,          binary,           shim,  sycl)
    ("cpu_pcg_fp64",   "cpu_pcg_fp64",   "GSLAM",          False, False),
    ("cpu_pcg_fp32",   "cpu_pcg_fp32",   "GSLAM",          False, False),
    ("fpga_pcg_fp64",  "fpga_pcg_fp64",  "GSLAM.cpu_test", False, True),
    ("fpga_pcg_fp32",  "fpga_pcg_fp32",  "GSLAM.cpu_test", False, True),
    ("gpu_pcg_fp64",   "gpu_pcg_fp64",   "GSLAM.cuda",     True,  False),
    ("gpu_pcg_fp32",   "gpu_pcg_fp32",   "GSLAM.cuda",     True,  False),
]

GJ_VARIANTS = [
    # (label,            src_dir,          binary,      shim,  sycl)
    ("cpu_gj_fp64",    "cpu_gj_fp64",    "GSLAM",     False, True),   # icpx-built
    ("cpu_gj_fp32",    "cpu_gj_fp32",    "GSLAM",     False, True),
    ("cpu_gj_mixed",   "cpu_gj_mixed",   "GSLAM",     False, True),   # icpx-built
    ("fpga_gj_fp64",   "fpga_gj_fp64",   "GSLAM",     False, True),
    ("fpga_gj_fp32",   "fpga_gj_fp32",   "GSLAM",     False, True),
    ("fpga_gj_mixed",  "fpga_gj_mixed",  "GSLAM",     False, True),
    ("gpu_gj_fp64",    "gpu_gj_fp64",    "GSLAM.cuda", True, False),
    ("gpu_gj_fp32",    "gpu_gj_fp32",    "GSLAM.cuda", True, False),
    ("gpu_gj_mixed",   "gpu_gj_mixed",   "GSLAM.cuda", True, False),
]


def run_variant(label, src_dir, binary, shim, sycl, datafile, rounds, dump_dir):
    src_path = os.path.join(REPO_ROOT, "src", src_dir)
    binpath  = os.path.join(src_path, binary)
    dump_file = os.path.join(dump_dir, f"{label}.txt")

    if not os.path.isfile(binpath):
        print(f"  [{label}] SKIP — binary not found")
        return None, None

    if shim:
        shutil.copy2(SHIM_SRC, os.path.join(src_path, "GSLAM_draw.py"))

    env = os.environ.copy()
    env["POSE_DUMP_FILE"] = dump_file
    if sycl:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = SYCL_LIB + (":" + existing if existing else "")

    cmd = [binpath, "1", "0.1", str(rounds), "1", datafile]
    print(f"  [{label}] running up to {rounds} rounds ...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=src_path, env=env, timeout=600)
    if result.returncode != 0:
        print(f"  [{label}] FAILED (exit {result.returncode})")
        if result.stderr:
            print(result.stderr[-1000:])
        return None, None

    stdout = result.stdout
    # Parse convergence round from stdout
    conv_round = parse_convergence_round(stdout)

    # Print last 4 lines of output
    for line in stdout.strip().splitlines()[-4:]:
        print(f"    {line}")

    if not os.path.isfile(dump_file):
        print(f"  [{label}] WARN — no pose dump written")
        return None, conv_round

    return dump_file, conv_round


def parse_convergence_round(stdout):
    """Extract last round executed from GSLAM stdout.
    Format: 'N rounds executed: diff_value'
    The last printed round is the convergence/plateau round."""
    last_round = None
    for line in stdout.splitlines():
        # Match "N rounds executed: value"
        m = re.match(r'^\s*(\d+)\s+rounds\s+executed', line)
        if m:
            last_round = int(m.group(1))
    return last_round


def load_poses(path):
    poses = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                step = int(parts[0])
                x, y, theta = float(parts[1]), float(parts[2]), float(parts[3])
                poses[step] = (x, y, theta)
    return poses


def drift(poses):
    """End-to-end drift: distance from first to last pose."""
    if len(poses) < 2:
        return 0.0
    steps = sorted(poses.keys())
    x0, y0, _ = poses[steps[0]]
    xn, yn, _ = poses[steps[-1]]
    return math.sqrt((xn - x0)**2 + (yn - y0)**2)


def load_initial_trajectory(datafile):
    """Load dead-reckoning (initial) poses from the log file (x lines)."""
    poses = {}
    with open(datafile) as f:
        for line in f:
            if line.startswith("x "):
                parts = line.split()
                step = int(parts[1])
                x, y, theta = float(parts[2]), float(parts[3]), float(parts[4])
                poses[step] = (x, y, theta)
    return poses


def plot_trajectories(dump_files, conv_rounds, initial_poses, title, output_path,
                      groups=None):
    """Generate trajectory comparison plot with two panels:
    Left: dead-reckoning overview. Right: zoomed SLAM result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22'
    ]

    # Load all optimized pose arrays
    all_poses = {}
    for label, dfile in dump_files.items():
        all_poses[label] = load_poses(dfile)

    # Compute bounding box of SLAM result (for zoom panel)
    all_xs, all_ys = [], []
    for poses in all_poses.values():
        steps = sorted(poses.keys())
        all_xs += [poses[s][0] for s in steps]
        all_ys += [poses[s][1] for s in steps]
    pad = 10
    slam_xmin, slam_xmax = min(all_xs) - pad, max(all_xs) + pad
    slam_ymin, slam_ymax = min(all_ys) - pad, max(all_ys) + pad

    fig, (ax_dr, ax_slam) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ---- LEFT PANEL: dead-reckoning overview ----
    if initial_poses:
        steps = sorted(initial_poses.keys())
        xs = [initial_poses[s][0] for s in steps]
        ys = [initial_poses[s][1] for s in steps]
        ax_dr.plot(xs, ys, color='steelblue', linewidth=1.0, alpha=0.7,
                   label='Dead-reckoning')
        ax_dr.plot(xs[0], ys[0], 'go', markersize=8, label='Start', zorder=5)
        ax_dr.plot(xs[-1], ys[-1], 'rs', markersize=8, label='End (DR)', zorder=5)

    # Show SLAM result bounding box on the DR plot
    from matplotlib.patches import Rectangle
    rect = Rectangle((slam_xmin, slam_ymin),
                      slam_xmax - slam_xmin, slam_ymax - slam_ymin,
                      linewidth=2, edgecolor='red', facecolor='none',
                      linestyle='--', label='SLAM region (see right)')
    ax_dr.add_patch(rect)

    ax_dr.set_title("Dead-reckoning (odometry only)", fontsize=11)
    ax_dr.set_xlabel("X (m)")
    ax_dr.set_ylabel("Y (m)")
    ax_dr.legend(loc='best', fontsize=9)
    ax_dr.set_aspect('equal', adjustable='datalim')
    ax_dr.grid(True, alpha=0.3)

    # ---- RIGHT PANEL: zoomed SLAM trajectories ----
    # Draw dead-reckoning faintly in background for context
    if initial_poses:
        steps = sorted(initial_poses.keys())
        xs = [initial_poses[s][0] for s in steps]
        ys = [initial_poses[s][1] for s in steps]
        ax_slam.plot(xs, ys, color='lightgray', linewidth=0.8, linestyle='--',
                     alpha=0.4, label='Dead-reckoning (clipped)', zorder=1)

    for i, (label, poses) in enumerate(all_poses.items()):
        steps = sorted(poses.keys())
        xs = [poses[s][0] for s in steps]
        ys = [poses[s][1] for s in steps]
        color = colors[i % len(colors)]
        conv = conv_rounds.get(label)
        conv_str = f"@r{conv}" if conv else ""
        d = drift(poses)
        lbl = f"{label} {conv_str}  drift={d:.1f}m"
        ax_slam.plot(xs, ys, color=color, linewidth=1.0, alpha=0.75,
                     label=lbl, zorder=2 + i)

    # Mark common start
    ax_slam.plot(0, 0, 'k^', markersize=10, label='Start (0,0)', zorder=20)

    ax_slam.set_xlim(slam_xmin, slam_xmax)
    ax_slam.set_ylim(slam_ymin, slam_ymax)
    ax_slam.set_title("SLAM-optimized trajectories (zoomed)", fontsize=11)
    ax_slam.set_xlabel("X (m)")
    ax_slam.set_ylabel("Y (m)")
    ax_slam.legend(loc='best', fontsize=8, framealpha=0.9)
    ax_slam.set_aspect('equal', adjustable='datalim')
    ax_slam.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def run_group(group_name, variants, datafile, max_rounds, dump_dir):
    print(f"\n{'='*60}")
    print(f"  {group_name}")
    print(f"  Dataset: {datafile}")
    print(f"{'='*60}")

    dump_files = {}
    conv_rounds = {}
    for label, src_dir, binary, shim, sycl in variants:
        dfile, conv = run_variant(label, src_dir, binary, shim, sycl,
                                  datafile, max_rounds, dump_dir)
        if dfile:
            dump_files[label] = dfile
            conv_rounds[label] = conv
        print()

    return dump_files, conv_rounds


def main():
    os.makedirs(DUMP_DIR, exist_ok=True)

    # ---------- PCG: full victoria-park ----------
    pcg_dumps, pcg_conv = run_group(
        "PCG variants — victoria-park (7120 poses)",
        PCG_VARIANTS, PCG_DATA, max_rounds=200, dump_dir=DUMP_DIR
    )

    # ---------- GJ: victoria-park-100 ----------
    gj_dumps, gj_conv = run_group(
        "GJ variants — victoria-park-100 (100 poses)",
        GJ_VARIANTS, GJ_DATA, max_rounds=200, dump_dir=DUMP_DIR
    )

    # ---------- Summary ----------
    print("\n" + "="*70)
    print("CONVERGENCE SUMMARY")
    print("="*70)
    print(f"\n{'Variant':<22}  {'Conv. Round':>12}  {'Drift (m)':>12}")
    print("-" * 52)
    for label, dfile in pcg_dumps.items():
        poses = load_poses(dfile)
        d = drift(poses)
        conv = pcg_conv.get(label, "?")
        print(f"{label:<22}  {str(conv):>12}  {d:>12.2f}")
    print()
    for label, dfile in gj_dumps.items():
        poses = load_poses(dfile)
        d = drift(poses)
        conv = gj_conv.get(label, "?")
        print(f"{label:<22}  {str(conv):>12}  {d:>12.2f}")

    # ---------- Plots ----------
    print("\nGenerating plots ...")
    try:
        initial_vp  = load_initial_trajectory(PCG_DATA)
        initial_100 = load_initial_trajectory(GJ_DATA)

        out_pcg = os.path.join(REPO_ROOT, "scripts", "trajectory_pcg_victoria_park.png")
        out_gj  = os.path.join(REPO_ROOT, "scripts", "trajectory_gj_victoria_park_100.png")

        if pcg_dumps:
            plot_trajectories(
                pcg_dumps, pcg_conv, initial_vp,
                "PCG Variants — Victoria Park (7120 poses)",
                out_pcg
            )
        if gj_dumps:
            plot_trajectories(
                gj_dumps, gj_conv, initial_100,
                "GJ Variants — Victoria Park (first 100 poses)",
                out_gj
            )
    except ImportError:
        print("  matplotlib not available — skipping plots")


if __name__ == "__main__":
    main()
