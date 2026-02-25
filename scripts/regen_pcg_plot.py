import math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import numpy as np

REPO_ROOT = '/home/he/Workspace/g-slam_on_fpga_and_gpu'
DUMP_DIR  = os.path.join(REPO_ROOT, 'scripts/_pose_dumps_vp')

PCG_VARIANTS = ['cpu_pcg_fp64','cpu_pcg_fp32','fpga_pcg_fp64','fpga_pcg_fp32','gpu_pcg_fp64','gpu_pcg_fp32']
PCG_CONV = {'cpu_pcg_fp64':116,'cpu_pcg_fp32':68,'fpga_pcg_fp64':84,'fpga_pcg_fp32':64,'gpu_pcg_fp64':65,'gpu_pcg_fp32':69}
COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']

def load_poses(path):
    poses = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 4:
                poses[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
    return poses

def drift(poses):
    steps = sorted(poses.keys())
    x0,y0,_ = poses[steps[0]]; xn,yn,_ = poses[steps[-1]]
    return math.sqrt((xn-x0)**2+(yn-y0)**2)

def load_dr(datafile):
    poses = {}
    with open(datafile) as f:
        for line in f:
            if line.startswith("x "):
                p = line.split()
                poses[int(p[1])] = (float(p[2]), float(p[3]))
    return poses

dr = load_dr(os.path.join(REPO_ROOT, 'data/victoria-park.log.txt'))
all_poses = {v: load_poses(os.path.join(DUMP_DIR, f'{v}.txt'))
             for v in PCG_VARIANTS
             if os.path.exists(os.path.join(DUMP_DIR, f'{v}.txt'))}

dr_steps = sorted(dr.keys())
dr_xs = [dr[s][0] for s in dr_steps]
dr_ys = [dr[s][1] for s in dr_steps]
dr_drift = math.sqrt((dr_xs[-1]-dr_xs[0])**2+(dr_ys[-1]-dr_ys[0])**2)

drifts  = [drift(all_poses[v]) for v in PCG_VARIANTS]
conv_rs = [PCG_CONV[v] for v in PCG_VARIANTS]

# ---- figure: 3 panels ----
fig = plt.figure(figsize=(20, 8))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.2, 0.85], wspace=0.4)
ax_dr   = fig.add_subplot(gs[0])
ax_slam = fig.add_subplot(gs[1])
ax_bar  = fig.add_subplot(gs[2])
fig.suptitle('PCG Variants — Victoria Park (7120 poses)', fontsize=15, fontweight='bold')

# ===================== LEFT: dead-reckoning =====================
ax_dr.plot(dr_xs, dr_ys, color='steelblue', linewidth=0.9, alpha=0.85)
ax_dr.plot(dr_xs[0],  dr_ys[0],  'go', markersize=9, zorder=5, label='Start')
ax_dr.plot(dr_xs[-1], dr_ys[-1], 'rs', markersize=9, zorder=5, label=f'End  (drift {dr_drift:.0f} m)')

ref = all_poses['cpu_pcg_fp64']
ref_steps = sorted(ref.keys())
ref_xs = [ref[s][0] for s in ref_steps]
ref_ys = [ref[s][1] for s in ref_steps]
pad = 15
bx0,bx1 = min(ref_xs)-pad, max(ref_xs)+pad
by0,by1 = min(ref_ys)-pad, max(ref_ys)+pad

ax_dr.add_patch(Rectangle((bx0,by0), bx1-bx0, by1-by0,
                           facecolor='gold', alpha=0.2, zorder=2, linewidth=0))
ax_dr.add_patch(Rectangle((bx0,by0), bx1-bx0, by1-by0,
                           linewidth=2, edgecolor='firebrick', facecolor='none',
                           linestyle='--', zorder=3))
ax_dr.text((bx0+bx1)/2, (by0+by1)/2, 'SLAM\nresult\n→',
           ha='center', va='center', fontsize=10, color='firebrick', fontweight='bold')

ax_dr.set_title('Dead-reckoning (raw odometry)', fontsize=11)
ax_dr.set_xlabel('X (m)'); ax_dr.set_ylabel('Y (m)')
ax_dr.legend(loc='lower left', fontsize=8.5)
ax_dr.grid(True, alpha=0.2)

# ===================== MIDDLE: reference SLAM trajectory only =====================
# Show dead-reckoning faintly
ax_slam.plot(dr_xs, dr_ys, color='#cccccc', linewidth=0.6, alpha=0.5,
             linestyle='--', label='Dead-reckoning', zorder=1)

# Draw all variants thin first, then reference thick on top
for i, label in enumerate(PCG_VARIANTS):
    if label not in all_poses or label == 'cpu_pcg_fp64':
        continue
    poses = all_poses[label]
    steps = sorted(poses.keys())
    xs = [poses[s][0] for s in steps]
    ys = [poses[s][1] for s in steps]
    ax_slam.plot(xs, ys, color=COLORS[i], linewidth=0.7, alpha=0.45, zorder=2)

# Reference on top
ax_slam.plot(ref_xs, ref_ys, color=COLORS[0], linewidth=1.8, alpha=0.95,
             label='cpu_pcg_fp64 (reference ★)', zorder=10)

ax_slam.plot(ref_xs[0], ref_ys[0], 'k^', markersize=11, zorder=20, label='Start')
ax_slam.plot(ref_xs[-1], ref_ys[-1], 'kD', markersize=8, zorder=20, label='End (reference)')

ax_slam.set_xlim(bx0, bx1)
ax_slam.set_ylim(by0, by1)
ax_slam.set_title('SLAM result — reference trajectory (cpu_pcg_fp64)\nother variants shown faintly', fontsize=10.5)
ax_slam.set_xlabel('X (m)'); ax_slam.set_ylabel('Y (m)')
ax_slam.legend(loc='lower left', fontsize=8.5, framealpha=0.9)
ax_slam.grid(True, alpha=0.2)

# ===================== RIGHT: bar chart =====================
y_pos = np.arange(len(PCG_VARIANTS))
short_labels = [v.replace('_pcg','') for v in PCG_VARIANTS]

bars = ax_bar.barh(y_pos, drifts, color=COLORS, edgecolor='white', height=0.55)
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(short_labels, fontsize=9)
ax_bar.set_xlabel('End-to-end drift (m)', fontsize=10)
ax_bar.set_title('Drift & convergence round\nper variant', fontsize=10.5)
ax_bar.grid(axis='x', alpha=0.3)
ax_bar.invert_yaxis()

# Add drift value + conv round as text
for i, (bar, d, cr) in enumerate(zip(bars, drifts, conv_rs)):
    ax_bar.text(d + 0.3, bar.get_y() + bar.get_height()/2,
                f'{d:.1f} m  (r{cr})', va='center', fontsize=8.5)

# Dead-reckoning reference line
ax_bar.axvline(dr_drift, color='steelblue', linestyle='--', linewidth=1.2, alpha=0.7)
ax_bar.text(dr_drift + 0.5, len(PCG_VARIANTS)-0.3,
            f'Dead-reckoning\n{dr_drift:.0f} m', color='steelblue',
            fontsize=7.5, va='top')
ax_bar.set_xlim(0, max(drifts) * 1.5)
ax_bar.axvline(dr_drift, color='steelblue', linestyle='--', linewidth=1.2, alpha=0.7,
               label=f'Dead-reckoning ({dr_drift:.0f} m)')
ax_bar.legend(loc='lower right', fontsize=7.5)

plt.subplots_adjust(left=0.05, right=0.97, top=0.92, bottom=0.1, wspace=0.38)
out = os.path.join(REPO_ROOT, 'scripts/trajectory_pcg_victoria_park.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
