# G-SLAM on GPU and FPGA

This repository contains implementations of **Graph-based Simultaneous Localization and Mapping (G-SLAM)** accelerated on GPU (CUDA) and FPGA (SYCL/oneAPI). It accompanies two peer-reviewed conference papers.

## Publications

[1] J. Zheng, Y. He, and M. Kondo, "Exploiting Data Parallelism in Graph-Based Simultaneous Localization and Mapping: A Case Study with GPU Accelerations," in *Proc. International Conference on High Performance Computing in Asia-Pacific Region (HPC Asia)*, Singapore, Feb. 2023, pp. 126–139.

[2] J. Wu, Y. He, and M. Kondo, "Accelerating Graph-Based SLAM through Data Parallelism and Mixed Precision on FPGAs," in *Proc. IEEE International Symposium on Embedded Multicore/Many-core Systems-on-Chip (MCSoC)*, 2023.

Papers are available from the conference proceedings.

## Background

G-SLAM represents the robot's environment as a pose graph. Solving the back-end requires solving a sparse **3N×3N linear system Ω·Δx = ξ** at each Gauss-Newton iteration, where N is the number of trajectory poses. This solve dominates runtime (~95% for N=800).

Two algorithms are implemented:

- **Gauss-Jordan (GJ)**: dense direct inversion, O(N³). Simple to parallelize — each of the N pivot steps is an embarrassingly parallel row operation. Used in the published papers.
- **Sparse PCG**: Preconditioned Conjugate Gradient with block-Jacobi preconditioning. Exploits the ~99% sparsity of Ω via CSR format. Converges in O(N·nnz) total work. FP32-friendly because block-Jacobi reduces κ from ~10⁵ to ~10³.

## Repository Structure

```
src/
  cpu_gj_fp32/       GJ · FP32 · single-thread C
  cpu_gj_fp64/       GJ · FP64 · single-thread C          ← CPU baseline
  cpu_gj_mixed/      GJ · FP32 inversion, FP64 elsewhere · C
  cpu_pcg_fp32/      Sparse PCG · FP32 · single-thread C
  cpu_pcg_fp64/      Sparse PCG · FP64 · single-thread C

  gpu_gj_fp32/       GJ · FP32 · CUDA                     ← paper results
  gpu_gj_fp64/       GJ · FP64 · CUDA                     ← paper results
  gpu_gj_mixed/      GJ · FP32 inversion, FP64 elsewhere · CUDA
  gpu_pcg_fp32/      Sparse PCG · FP32 · cuSPARSE + cuBLAS
  gpu_pcg_fp64/      Sparse PCG · FP64 · cuSPARSE + cuBLAS

  fpga_gj_fp32/      GJ · FP32 · SYCL/oneAPI              ← paper results
  fpga_gj_fp64/      GJ · FP64 · SYCL/oneAPI
  fpga_gj_mixed/     GJ · FP32 inversion, FP64 elsewhere · SYCL
  fpga_pcg_fp32/     Sparse PCG · FP32 · SYCL streaming SpMV
  fpga_pcg_fp64/     Sparse PCG · FP64 · SYCL streaming SpMV

  python_reference/  Reference · np.linalg.inv (LAPACK FP64) · Python

data/                Test datasets (cityTrees, Victoria Park)
scripts/             Trajectory plots, pose dump tools, run scripts
```

## Prerequisites

| Target | Requirements |
|--------|-------------|
| CPU (C) | GCC, `-lm` |
| GPU (CUDA) | CUDA Toolkit ≥ 11, `nvcc` at `/usr/local/cuda/bin/nvcc`, cuSPARSE + cuBLAS |
| FPGA/CPU/GPU (SYCL) | Intel oneAPI Base Toolkit (`icpx -fsycl`) |
| Python | Python 3, numpy, matplotlib |

## Building and Running

All implementations use the same command-line interface:

```bash
./GSLAM <use_diff> <lambda> <num_rounds> <drawing_mode> <input_file>
```

| Argument | Values | Description |
|----------|--------|-------------|
| `use_diff` | 0 or 1 | Convergence metric: 0 = cost difference, 1 = ‖Δx‖/√dim |
| `lambda` | e.g. 1.0 | Edge weight scale factor (see [Choosing Lambda](#choosing-lambda)) |
| `num_rounds` | e.g. 1000 | Maximum Gauss-Newton iterations |
| `drawing_mode` | 0–4 | Visualization: 0 = none |
| `input_file` | path | Dataset file (`.log.txt` format) |

### CPU implementations

```bash
cd src/cpu_gj_fp64
make
./GSLAM 1 0.1 1000 0 ../../data/cityTrees800.txt
```

Makefile targets: `make` (default, gcc -O3), `make x86` (same).

### GPU implementations

```bash
cd src/gpu_gj_fp64
make cuda
./GSLAM.cuda 1 0.1 1000 0 ../../data/cityTrees800.txt
```

PCG variants also link `-lcusparse -lcublas`:
```bash
cd src/gpu_pcg_fp64
make cuda
./GSLAM.cuda 1 1.0 1000 0 ../../data/victoria-park.log.txt
```

### FPGA implementations

SYCL code compiles for CPU, GPU, emulator, or FPGA hardware depending on the target:

```bash
cd src/fpga_gj_fp32

# Test on CPU (no FPGA required)
make cpu_test
./GSLAM.cpu_test 1 0.1 1000 0 ../../data/cityTrees800.txt

# FPGA emulator
make emu
./GSLAM.emu 1 0.1 1000 0 ../../data/cityTrees800.txt

# FPGA hardware synthesis (hours, requires board)
make s10    # Stratix 10 SX
make a10    # Arria 10 GX
```

Set `LD_LIBRARY_PATH` for the SYCL runtime:
```bash
export LD_LIBRARY_PATH=/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/umf/1.0/lib
```

### Python reference

```bash
cd src/python_reference
python3 GSLAM.py ../../data/cityTrees800.txt
```

## Choosing Lambda

The `lambda` parameter scales all edge weights equally. The optimal value depends on
the dataset's observation density and noise calibration, not the algorithm or
precision variant. **The same lambda works across all CPU/GPU/FPGA variants on a given
dataset.**

| Dataset | Best λ | Notes |
|---------|--------|-------|
| `cityTrees800.txt` | 0.1 | Dense city-grid, 10 rounds sufficient |
| `victoria-park.log.txt` (PCG) | **1.0** | 7120 poses, rich loop closures |
| `victoria-park-100.log.txt` (GJ) | **0.06** | 100-pose subset |

**How to find lambda for a new dataset** — use a coarse-then-fine grid search with the
fast CPU FP64 reference:

```bash
# Step 1: coarse sweep — decades
for lam in 0.01 0.1 1.0 10.0; do
  drift=$(./GSLAM 1 $lam 200 0 dataset.log.txt 2>&1 | \
    python3 -c "import sys,math; \
    poses={int(p[0]):(float(p[1]),float(p[2])) for line in sys.stdin \
    for p in [line.split()] if len(p)==4}; \
    s=sorted(poses); x0,y0=poses[s[0]]; xn,yn=poses[s[-1]]; \
    print(f'{math.sqrt((xn-x0)**2+(yn-y0)**2):.2f}')" 2>/dev/null || echo "N/A")
  echo "lambda=$lam  drift=${drift}m"
done

# Step 2: fine sweep — narrow around the best decade
for lam in 0.5 0.7 0.8 0.9 1.0 1.1 1.2 1.5; do
  # same one-liner
done
```

Alternatively, use `POSE_DUMP_FILE` to save poses and compute drift externally:

```bash
POSE_DUMP_FILE=/tmp/poses.txt ./GSLAM 1 1.0 1000 0 dataset.log.txt > /dev/null
python3 scripts/pose_dump_draw.py  # or compute drift from the dump
```

**Why the optimum can be sharp:** λ rescales the information matrix uniformly. When
λ equals the reciprocal of the average observation noise variance, the information
matrix is at its natural scale and loop-closure geometry closes tightly. The sharp
peak observed at λ=1.0 for Victoria Park (0.16 m vs >4 m at λ=0.9 or λ=1.1) reflects
accurate noise calibration in that dataset — the covariance parameters are already
correct at unity scale.

## Algorithm and Precision Variants

### Gauss-Jordan (GJ) — Dense Direct

All N pivot steps are sequential, but each step parallelizes ratio computation (N threads)
and row elimination (N×N threads). Parallelism scales as O(N²) per step.

**Precision notes:**
- **FP64**: Accurate for all dataset sizes tested.
- **FP32**: Diverges on cityTrees800 (κ ≈ 10⁵). Useful as a baseline comparison.
- **Mixed**: FP32 for inversion only, FP64 for everything else. Gives ~3×10⁻⁴ error — acceptable for robotics.

### Sparse PCG — Iterative

Solves Ω·x = ξ iteratively without forming the inverse. Key components:

- **CSR matrix**: Built directly from graph edges; never allocates the full N×N dense Ω.
- **SpMV**: q = Ω·p using CSR row-wise dot products (CPU loop / cuSPARSE / SYCL kernel).
- **Block-Jacobi preconditioner**: Invert each 3×3 diagonal block of Ω analytically. O(N) cost, reduces κ from ~10⁵ to ~10³.
- **Convergence**: ‖r‖ < 10⁻⁶·‖b‖, max 500 inner iterations per Gauss-Newton round.
- **GPU optimization**: All CSR assembly and PCG iterations run on device; only `hat_xs` (~480 KB for N=20k) transfers per round.

**Precision notes:**
- **FP64**: Full accuracy.
- **FP32**: Works correctly — block-Jacobi preconditioning tames the condition number.

## Datasets

| File | Poses | Observations | Source |
|------|-------|--------------|--------|
| `cityTrees800.txt` | 800 | ~3k pairs | Real-world |
| `cityTrees1600.txt` | 1600 | ~6k pairs | Real-world |
| `victoria-park.log.txt` | 7120 | 105k pairs | Real-world outdoor |
| `victoria-park-100.log.txt` | 100 | ~70 pairs | Subset of above |
| `cityTrees20k.txt` | 20000 | 109k pairs | Synthetic (generator) |
| `cityTrees50k.txt` | 50000 | — | Synthetic |

Synthetic datasets generated with `data/generate_citytrees.py`:
```bash
python3 data/generate_citytrees.py <n_poses> <output.txt>
```

## Citation

If you use this code, please cite the relevant paper(s):

```bibtex
@inproceedings{zheng2023exploiting,
  title     = {Exploiting Data Parallelism in Graph-Based Simultaneous Localization and Mapping: A Case Study with {GPU} Accelerations},
  author    = {Zheng, Junyuan and He, Yuan and Kondo, Masaaki},
  booktitle = {Proceedings of the International Conference on High Performance Computing in Asia-Pacific Region (HPC Asia)},
  pages     = {126--139},
  year      = {2023},
  address   = {Singapore},
  publisher = {ACM}
}

@inproceedings{wu2023accelerating,
  title     = {Accelerating Graph-Based {SLAM} through Data Parallelism and Mixed Precision on {FPGAs}},
  author    = {Wu, Junfeng and He, Yuan and Kondo, Masaaki},
  booktitle = {Proceedings of the IEEE International Symposium on Embedded Multicore/Many-core Systems-on-Chip (MCSoC)},
  year      = {2023}
}
```

## Acknowledgements

The Python reference implementation (`src/python_reference/`) is derived from
[ryuichiueda/LNPR_BOOK_CODES](https://github.com/ryuichiueda/LNPR_BOOK_CODES)
by Ryuichi Ueda, the companion code for the textbook
*詳解確率ロボティクス (Probabilistic Robotics in Detail)*,
used under the [MIT License](https://github.com/ryuichiueda/LNPR_BOOK_CODES/blob/master/LICENSE.md).

## License

[MIT](LICENSE)
