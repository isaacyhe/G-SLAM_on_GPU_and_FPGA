# G-SLAM Python Reference Implementation

Python reference implementation using `numpy.linalg.inv` (LAPACK LU, FP64).

## Attribution

Derived from [ryuichiueda/LNPR_BOOK_CODES](https://github.com/ryuichiueda/LNPR_BOOK_CODES)
by Ryuichi Ueda (`section_graph_slam/`), companion code for
*詳解確率ロボティクス (Probabilistic Robotics in Detail)*.
Used under the [MIT License](https://github.com/ryuichiueda/LNPR_BOOK_CODES/blob/master/LICENSE.md).

## Algorithm

- Dense matrix inversion via `np.linalg.inv` (LAPACK LU decomposition, FP64)
- Graph construction: `MotionEdge`, `ObsEdge`, `MapEdge` classes
- Landmark positions estimated by averaging observation-consistent positions

## Build

No compilation needed.

```bash
pip install numpy matplotlib
```

## Run

```bash
python3 GSLAM.py ../../data/cityTrees800.txt
```

## Related

- `../cpu_gj_fp64/` — equivalent C implementation
- `../cpu_pcg_fp64/` — sparse iterative alternative
