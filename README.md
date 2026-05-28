# Learning Transferable Motor Skills for Geometry-Aware Robotic Surface Tasks

Code for the ICRA-WS paper. A research pipeline for extracting and transferring human motor skills from demonstrations to geometry-aware robotic surface tasks. The system generates synthetic 3D object datasets, simulates robot end-effector trajectories with parameterised skill rules (velocity scaling, spatial position/orientation, geometry proximity), trains a multimodal model (GRU + PointNet CAD encoder) to predict those rules, and evaluates transfer to out-of-distribution geometries.

---

## Pipeline Overview

```
generate_X_samples  →  generate_dataset  →  training/train
       ↓                      ↓                    ↓
  OBJ meshes +          train/val/test          best_fusion_model.pt
  trajectory.txt           CSVs
                             ↓
                    visualize_dataset   (inspect generated CSVs)
                    plot_velocity_profile  (inspect a single trajectory)
```

---

## Step-by-step Usage

All scripts are run as Python modules from the **repository root**:

```bash
cd /path/to/SkillTrace
```

---

### Step 1 — Generate 3D object samples

Each script creates a set of randomly-sized geometric variants of a template mesh (`.obj`) together with a matching `trajectory.txt` path file. Pick the script for the shape you need:

| Shape | Script | Default output |
|-------|--------|----------------|
| L-shape | `data_generation/generate_L_samples.py` | `datasets/L_shape/` |
| I-shape (straight stick) | `data_generation/generate_I_samples.py` | `datasets/I_shape/` |
| Window / cross | `data_generation/generate_window_cross_samples.py` | `datasets/windows-v2/` |

**Usage (L-shape example):**

```bash
# Default: 1000 L-shape samples
python -m data_generation.generate_L_samples

# Custom
python -m data_generation.generate_L_samples \
    --source-dir datasets/L_shape/00_source \
    --output-dir datasets/L_shape \
    --sample-name L_shape \
    --num-samples 500 \
    --min-length 100 \
    --max-length 500
```

Each sample is written to `<output-dir>/<i>_<sample-name>/`:
- `<i>_<sample-name>.obj` — the mesh
- `trajectory.txt` — the waypoint path

The source directory (`00_source/`) must contain a template `.obj` and a `trajectory.txt`.

---

### Step 2 — Generate the dataset (train / val / test splits)

`data_generation/generate_dataset.py` reads the sample directories created above, runs the physics simulation with configurable rules, and produces three CSV files.

```bash
# L-shape (default)
python -m data_generation.generate_dataset

# I-shape
python -m data_generation.generate_dataset \
    --samples-dir datasets/I_shape \
    --sample-name I_shape \
    --output-dir datasets/I_shape \
    --total-windows 1000

# Window / cross
python -m data_generation.generate_dataset \
    --samples-dir datasets/windows-v2 \
    --sample-name wr1fr_1 \
    --output-dir datasets/windows-v2 \
    --total-windows 1000
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--samples-dir` | `datasets/L_shape` | Root directory with sample sub-folders |
| `--sample-name` | `L_shape` | Base name used to discover `<i>_<name>/` folders |
| `--output-dir` | `datasets/L_shape` | Where CSVs and split JSONs are written |
| `--total-windows` | 100 | Number of sample windows to use |
| `--train-ratio` | 0.80 | Fraction for training |
| `--val-ratio` | 0.10 | Fraction for validation (remainder → test) |
| `--disable-vel-rule` | off | Disable velocity scaling rule |
| `--enable-pos-rule` | off | Enable spatial position rule |
| `--enable-ori-rule` | off | Enable spatial orientation rule |
| `--enable-prox-rule` | off | Enable geometry proximity rule |

**Outputs** written to `<output-dir>/`:
- `train.csv`, `val.csv`, `test.csv`
- `train_split.json`, `val_split.json`, `test_split.json`

CSV columns: `time(s), x, y, z, qx, qy, qz, qw, velocity, stroke_id, segment_type, window_id, rule_vel_scale, rule_pos_y, rule_ori_x, geometric_proximity, …_geom`

---

### Step 3 — Inspect the generated dataset

**`ros2/visualize_trajectory.py`** — replay any generated dataset CSV in RViz2. Randomly cycles through strokes and renders the matching 3D mesh alongside the coloured trajectory (green = straight, red = corner). Works with L-shape, I-shape, window/cross, or any other shape.

```bash
# Default dataset
python3 -m ros2.visualize_trajectory

# Custom CSV
python3 -m ros2.visualize_trajectory datasets/windows-v2/test.csv
python3 -m ros2.visualize_trajectory datasets/L_shape/test.csv
```

**`scripts/visualize_dataset.py`** — matplotlib view of velocity profiles from a full train/val/test CSV, grouped by `rule_vel_scale` range and segment type (corner vs straight).

```bash
python -m scripts.visualize_dataset
# edit the default csv_path inside the script to point at your dataset
```

**`scripts/plot_velocity_profile.py`** — plot the velocity over time for a single trajectory CSV.

```bash
python -m scripts.plot_velocity_profile \
    outputs/test_load_data/trajectory.csv
```

---

### Step 4 — Train the model

`training/train.py` trains a **FusionModel** (GRU sequence encoder + PointNet-based CAD encoder) that jointly predicts continuous rule values (velocity scale, position offset, orientation) and discrete geometry classes (straight / corner / crossing).

```bash
# Default: L-shape dataset
python -m training.train

# Custom dataset
python -m training.train \
    --dataset-path datasets/windows-v2 \
    --output-dir outputs/windows_v2

# Evaluate only (load existing checkpoint)
python -m training.train \
    --dataset-path datasets/windows-v2 \
    --output-dir outputs/windows_v2 \
    --eval-only
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-path` | `datasets/L_shape` | Root containing `train.csv`, `val.csv`, `test.csv`, and OBJ files |
| `--output-dir` | `outputs/L_shape_fusion_test` | Where the checkpoint and plots are saved |
| `--eval-only` | off | Skip training, evaluate saved checkpoint on test set |

The best checkpoint is saved as `<output-dir>/best_fusion_model.pt`. Test metrics (MAE, RMSE, geometry F1, confusion matrix) and scatter plots are saved to `<output-dir>/`.

---

### Step 5 — Evaluate on real-world data

`scripts/eval_real_data.py` runs a trained checkpoint against real robot recordings. It prints per-trajectory predictions and aggregate MAE/RMSE/F1 metrics, and saves scatter plots and confusion matrices.

```bash
# Default paths
python -m scripts.eval_real_data

# Custom checkpoint and CSV
python -m scripts.eval_real_data \
    --model-path outputs/windows_v2_fusion_test/best_fusion_model.pt \
    --real-csv datasets/real_experiment/combined.csv \
    --cad-root measurements \
    --output-dir outputs/real_experiment_eval
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `outputs/windows_v2_fusion_test/best_fusion_model.pt` | Trained checkpoint to evaluate |
| `--real-csv` | `datasets/real_experiment/combined.csv` | Real recording CSV |
| `--cad-root` | `measurements` | Directory containing `<window_id>/<window_id>.obj` CAD files |
| `--output-dir` | `outputs/real_experiment_eval` | Where plots are saved |
| `--position-scale` | `1.0` | Multiply positions by this factor before inference. Use `1000.0` if real data is in metres and the model was trained on mm |
| `--upsample-dt` | off | Resample sparse trajectories to a fixed dt in seconds (e.g. `0.02` for 50 Hz) |
| `--headless` | off | Save plots to disk without displaying them |

**Expected CSV format** (`--real-csv`): same columns as the generated datasets — `time(s), x, y, z, qx, qy, qz, qw, window_id, stroke_id, rule_vel_scale, rule_ori_x, geometric_proximity, rule_vel_scale_geom, rule_ori_x_geom, geometric_proximity_geom`.

**CAD files** must be placed at `<cad-root>/<window_id>/<window_id>.obj` (same layout as the `datasets/` directories).

---

## Directory Structure

```
SkillTrace/
├── data_generation/          # Sample generation + simulation pipeline
│   ├── generate_L_samples.py
│   ├── generate_I_samples.py
│   ├── generate_window_cross_samples.py
│   ├── generate_dataset.py
│   ├── rules.py              # VelocityScaling, SpatialPosition, … rules
│   ├── simulation.py         # Pure-pursuit end-effector simulation
│   └── load_data.py
├── datasets/                 # Generated sample directories + CSVs
│   ├── L_shape/
│   ├── I_shape/
│   └── windows-v2/
├── model/                    # FusionModel, CAD encoder, RNN definitions
├── training/
│   ├── train.py              # Training + evaluation entry point
│   ├── dataset.py            # FusionDataset (reads CSV + OBJ)
│   └── evaluate.py
├── scripts/
│   ├── visualize_dataset.py  # Inspect dataset CSVs (velocity by rule bin)
│   └── plot_velocity_profile.py  # Plot velocity of a single trajectory
├── utils/
│   └── visualization.py      # 3D trajectory plotting helpers
├── ros2/
│   ├── visualize_trajectory.py   # RViz2 replay of any generated CSV dataset
│   └── …                         # other ROS2 publishers / subscribers
├── configs/
│   └── sim_params.json       # Pure-pursuit simulation parameters
└── tests/                    # pytest unit tests
```

---

## Configuration

Simulation parameters (step size, lookahead distance, damping, …) live in `configs/sim_params.json` and are loaded automatically by `generate_dataset.py`.

---

## Requirements

- Python 3.10
- PyTorch (CUDA optional)
- NumPy, Pandas, Matplotlib, SciPy
- Open3D or similar (for skeleton extraction)
- pytest (for tests)
