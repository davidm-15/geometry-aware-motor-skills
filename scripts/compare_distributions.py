"""
Distributional Shift Analysis: train.csv vs test.csv
Compares spatial coordinates (X,Y,Z), orientation angles (A,B,C),
and kinematic derivatives (velocity, jerk) between in-distribution
training set and out-of-distribution test set.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial.transform import Rotation as R
import gc
import warnings
warnings.filterwarnings('ignore')

# ─── Load Data ───────────────────────────────────────────────────────────────
def process_df(path):
    # Load required columns
    cols = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'stroke_id', 'demonstration_id']
    available = pd.read_csv(path, nrows=0).columns.tolist()
    cols = [c for c in cols if c in available]
    df = pd.read_csv(path, usecols=cols)

    # Rename columns to match prompt
    df = df.rename(columns={'x': 'X', 'y': 'Y', 'z': 'Z'})
    if 'demonstration_id' in df.columns:
        df['ID'] = df['demonstration_id'].astype(str) + '_' + df['stroke_id'].astype(str)
    else:
        df['ID'] = df['stroke_id'].astype(str)
    
    # Convert Quaternions to Euler Angles
    quats = df[['qx', 'qy', 'qz', 'qw']].values
    euler = R.from_quat(quats).as_euler('xyz', degrees=True)
    df['A'] = euler[:, 0]
    df['B'] = euler[:, 1]
    df['C'] = euler[:, 2]
    
    df = df.drop(columns=['qx', 'qy', 'qz', 'qw'])
    return df

print("Loading and processing data...")
train = process_df('datasets/training/train.csv')
test  = process_df('datasets/training/test.csv')
print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. BOUNDARY LIMITS: Spatial Coordinates (X, Y, Z)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("1. BOUNDARY LIMITS — Spatial Coordinates (X, Y, Z)")
print("=" * 80)

spatial_cols = ['X', 'Y', 'Z']

for col in spatial_cols:
    tr_min, tr_max, tr_var = train[col].min(), train[col].max(), train[col].var()
    te_min, te_max, te_var = test[col].min(),  test[col].max(),  test[col].var()
    print(f"\n  {col}:")
    print(f"    Train — min: {tr_min:12.4f}  max: {tr_max:12.4f}  var: {tr_var:12.4f}")
    print(f"    Test  — min: {te_min:12.4f}  max: {te_max:12.4f}  var: {te_var:12.4f}")
    
    below = (test[col] < tr_min).sum()
    above = (test[col] > tr_max).sum()
    pct_below = 100 * below / len(test)
    pct_above = 100 * above / len(test)
    print(f"    Test points BELOW train min: {below} ({pct_below:.2f}%)")
    print(f"    Test points ABOVE train max: {above} ({pct_above:.2f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ORIENTATION EXTREMES: Euler Angles (A, B, C)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("2. ORIENTATION EXTREMES — Euler Angles (A, B, C)")
print("=" * 80)

angle_cols = ['A', 'B', 'C']

for col in angle_cols:
    tr_min, tr_max, tr_var = train[col].min(), train[col].max(), train[col].var()
    te_min, te_max, te_var = test[col].min(),  test[col].max(),  test[col].var()
    print(f"\n  {col}:")
    print(f"    Train — min: {tr_min:12.4f}  max: {tr_max:12.4f}  var: {tr_var:12.4f}")
    print(f"    Test  — min: {te_min:12.4f}  max: {te_max:12.4f}  var: {te_var:12.4f}")
    below = (test[col] < tr_min).sum()
    above = (test[col] > tr_max).sum()
    pct_oob = 100 * (below + above) / len(test)
    print(f"    Test points outside train range: {below + above} ({pct_oob:.2f}%)")

print("\n  --- KL Divergence (test || train) ---")
n_bins = 100
for col in angle_cols:
    all_min = min(train[col].min(), test[col].min())
    all_max = max(train[col].max(), test[col].max())
    bins = np.linspace(all_min, all_max, n_bins + 1)
    
    tr_hist, _ = np.histogram(train[col], bins=bins, density=True)
    te_hist, _ = np.histogram(test[col],  bins=bins, density=True)
    
    eps = 1e-10
    tr_hist = (tr_hist + eps) / (tr_hist + eps).sum()
    te_hist = (te_hist + eps) / (te_hist + eps).sum()
    
    kl_div = entropy(te_hist, tr_hist)
    print(f"    KL({col}_test || {col}_train) = {kl_div:.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. KINEMATIC DERIVATIVES: Velocity, Jerk (per ID)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("3. KINEMATIC DERIVATIVES — Velocity, Jerk (per ID)")
print("=" * 80)

def compute_kinematics(df):
    vel_means = {c: [] for c in spatial_cols}
    vel_maxs  = {c: [] for c in spatial_cols}
    jerk_means = {c: [] for c in spatial_cols}
    jerk_maxs  = {c: [] for c in spatial_cols}
    
    # Sort first just in case
    for _, grp in df.groupby('ID'):
        for col in spatial_cols:
            vals = grp[col].values
            if len(vals) < 4:
                continue
            vel = np.diff(vals, n=1)
            jerk = np.diff(vals, n=3)
            vel_means[col].append(np.mean(np.abs(vel)))
            vel_maxs[col].append(np.max(np.abs(vel)))
            jerk_means[col].append(np.mean(np.abs(jerk)))
            jerk_maxs[col].append(np.max(np.abs(jerk)))
            
    return vel_means, vel_maxs, jerk_means, jerk_maxs

print("Computing kinematics for train set...")
tr_vm, tr_vmax, tr_jm, tr_jmax = compute_kinematics(train)
print("Computing kinematics for test set...")
te_vm, te_vmax, te_jm, te_jmax = compute_kinematics(test)

print("\n  --- Velocity (|Δx| per step) ---")
for col in spatial_cols:
    tr_mean_v = np.mean(tr_vm[col])
    te_mean_v = np.mean(te_vm[col])
    tr_max_v  = np.max(tr_vmax[col])
    te_max_v  = np.max(te_vmax[col])
    print(f"\n  {col}:")
    print(f"    Train — mean(|v|): {tr_mean_v:.6f}  max(|v|): {tr_max_v:.6f}")
    print(f"    Test  — mean(|v|): {te_mean_v:.6f}  max(|v|): {te_max_v:.6f}")
    print(f"    Ratio mean(|v|): {te_mean_v / tr_mean_v:.4f}")

print("\n  --- Jerk (|Δ³x| per step) ---")
for col in spatial_cols:
    tr_mean_j = np.mean(tr_jm[col])
    te_mean_j = np.mean(te_jm[col])
    tr_max_j  = np.max(tr_jmax[col])
    te_max_j  = np.max(te_jmax[col])
    print(f"\n  {col}:")
    print(f"    Train — mean(|j|): {tr_mean_j:.6f}  max(|j|): {tr_max_j:.6f}")
    print(f"    Test  — mean(|j|): {te_mean_j:.6f}  max(|j|): {te_max_j:.6f}")
    print(f"    Ratio mean(|j|): {te_mean_j / tr_mean_j:.4f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
