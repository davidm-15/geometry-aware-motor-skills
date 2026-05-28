import copy
import os
import numpy as np
import pandas as pd
import torch
import trimesh
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

ACTIVE_RULES = ["vel_scale", "ori_x", "proximity"]

# Per-rule geometry class names: 0=class0, 1=class1, 2=none
RULE_CLASS_NAMES = {
    "vel_scale":  ["straight", "corner", "none"],
    "ori_x":      ["straight", "corner", "none"],
    "proximity":  ["edge",     "crossing", "none"],
}

# Kept for backward-compat with code that imports CLASS_NAMES
CLASS_NAMES = ["straight", "corner", "none"]


def _geom_to_idx(arr, class0, class1):
    """Map string geometry labels to 0/1/2 (none)."""
    return np.where(arr == class0, 0, np.where(arr == class1, 1, 2)).astype(np.int64)


class TrajDataset(Dataset):
    def __init__(self, csv_file, feature_stats=None, target_stats=None):
        data = pd.read_csv(csv_file)

        raw_coords  = data[["x", "y", "z"]].values.astype(np.float32)
        raw_time    = data["time(s)"].values.astype(np.float32).reshape(-1, 1)
        raw_quats   = data[["qx", "qy", "qz", "qw"]].values.astype(np.float32)
        window_id   = data["window_id"].values.astype(str)
        demo_id     = data["demonstration_id"].values.astype(str) if "demonstration_id" in data.columns else window_id

        # Continuous rule targets
        rule_vel   = data["rule_vel_scale"].values.astype(np.float32).reshape(-1, 1)
        rule_ori   = data["rule_ori_x"].values.astype(np.float32).reshape(-1, 1)
        has_prox   = "geometric_proximity" in data.columns
        rule_prox  = (data["geometric_proximity"].values.astype(np.float32).reshape(-1, 1)
                      if has_prox else np.ones((len(data), 1), dtype=np.float32))

        # Geometry class targets
        vel_geom  = _geom_to_idx(data["rule_vel_scale_geom"].values.astype(str),  "straight", "corner").reshape(-1, 1)
        ori_geom  = _geom_to_idx(data["rule_ori_x_geom"].values.astype(str),      "straight", "corner").reshape(-1, 1)
        prox_geom = (_geom_to_idx(data["geometric_proximity_geom"].values.astype(str), "edge", "crossing").reshape(-1, 1)
                     if has_prox else np.full((len(data), 1), 2, dtype=np.int64))

        # targets_raw_vals columns: vel, ori, prox, vel_geom, ori_geom, prox_geom
        targets_raw_vals = np.concatenate(
            [rule_vel, rule_ori, rule_prox, vel_geom, ori_geom, prox_geom], axis=1
        )

        # Feature engineering
        p_delta = np.diff(raw_coords, axis=0)
        t_delta = np.diff(raw_time, axis=0)
        R_mats  = R.from_quat(raw_quats).as_matrix()
        feat_R  = R_mats[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)[:-1]
        safe_t  = np.where(t_delta == 0, 1e-6, t_delta)
        vel_raw = p_delta / safe_t
        features_all = np.concatenate([t_delta, vel_raw, feat_R], axis=1)

        # Group consecutive rows with the same window_id into sequences
        raw_samples = []
        seq, cur_wid = [], None
        for i in range(len(features_all) - 1):
            if window_id[i] == window_id[i + 1] and demo_id[i] == demo_id[i + 1]:
                seq.append(features_all[i])
                cur_wid = window_id[i]
            elif seq:
                raw_samples.append((copy.deepcopy(seq), targets_raw_vals[i].copy(), cur_wid))
                seq, cur_wid = [], None

        # Normalisation stats
        if feature_stats is None:
            feat_mean  = np.mean(features_all[:, 1:4], axis=0)
            feat_std   = np.std(features_all[:, 1:4],  axis=0) + 1e-6
            time_mean  = float(np.mean(features_all[:, 0]))
            time_std   = float(np.std(features_all[:, 0]) + 1e-6)
            self.feature_stats = (feat_mean, feat_std, time_mean, time_std)
        else:
            feat_mean, feat_std, time_mean, time_std = feature_stats
            self.feature_stats = feature_stats

        # Continuous target stats for all 3 rules
        cont_targets = np.array([s[1][0:3] for s in raw_samples], dtype=np.float32)
        if target_stats is None:
            target_mean = np.mean(cont_targets, axis=0)
            target_std  = np.std(cont_targets,  axis=0) + 1e-6
            self.target_stats = (target_mean, target_std)
        else:
            target_mean, target_std = target_stats
            # Handle old 2-rule checkpoints: pad to 3
            if len(target_mean) == 2:
                prox_mean = np.mean(cont_targets[:, 2]) if len(raw_samples) else 1.0
                prox_std  = np.std(cont_targets[:, 2])  + 1e-6 if len(raw_samples) else 1e-6
                target_mean = np.append(target_mean, prox_mean)
                target_std  = np.append(target_std,  prox_std)
            self.target_stats = (target_mean, target_std)
            target_mean, target_std = self.target_stats

        self.samples = []
        longest_len = max(len(s[0]) for s in raw_samples) if raw_samples else 0

        for seq, target, wid in raw_samples:
            seq = np.array(seq, dtype=np.float32)
            actual_len = len(seq)

            seq[:, 0]   = (seq[:, 0]   - time_mean) / time_std
            seq[:, 1:4] = (seq[:, 1:4] - feat_mean) / feat_std

            cont_raw  = target[0:3].astype(np.float32)
            cont_norm = (cont_raw - target_mean) / target_std

            vel_geom_t  = int(target[3])
            ori_geom_t  = int(target[4])
            prox_geom_t = int(target[5])

            if longest_len - actual_len > 0:
                seq = np.pad(seq, ((0, longest_len - actual_len), (0, 0)),
                             mode="constant", constant_values=0)

            self.samples.append(
                (seq, cont_norm, vel_geom_t, ori_geom_t, prox_geom_t, actual_len, wid)
            )

        print(f"Loaded {csv_file}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, cont_norm, vel_geom, ori_geom, prox_geom, actual_len, _ = self.samples[idx]
        return (
            torch.tensor(seq, dtype=torch.float32).transpose(0, 1),   # [C, T]
            torch.tensor(cont_norm, dtype=torch.float32),              # [3]
            torch.tensor(vel_geom,  dtype=torch.long),
            torch.tensor(ori_geom,  dtype=torch.long),
            torch.tensor(prox_geom, dtype=torch.long),
            torch.tensor(actual_len, dtype=torch.long),
        )


def load_obj_as_pointcloud(path, n_points=1024, use_cache=True):
    cache_path = path.replace(".obj", f"_{n_points}pts.npy")

    if use_cache and os.path.exists(cache_path):
        return np.load(cache_path).astype(np.float32)

    mesh = trimesh.load(path, force="mesh", process=False)
    points, _ = trimesh.sample.sample_surface(mesh, n_points)

    points = points - points.mean(axis=0)
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points = points / scale

    points = points.astype(np.float32)

    if use_cache:
        np.save(cache_path, points)

    return points


class FusionDataset(TrajDataset):
    """
    Extends TrajDataset to also return the CAD point cloud for each sample.
    Expects CAD OBJ files at: <cad_root>/<window_id>/<window_id>.obj
    """

    def __init__(self, csv_file, cad_root, feature_stats=None, target_stats=None, n_points=1024):
        super().__init__(csv_file, feature_stats=feature_stats, target_stats=target_stats)
        self.cad_root = cad_root
        self.n_points = n_points

    def __getitem__(self, idx):
        seq, cont_norm, vel_geom, ori_geom, prox_geom, actual_len, window_id = self.samples[idx]

        obj_path = os.path.join(self.cad_root, window_id, f"{window_id}.obj")
        points = load_obj_as_pointcloud(obj_path, n_points=self.n_points, use_cache=True)
        cad = torch.from_numpy(points).t().contiguous()  # [3, N]

        return (
            torch.tensor(seq, dtype=torch.float32).transpose(0, 1),  # [C, T]
            torch.tensor(cont_norm, dtype=torch.float32),             # [3]
            torch.tensor(vel_geom,  dtype=torch.long),
            torch.tensor(ori_geom,  dtype=torch.long),
            torch.tensor(prox_geom, dtype=torch.long),
            torch.tensor(actual_len, dtype=torch.long),
            cad,                                                       # [3, N]
        )
