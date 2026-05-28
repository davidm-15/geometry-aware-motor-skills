"""
Evaluate the FusionModel on real experimental data.

Usage:
    python -m scripts.eval_real_data
    python -m scripts.eval_real_data --model-path outputs/windows_v2_fusion_test/best_fusion_model.pt
    python -m scripts.eval_real_data --headless
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from torch.utils.data import Dataset, DataLoader
import trimesh

from model.fusion_model import FusionModel
from model.CAD_encoder import CADEncoder
from training.dataset import CLASS_NAMES, ACTIVE_RULES, RULE_CLASS_NAMES
from training.evaluate import evaluate, plot_confusion_matrix, plot_scatter_plot


# ---------------------------------------------------------------------------
# Legacy GRU-based trajectory encoder — matches checkpoints trained before
# the RNN.py refactor to pure-CNN.  Used only for loading old .pt files.
# ---------------------------------------------------------------------------
class _LegacyTrajectoryEncoder(nn.Module):
    def __init__(self, in_channels=10, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(input_size=32, hidden_size=hidden_dim,
                          num_layers=1, batch_first=True)
        self.output_dim = hidden_dim

    def forward(self, x, lengths, return_last=False):
        x = self.feature_extractor(x).transpose(1, 2).contiguous()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed = nn.utils.rnn.PackedSequence(
            packed.data.contiguous(), packed.batch_sizes,
            packed.sorted_indices, packed.unsorted_indices,
        )
        packed_out, h_n = self.rnn(packed)
        if return_last:
            return h_n[-1]
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out


class _LegacyFusionModel(nn.Module):
    """FusionModel backed by the old GRU encoder, for loading pre-refactor checkpoints."""
    def __init__(self, num_rules=2, num_geom_classes=3, hidden_dim=64, cad_embed_dim=256):
        super().__init__()
        self.num_rules = num_rules
        self.traj_encoder = _LegacyTrajectoryEncoder(in_channels=10, hidden_dim=hidden_dim)
        self.cad_encoder = CADEncoder(embed_dim=cad_embed_dim)
        self.fusion = nn.Sequential(nn.Linear(hidden_dim + cad_embed_dim, hidden_dim), nn.ReLU())
        self.shared_fc = nn.Sequential(nn.Linear(hidden_dim, 16), nn.ReLU(), nn.Dropout(0.1))
        self.continuous_head = nn.Linear(16, num_rules)
        self.discrete_heads = nn.ModuleList([nn.Linear(16, num_geom_classes) for _ in range(num_rules)])

    def forward(self, traj, lengths, cad):
        traj_feat = self.traj_encoder(traj, lengths, return_last=True)
        cad_feat, _ = self.cad_encoder(cad)
        fused = self.fusion(torch.cat([traj_feat, cad_feat], dim=-1))
        shared = self.shared_fc(fused)
        return self.continuous_head(shared), [h(shared) for h in self.discrete_heads]


def _num_geom_classes(class_names):
    if isinstance(class_names, dict):
        return max(len(v) for v in class_names.values())
    return len(class_names)


def _load_model(ckpt, rule_names, class_names, device):
    """Load model from checkpoint, falling back to legacy GRU architecture if needed."""
    n = _num_geom_classes(class_names)
    model = FusionModel(num_rules=len(rule_names), num_geom_classes=n).to(device)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
        return model
    except RuntimeError:
        pass
    legacy = _LegacyFusionModel(num_rules=len(rule_names), num_geom_classes=n).to(device)
    legacy.load_state_dict(ckpt["model_state_dict"])
    print("  (loaded with legacy GRU architecture — checkpoint predates RNN.py refactor)")
    return legacy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_cad(obj_path, n_points=1024):
    cache_path = obj_path.replace(".obj", f"_{n_points}pts.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path).astype(np.float32)
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    points = points - points.mean(axis=0)
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points = points / scale
    points = points.astype(np.float32)
    np.save(cache_path, points)
    return points


def _geom_label_vc(s):
    """Map vel_scale / ori_x geometry label string → int (straight=0, corner=1, none=2)."""
    return 0 if s == "straight" else (1 if s == "corner" else 2)


def _geom_label_prox(s):
    """Map proximity geometry label string → int (edge=0, crossing=1, none=2)."""
    return 0 if s == "edge" else (1 if s == "crossing" else 2)


def _upsample(time, coords, quats, target_dt):
    """Resample to uniform target_dt: linear interp for position, SLERP for quaternions."""
    t_new = np.arange(time[0], time[-1], target_dt)
    if len(t_new) < 2:
        return time, coords, quats
    new_coords = interp1d(time, coords, axis=0, kind="linear")(t_new)
    new_quats = Slerp(time, R.from_quat(quats))(t_new).as_quat()
    return t_new, new_coords, new_quats.astype(np.float32)


class RealFusionDataset(Dataset):
    """
    Multi-trajectory dataset for real experiment data.

    Groups rows by (window_id, stroke_id) — one sample per trajectory.
    Each trajectory shares the same CAD model (looked up by window_id).
    Pass upsample_dt (seconds) to resample sparse trajectories before feature extraction.
    """

    def __init__(self, csv_file, cad_root, feature_stats, target_stats, n_points=1024, upsample_dt=None, position_scale=1.0):
        data = pd.read_csv(csv_file)

        feat_mean, feat_std, time_mean, time_std = feature_stats
        target_mean, target_std = target_stats

        has_prox = "geometric_proximity" in data.columns

        # Pre-load CAD per unique window_id
        cad_cache = {}
        for wid in data["window_id"].unique():
            obj_path = os.path.join(cad_root, str(wid), f"{wid}.obj")
            if not os.path.exists(obj_path):
                raise FileNotFoundError(
                    f"CAD file not found: {obj_path}\n"
                    f"Create: mkdir -p {cad_root}/{wid} && ln -s <obj> {obj_path}"
                )
            cad_cache[wid] = _load_cad(obj_path, n_points).T  # [3, N]

        self.samples = []
        for (wid, sid), group in data.groupby(["window_id", "stroke_id"], sort=True):
            group = group.reset_index(drop=True)

            coords = group[["x", "y", "z"]].values.astype(np.float32) * position_scale
            time = group["time(s)"].values.astype(np.float32)
            quats = group[["qx", "qy", "qz", "qw"]].values.astype(np.float32)

            if upsample_dt is not None and len(time) >= 2:
                time, coords, quats = _upsample(time, coords, quats, upsample_dt)

            time = time.reshape(-1, 1)
            p_delta = np.diff(coords, axis=0)
            t_delta = np.diff(time, axis=0)
            R_mats = R.from_quat(quats).as_matrix()
            feat_R = R_mats[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)[:-1]
            safe_t = np.where(t_delta == 0, 1e-6, t_delta)
            vel = p_delta / safe_t
            features = np.concatenate([t_delta, vel, feat_R], axis=1).astype(np.float32)

            seq = features.copy()
            seq[:, 0] = (seq[:, 0] - time_mean) / time_std
            seq[:, 1:4] = (seq[:, 1:4] - feat_mean) / feat_std

            last = group.iloc[-1]
            vel_scale_raw = float(last["rule_vel_scale"])
            ori_x_raw = float(last["rule_ori_x"])
            prox_raw = float(last["geometric_proximity"]) if has_prox else 1.0

            cont_raw = np.array([vel_scale_raw, ori_x_raw, prox_raw], dtype=np.float32)
            # target_stats may be 2-element (old checkpoint) or 3-element
            if len(target_mean) == 2:
                cont_norm = (cont_raw[:2] - target_mean) / target_std
                cont_norm = np.append(cont_norm, 0.0).astype(np.float32)
            else:
                cont_norm = (cont_raw - target_mean) / target_std

            vel_geom = _geom_label_vc(str(last["rule_vel_scale_geom"]))
            ori_geom = _geom_label_vc(str(last["rule_ori_x_geom"]))
            prox_geom = _geom_label_prox(str(last["geometric_proximity_geom"])) if has_prox else 2

            self.samples.append({
                "seq": seq,
                "actual_len": len(seq),
                "cont_norm": cont_norm,
                "cont_raw": cont_raw,
                "vel_geom": vel_geom,
                "ori_geom": ori_geom,
                "prox_geom": prox_geom,
                "cad": cad_cache[wid],
                "window_id": wid,
                "stroke_id": sid,
            })

        print(f"Loaded {csv_file}: {len(self.samples)} trajectories")
        for s in self.samples:
            print(
                f"  window={s['window_id']} stroke={s['stroke_id']}  "
                f"len={s['actual_len']}  "
                f"vel_scale={s['cont_raw'][0]:.2f}  "
                f"vel_geom={CLASS_NAMES[s['vel_geom']]}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s["seq"], dtype=torch.float32).T,        # [C, T]
            torch.tensor(s["cont_norm"], dtype=torch.float32),     # [3]
            torch.tensor(s["vel_geom"], dtype=torch.long),
            torch.tensor(s["ori_geom"], dtype=torch.long),
            torch.tensor(s["prox_geom"], dtype=torch.long),
            torch.tensor(s["actual_len"], dtype=torch.long),
            torch.tensor(s["cad"], dtype=torch.float32),           # [3, N]
        )


def _collate(batch):
    seqs, cont, vel_geom, ori_geom, prox_geom, lengths, cads = zip(*batch)
    max_len = max(s.shape[1] for s in seqs)
    padded = torch.zeros(len(seqs), seqs[0].shape[0], max_len)
    for i, s in enumerate(seqs):
        padded[i, :, :s.shape[1]] = s
    return (
        padded,
        torch.stack(cont),
        torch.stack(vel_geom),
        torch.stack(ori_geom),
        torch.stack(prox_geom),
        torch.stack(lengths),
        torch.stack(cads),
    )


def _forward(model, batch, device):
    seq, cont_targets, vel_geom_target, ori_geom_target, prox_geom_target, lengths, cad = batch
    seq = seq.to(device)
    lengths = lengths.to(device)
    cad = cad.to(device)
    cont_targets = cont_targets.to(device)
    geom_targets = [
        vel_geom_target.to(device),
        ori_geom_target.to(device),
        prox_geom_target.to(device),
    ]
    cont_preds, geom_logits = model(seq, lengths, cad)
    return cont_preds, geom_logits, cont_targets, geom_targets


def eval_real_data(
    real_csv="datasets/real_experiment/combined.csv",
    cad_root="measurements",
    model_path="outputs/windows_v2_fusion_test/best_fusion_model.pt",
    output_dir="outputs/real_experiment_eval",
    headless=True,
    upsample_dt=None,
    position_scale=1.0,
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Device: {device}")
    print(f"Loading checkpoint: {model_path}\n")

    ckpt = torch.load(model_path, map_location=device)
    feature_stats = (
        np.array(ckpt["feature_mean"], dtype=np.float32),
        np.array(ckpt["feature_std"], dtype=np.float32),
        float(ckpt["time_mean"]),
        float(ckpt["time_std"]),
    )
    target_stats = (
        np.array(ckpt["target_mean"], dtype=np.float32),
        np.array(ckpt["target_std"], dtype=np.float32),
    )
    rule_names = ckpt.get("rule_names", ACTIVE_RULES)
    class_names = ckpt.get("rule_class_names", ckpt.get("class_names", RULE_CLASS_NAMES))

    dataset = RealFusionDataset(real_csv, cad_root, feature_stats, target_stats, upsample_dt=upsample_dt, position_scale=position_scale)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=_collate)

    model = _load_model(ckpt, rule_names, class_names, device)
    model.eval()

    target_mean, target_std = target_stats

    # Per-rule class name lookup
    if isinstance(class_names, dict):
        cn_per_rule = class_names
    else:
        cn_per_rule = {r: RULE_CLASS_NAMES.get(r, CLASS_NAMES) for r in rule_names}

    # ── Raw output per trajectory ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RAW MODEL OUTPUT  (per trajectory)")
    print("=" * 60)
    geom_true_keys = ["vel_geom", "ori_geom", "prox_geom"]
    with torch.no_grad():
        for s in dataset.samples:
            seq = torch.tensor(s["seq"], dtype=torch.float32).T.unsqueeze(0).to(device)
            length = torch.tensor([s["actual_len"]], dtype=torch.long).to(device)
            cad = torch.tensor(s["cad"], dtype=torch.float32).unsqueeze(0).to(device)
            cont_preds, geom_logits = model(seq, length, cad)
            pred = cont_preds.cpu().numpy()[0] * target_std + target_mean

            print(f"\n  window={s['window_id']} stroke={s['stroke_id']}  ({s['actual_len']} steps)")
            for i, rule_name in enumerate(rule_names):
                cn = cn_per_rule.get(rule_name, CLASS_NAMES)
                probs = torch.softmax(geom_logits[i], dim=1).cpu().numpy()[0]
                pred_cls = geom_logits[i].argmax(dim=1).item()
                true_cls = s[geom_true_keys[i]] if i < len(geom_true_keys) else 2
                true_lbl = cn[true_cls] if true_cls < len(cn) else str(true_cls)
                pred_lbl = cn[pred_cls] if pred_cls < len(cn) else str(pred_cls)
                print(f"    [{rule_name}]  true={s['cont_raw'][i]:.4f}  pred={pred[i]:.4f}  "
                      f"geom_true={true_lbl}  geom_pred={pred_lbl}  "
                      f"probs={[round(float(p),3) for p in probs]}")

    # ── Aggregate metrics ────────────────────────────────────────────────────
    metrics = evaluate(
        model, lambda b, d: _forward(model, b, d), loader, device,
        target_mean=target_mean, target_std=target_std,
        rule_names=rule_names, class_names=class_names,
    )

    print("\n" + "=" * 60)
    print("EVALUATION METRICS  (all trajectories)")
    print("=" * 60)
    for rule_name in rule_names:
        rm = metrics["per_rule"][rule_name]
        print(f"\n[ {rule_name} ]")
        print(f"  MAE:           {rm['mae']:.4f}")
        print(f"  RMSE:          {rm['rmse']:.4f}")
        print(f"  Geom accuracy: {rm['geom_accuracy']:.4f}")
        print(f"  Geom macro F1: {rm['geom_macro_f1']:.4f}")
        print(rm["geom_classification_report"])

    print("[ Combined Geom Metrics (All Rules) ]")
    print(f"  Accuracy:  {metrics['combined_geom_accuracy']:.4f}")
    print(f"  Precision: {metrics['combined_geom_precision']:.4f}")
    print(f"  Recall:    {metrics['combined_geom_recall']:.4f}")
    print(f"  F1 Score:  {metrics['combined_geom_f1']:.4f}")

    vel_cn = metrics["per_rule"]["vel_scale"]["class_names"]
    plot_scatter_plot(
        metrics["vel_true_denorm"], metrics["vel_pred_denorm"],
        metrics["geom_true"], metrics["geom_pred"],
        headless=headless, output_dir=output_dir,
    )
    plot_confusion_matrix(
        metrics["geom_confusion_matrix"], class_names=vel_cn,
        normalize=False, headless=headless, output_dir=output_dir,
    )
    plot_confusion_matrix(
        metrics["geom_confusion_matrix"], class_names=vel_cn,
        normalize=True, headless=headless, output_dir=output_dir,
    )
    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-csv", default="datasets/real_experiment/combined.csv")
    parser.add_argument("--cad-root", default="measurements")
    parser.add_argument("--model-path", default="outputs/windows_v2_fusion_test/best_fusion_model.pt")
    parser.add_argument("--output-dir", default="outputs/real_experiment_eval")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--position-scale", type=float, default=1.0,
                        help="Multiply raw positions by this factor before feature extraction. "
                             "Use 1000.0 to convert real-data metres to the mm scale the model was trained on.")
    parser.add_argument("--upsample-dt", type=float, default=None,
                        help="Resample trajectories to this fixed dt in seconds (e.g. 0.02 for 50 Hz). "
                             "Default: no upsampling.")
    args = parser.parse_args()

    eval_real_data(
        real_csv=args.real_csv,
        cad_root=args.cad_root,
        model_path=args.model_path,
        output_dir=args.output_dir,
        headless=args.headless,
        upsample_dt=args.upsample_dt,
        position_scale=args.position_scale,
    )
