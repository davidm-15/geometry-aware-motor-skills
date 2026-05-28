"""
2x2 scatter plot: L-shape (top) vs Window (bottom), Straight (left) vs Corner (right).
Velocity scale = blue circles, Orientation = orange triangles.

Run from the project root:
    python -m scripts.plot_2x2_scatter
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from model.fusion_model import FusionModel
from training.dataset import FusionDataset, ACTIVE_RULES, RULE_CLASS_NAMES
from training.evaluate import evaluate
from training.train import _fusion_forward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_eval(ckpt_path, dataset_path):
    ckpt = torch.load(ckpt_path, map_location=device)

    feature_stats = (
        np.array(ckpt["feature_mean"], dtype=np.float32),
        np.array(ckpt["feature_std"],  dtype=np.float32),
        float(ckpt["time_mean"]),
        float(ckpt["time_std"]),
    )
    target_stats = (
        np.array(ckpt["target_mean"], dtype=np.float32),
        np.array(ckpt["target_std"],  dtype=np.float32),
    )
    rule_names  = ckpt.get("rule_names", ACTIVE_RULES)
    class_names = ckpt.get("rule_class_names", ckpt.get("class_names", RULE_CLASS_NAMES))

    test_ds = FusionDataset(
        os.path.join(dataset_path, "test.csv"),
        cad_root=dataset_path,
        feature_stats=feature_stats,
        target_stats=target_stats,
    )
    loader = DataLoader(
        test_ds, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    num_geom_classes = (
        max(len(v) for v in class_names.values())
        if isinstance(class_names, dict) else len(class_names)
    )
    model = FusionModel(num_rules=len(rule_names), num_geom_classes=num_geom_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = evaluate(
        model, lambda b, d: _fusion_forward(model, b, d), loader, device,
        target_mean=target_stats[0], target_std=target_stats[1],
        rule_names=rule_names, class_names=class_names,
    )
    return metrics, rule_names


def print_metrics(metrics, rule_names, label=""):
    print(f"\n=== TEST RESULTS{' — ' + label if label else ''} (FusionModel) ===")
    for rule_name in rule_names:
        rule_metrics = metrics["per_rule"][rule_name]
        print(f"\n[ {rule_name} ]")
        print(f"  MAE:       {rule_metrics['mae']:.4f}")
        print(f"  RMSE:      {rule_metrics['rmse']:.4f}")
        print(f"  Geom accuracy: {rule_metrics['geom_accuracy']:.4f}")
        print(f"  Geom macro F1: {rule_metrics['geom_macro_f1']:.4f}")
        print(rule_metrics["geom_classification_report"])
    print("[ Combined Geom Metrics (All Rules) ]")
    print(f"  Accuracy:  {metrics['combined_geom_accuracy']:.4f}")
    print(f"  Precision: {metrics['combined_geom_precision']:.4f}")
    print(f"  Recall:    {metrics['combined_geom_recall']:.4f}")
    print(f"  F1 Score:  {metrics['combined_geom_f1']:.4f}")


def plot_2x2(l_metrics, w_metrics, rule_names, out="outputs/plots/2x2_scatter.png"):
    rule_style = {
        "vel_scale": dict(marker="o", color="#1f77b4", label="Velocity scale", s=14),
        "ori_x":     dict(marker="^", color="#ff7f0e", label="Orientation",    s=18),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    panels = [
        (axes[0, 0], l_metrics, 0, "L-shape – Straight"),
        (axes[0, 1], l_metrics, 1, "L-shape – Corner"),
        (axes[1, 0], w_metrics, 0, "Window – Straight"),
        (axes[1, 1], w_metrics, 1, "Window – Corner"),
    ]

    for ax, metrics, geom_cls, title in panels:
        all_true, all_pred = [], []

        for rule in ["vel_scale", "ori_x"]:
            if rule not in rule_names:
                continue
            style = rule_style[rule]
            pr    = metrics["per_rule"][rule]
            mask  = pr["geom_true"] == geom_cls
            if not np.any(mask):
                continue
            true_vals = pr["cont_true_denorm"][mask]
            pred_vals = pr["cont_pred_denorm"][mask]
            ax.scatter(
                true_vals, pred_vals,
                marker=style["marker"], color=style["color"],
                alpha=0.5, s=style["s"], label=style["label"],
            )
            all_true.append(true_vals)
            all_pred.append(pred_vals)

        if all_true:
            all_v = np.concatenate(all_true + all_pred)
            lo, hi = all_v.min(), all_v.max()
            pad = (hi - lo) * 0.05
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", linewidth=1)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Actual value")
        ax.set_ylabel("Predicted value")
        ax.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200)
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("Evaluating L-shape model...")
    l_metrics, rule_names = load_and_eval(
        "outputs/L_shape_fusion_test/best_fusion_model.pt",
        "datasets/L_shape",
    )
    print_metrics(l_metrics, rule_names, label="L-shape")

    print("Evaluating Window model...")
    w_metrics, _ = load_and_eval(
        "outputs/windows_fusion_test/best_fusion_model.pt",
        "datasets/windows-v2",
    )
    print_metrics(w_metrics, rule_names, label="Window")

    plot_2x2(l_metrics, w_metrics, rule_names)
