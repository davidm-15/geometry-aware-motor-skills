import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    mean_absolute_error,
    mean_squared_error,
)
import torch

from training.dataset import ACTIVE_RULES, CLASS_NAMES, RULE_CLASS_NAMES


def evaluate(model, forward_fn, loader, device, target_mean, target_std, rule_names=None, class_names=None):
    """
    Evaluate model on loader.

    forward_fn(batch, device) must return:
        (cont_preds [B, R], geom_logits list[R x [B, C]], cont_targets [B, R], geom_targets list[R x [B]])

    class_names: list used for all rules, OR dict mapping rule_name -> list for per-rule names.
    """
    model.eval()

    rule_names = rule_names or ACTIVE_RULES
    # Build per-rule class name lookup
    if isinstance(class_names, dict):
        cn_per_rule = class_names
    else:
        fallback = class_names or CLASS_NAMES
        cn_per_rule = {r: RULE_CLASS_NAMES.get(r, fallback) for r in rule_names}

    cont_preds_all = []
    cont_targets_all = []
    geom_preds_all = [[] for _ in rule_names]
    geom_targets_all = [[] for _ in rule_names]

    with torch.no_grad():
        for batch in loader:
            cont_preds, geom_logits, cont_targets, geom_targets = forward_fn(batch, device)

            cont_preds_all.append(cont_preds.cpu().numpy())
            cont_targets_all.append(cont_targets.cpu().numpy())
            for i, (rule_logits, rule_target) in enumerate(zip(geom_logits, geom_targets)):
                geom_preds_all[i].append(rule_logits.argmax(dim=1).cpu().numpy())
                geom_targets_all[i].append(rule_target.cpu().numpy())

    cont_pred = np.concatenate(cont_preds_all, axis=0)
    cont_true = np.concatenate(cont_targets_all, axis=0)
    geom_pred = [np.concatenate(p, axis=0) for p in geom_preds_all]
    geom_true = [np.concatenate(t, axis=0) for t in geom_targets_all]

    target_mean = np.asarray(target_mean, dtype=np.float32)
    target_std = np.asarray(target_std, dtype=np.float32)
    n = len(target_std)
    cont_pred_denorm = cont_pred[:, :n] * target_std + target_mean
    cont_true_denorm = cont_true[:, :n] * target_std + target_mean

    per_rule = {}
    combined_geom_true = []
    combined_geom_pred = []

    for i, rule_name in enumerate(rule_names):
        rule_true = cont_true_denorm[:, i]
        rule_pred = cont_pred_denorm[:, i]
        rule_geom_true = geom_true[i]
        rule_geom_pred = geom_pred[i]
        cn = cn_per_rule.get(rule_name, CLASS_NAMES)

        precision, recall, f1, support = precision_recall_fscore_support(
            rule_geom_true, rule_geom_pred,
            labels=list(range(len(cn))), average=None, zero_division=0,
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            rule_geom_true, rule_geom_pred,
            labels=list(range(len(cn))), average="macro", zero_division=0,
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            rule_geom_true, rule_geom_pred,
            labels=list(range(len(cn))), average="weighted", zero_division=0,
        )

        per_rule[rule_name] = {
            "mae": mean_absolute_error(rule_true, rule_pred),
            "rmse": np.sqrt(mean_squared_error(rule_true, rule_pred)),
            "geom_accuracy": accuracy_score(rule_geom_true, rule_geom_pred),
            "geom_macro_precision": macro_p,
            "geom_macro_recall": macro_r,
            "geom_macro_f1": macro_f1,
            "geom_weighted_precision": weighted_p,
            "geom_weighted_recall": weighted_r,
            "geom_weighted_f1": weighted_f1,
            "geom_per_class_precision": precision,
            "geom_per_class_recall": recall,
            "geom_per_class_f1": f1,
            "geom_per_class_support": support,
            "geom_confusion_matrix": confusion_matrix(
                rule_geom_true, rule_geom_pred, labels=list(range(len(cn)))
            ),
            "geom_classification_report": classification_report(
                rule_geom_true, rule_geom_pred,
                labels=list(range(len(cn))),
                target_names=cn, digits=4, zero_division=0,
            ),
            "class_names": cn,
            "cont_true_denorm": rule_true,
            "cont_pred_denorm": rule_pred,
            "geom_true": rule_geom_true,
            "geom_pred": rule_geom_pred,
        }

        combined_geom_true.append(rule_geom_true)
        combined_geom_pred.append(rule_geom_pred)

    combined_geom_true = np.concatenate(combined_geom_true, axis=0)
    combined_geom_pred = np.concatenate(combined_geom_pred, axis=0)
    n_combined_classes = max(len(v) for v in class_names.values()) if isinstance(class_names, dict) else len(class_names or CLASS_NAMES)
    combined_p, combined_r, combined_f1, _ = precision_recall_fscore_support(
        combined_geom_true, combined_geom_pred,
        labels=list(range(n_combined_classes)), average="weighted", zero_division=0,
    )

    vel_metrics = per_rule["vel_scale"]

    return {
        "combined_cont_mae": np.mean(np.abs(cont_true_denorm - cont_pred_denorm)),
        "combined_geom_accuracy": accuracy_score(combined_geom_true, combined_geom_pred),
        "combined_geom_precision": combined_p,
        "combined_geom_recall": combined_r,
        "combined_geom_f1": combined_f1,
        "per_rule": per_rule,
        "cont_pred_denorm": cont_pred_denorm,
        "cont_true_denorm": cont_true_denorm,
        "geom_true_by_rule": geom_true,
        "geom_pred_by_rule": geom_pred,
        "vel_mae": vel_metrics["mae"],
        "vel_rmse": vel_metrics["rmse"],
        "geom_accuracy": vel_metrics["geom_accuracy"],
        "geom_macro_precision": vel_metrics["geom_macro_precision"],
        "geom_macro_recall": vel_metrics["geom_macro_recall"],
        "geom_macro_f1": vel_metrics["geom_macro_f1"],
        "geom_confusion_matrix": vel_metrics["geom_confusion_matrix"],
        "geom_classification_report": vel_metrics["geom_classification_report"],
        "vel_pred_denorm": vel_metrics["cont_pred_denorm"],
        "vel_true_denorm": vel_metrics["cont_true_denorm"],
        "geom_true": vel_metrics["geom_true"],
        "geom_pred": vel_metrics["geom_pred"],
    }


def plot_confusion_matrix(cm, class_names=None, normalize=False, output_dir=".", headless=False):
    if normalize:
        cm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest")
    plt.colorbar(im, ax=ax)

    n_classes = cm.shape[0]
    tick_labels = class_names if class_names is not None else list(range(n_classes))
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="Geom Confusion Matrix" + (" (Normalized)" if normalize else ""),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > thresh else "black")

    fig.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "confusion_matrix" + ("_normalized" if normalize else "") + ".png")
    )
    if not headless:
        plt.show()


def plot_scatter_plot(vel_true, vel_pred, geom_true, geom_pred, output_dir=".", headless=False):
    class_names = CLASS_NAMES
    colors = {0: "green", 1: "red", 2: "blue"}
    markers = {0: "o", 1: "s", 2: "^"}

    plt.figure(figsize=(9, 7))

    for true_cls in [0, 1, 2]:
        for pred_cls in [0, 1, 2]:
            mask = (geom_true == true_cls) & (geom_pred == pred_cls)
            if not np.any(mask):
                continue
            label = f"true {class_names[true_cls]} → pred {class_names[pred_cls]}"
            plt.scatter(
                vel_true[mask], vel_pred[mask],
                c=colors[true_cls], marker=markers[pred_cls], alpha=0.65,
                edgecolors="black" if true_cls != pred_cls else "none",
                linewidths=0.8, label=label,
            )

    min_val = min(vel_true.min(), vel_pred.min())
    max_val = max(vel_true.max(), vel_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.7)

    plt.xlabel("Actual Velocity Scale")
    plt.ylabel("Predicted Velocity Scale")
    plt.title("Predicted vs Actual Velocity Scale\nColor = true class, marker = predicted class")
    plt.grid(True)
    plt.legend(fontsize=8, ncols=2)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "scatter_plot.png"), dpi=200)
    if not headless:
        plt.show()
    plt.close()
