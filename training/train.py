import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.fusion_model import FusionModel
from training.dataset import FusionDataset, ACTIVE_RULES, CLASS_NAMES, RULE_CLASS_NAMES
from training.evaluate import evaluate, plot_confusion_matrix, plot_scatter_plot

# python -m training.train
# python -m training.train --eval-only
# python -m training.train --dataset-path datasets/windows-v2 --output-dir outputs/windows_v2_fusion_test
# python -m training.train --dataset-path datasets/window_cross --output-dir outputs/window_cross_fusion_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fusion_forward(model, batch, device):
    seq, cont_targets, vel_geom, ori_geom, prox_geom, lengths, cad = batch
    seq          = seq.to(device, non_blocking=True)
    lengths      = lengths.to(device, non_blocking=True)
    cad          = cad.to(device, non_blocking=True)
    cont_targets = cont_targets.to(device, non_blocking=True)
    geom_targets = [
        vel_geom.to(device,  non_blocking=True),
        ori_geom.to(device,  non_blocking=True),
        prox_geom.to(device, non_blocking=True),
    ]
    cont_preds, geom_logits = model(seq, lengths, cad)
    return cont_preds, geom_logits, cont_targets, geom_targets


def train_one_epoch(model, loader, optimizer, reg_criterion, cls_criterion, device, alpha_reg=1.0, alpha_cls=1.0):
    model.train()

    total_loss = 0.0
    total_reg_loss = 0.0
    total_cls_loss = 0.0
    total_samples = 0
    total_correct = np.zeros(model.num_rules, dtype=np.int64)

    for batch in loader:
        seq, cont_targets, vel_geom_target, ori_geom_target, prox_geom_target, lengths, cad = batch
        seq = seq.to(device, non_blocking=True)
        cont_targets = cont_targets.to(device, non_blocking=True)
        geom_targets = [
            vel_geom_target.to(device, non_blocking=True),
            ori_geom_target.to(device, non_blocking=True),
            prox_geom_target.to(device, non_blocking=True),
        ]
        lengths = lengths.to(device, non_blocking=True)
        cad = cad.to(device, non_blocking=True)

        optimizer.zero_grad()

        cont_preds, geom_logits = model(seq, lengths, cad)

        reg_loss = reg_criterion(cont_preds, cont_targets)
        cls_loss = sum(
            cls_criterion(rule_logits, rule_target)
            for rule_logits, rule_target in zip(geom_logits, geom_targets)
        )
        loss = alpha_reg * reg_loss + alpha_cls * cls_loss

        loss.backward()
        optimizer.step()

        batch_size = seq.size(0)
        total_loss += loss.item() * batch_size
        total_reg_loss += reg_loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_samples += batch_size
        for i, (rule_logits, rule_target) in enumerate(zip(geom_logits, geom_targets)):
            total_correct[i] += (rule_logits.argmax(dim=1) == rule_target).sum().item()

    return {
        "loss": total_loss / total_samples,
        "reg_loss": total_reg_loss / total_samples,
        "cls_loss": total_cls_loss / total_samples,
        "geom_acc": total_correct / total_samples,
    }


def train(
    output_dir="outputs/L_shape_fusion_test",
    dataset_path="datasets/L_shape",
    headless=False,
    eval_only=False,
    epochs=200,
):
    cad_root = dataset_path

    train_csv = os.path.join(dataset_path, "train.csv")
    val_csv = os.path.join(dataset_path, "val.csv")
    test_csv = os.path.join(dataset_path, "test.csv")

    os.makedirs(output_dir, exist_ok=True)

    ckpt_path = os.path.join(output_dir, "best_fusion_model.pt")

    if eval_only:
        ckpt = torch.load(ckpt_path, map_location=device)

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

        test_dataset = FusionDataset(
            test_csv, cad_root=cad_root,
            feature_stats=feature_stats, target_stats=target_stats,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )

        num_geom_classes = max(len(v) for v in class_names.values()) if isinstance(class_names, dict) else len(class_names)
        model = FusionModel(num_rules=len(rule_names), num_geom_classes=num_geom_classes).to(device)
        try:
            model.load_state_dict(ckpt["model_state_dict"])
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint is incompatible with the current FusionModel layout. "
                "Run without --eval-only to retrain."
            ) from exc

    else:
        train_dataset = FusionDataset(train_csv, cad_root=cad_root)
        feature_stats = train_dataset.feature_stats
        target_stats = train_dataset.target_stats

        val_dataset = FusionDataset(
            val_csv, cad_root=cad_root,
            feature_stats=feature_stats, target_stats=target_stats,
        )
        test_dataset = FusionDataset(
            test_csv, cad_root=cad_root,
            feature_stats=feature_stats, target_stats=target_stats,
        )

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        rule_names = ACTIVE_RULES
        class_names = RULE_CLASS_NAMES
        num_geom_classes = max(len(v) for v in class_names.values())
        model = FusionModel(num_rules=len(rule_names), num_geom_classes=num_geom_classes).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        reg_criterion = nn.HuberLoss(delta=1.0)
        cls_criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")

        for epoch in range(epochs):
            train_stats = train_one_epoch(
                model, train_loader, optimizer, reg_criterion, cls_criterion, device,
            )

            val_metrics = evaluate(
                model, lambda b, d: _fusion_forward(model, b, d), val_loader, device,
                target_mean=target_stats[0], target_std=target_stats[1],
                rule_names=rule_names, class_names=class_names,
            )

            val_loss = val_metrics["combined_cont_mae"]

            print(
                f"Epoch {epoch+1:03d} | loss {train_stats['loss']:.4f} "
                f"| val MAE {val_metrics['vel_mae']:.4f} "
                f"| val combined MAE {val_metrics['combined_cont_mae']:.4f} "
                f"| val geom F1 {val_metrics['geom_macro_f1']:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "feature_mean": feature_stats[0].tolist(),
                    "feature_std": feature_stats[1].tolist(),
                    "time_mean": float(feature_stats[2]),
                    "time_std": float(feature_stats[3]),
                    "target_mean": target_stats[0].tolist(),
                    "target_std": target_stats[1].tolist(),
                    "rule_names": rule_names,
                    "rule_class_names": class_names,
                }, ckpt_path)

        ckpt = torch.load(ckpt_path, map_location=device)
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
        model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(
        model, lambda b, d: _fusion_forward(model, b, d), test_loader, device,
        target_mean=target_stats[0], target_std=target_stats[1],
        rule_names=rule_names, class_names=class_names,
    )

    print("\n=== TEST RESULTS (FusionModel) ===")
    for rule_name in rule_names:
        rule_metrics = test_metrics["per_rule"][rule_name]
        print(f"\n[ {rule_name} ]")
        print(f"  MAE:       {rule_metrics['mae']:.4f}")
        print(f"  RMSE:      {rule_metrics['rmse']:.4f}")
        print(f"  Geom accuracy: {rule_metrics['geom_accuracy']:.4f}")
        print(f"  Geom macro F1: {rule_metrics['geom_macro_f1']:.4f}")
        print(rule_metrics["geom_classification_report"])

    print("[ Combined Geom Metrics (All Rules) ]")
    print(f"  Accuracy:  {test_metrics['combined_geom_accuracy']:.4f}")
    print(f"  Precision: {test_metrics['combined_geom_precision']:.4f}")
    print(f"  Recall:    {test_metrics['combined_geom_recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['combined_geom_f1']:.4f}")

    vel_cn = test_metrics["per_rule"]["vel_scale"]["class_names"]
    plot_scatter_plot(
        test_metrics["vel_true_denorm"], test_metrics["vel_pred_denorm"],
        test_metrics["geom_true"], test_metrics["geom_pred"],
        headless=headless, output_dir=output_dir,
    )
    plot_confusion_matrix(test_metrics["geom_confusion_matrix"], class_names=vel_cn, normalize=False, headless=headless, output_dir=output_dir)
    plot_confusion_matrix(test_metrics["geom_confusion_matrix"], class_names=vel_cn, normalize=True, headless=headless, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train FusionModel (RNN + CAD encoder) on trajectory data."
    )
    parser.add_argument("--output-dir", type=str, default="outputs/L_shape_fusion_test")
    parser.add_argument("--dataset-path", type=str, default="datasets/L_shape", help="Path to dataset root containing train.csv, val.csv, test.csv, and CAD OBJ files.")
    parser.add_argument("--headless", action="store_false", help="Run in headless mode (no plots shown, only saved)")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate on test set using saved checkpoint")

    args = parser.parse_args()
    train(
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        headless=args.headless,
        eval_only=args.eval_only,
    )
