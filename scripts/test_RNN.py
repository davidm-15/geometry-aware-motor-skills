import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.RNN import TrajectoryEncoder
from training.dataset import TrajDataset, ACTIVE_RULES, CLASS_NAMES
from training.evaluate import evaluate, plot_confusion_matrix, plot_scatter_plot


"""
This file tests the TrajectoryEncoder on the same rule task as rule_regression:
- regression: rule_vel_scale, rule_ori_x
- classification: rule_vel_scale_geom, rule_ori_x_geom
"""

# python -m scripts.test_RNN
# python -m scripts.test_RNN --eval-only

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNNHeadModel(nn.Module):
    def __init__(self, hidden_dim=64, num_rules=len(ACTIVE_RULES), num_geom_classes=3):
        super().__init__()
        self.num_rules = num_rules

        self.encoder = TrajectoryEncoder(
            in_channels=10,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            project=False,
        )

        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.continuous_head = nn.Linear(16, num_rules)
        self.discrete_heads = nn.ModuleList(
            [nn.Linear(16, num_geom_classes) for _ in range(num_rules)]
        )

    def forward(self, x, lengths):
        final_feats = self.encoder(x, lengths, return_last=True)
        final_feats = self.shared_fc(final_feats)
        cont_preds = self.continuous_head(final_feats)
        geom_logits = [head(final_feats) for head in self.discrete_heads]
        return cont_preds, geom_logits


def _rnn_forward(model, batch, device):
    seq, cont_targets, vel_geom_target, ori_geom_target, lengths = batch
    seq = seq.to(device, non_blocking=True)
    lengths = lengths.to(device, non_blocking=True)
    cont_targets = cont_targets.to(device, non_blocking=True)
    geom_targets = [
        vel_geom_target.to(device, non_blocking=True),
        ori_geom_target.to(device, non_blocking=True),
    ]
    cont_preds, geom_logits = model(seq, lengths)
    return cont_preds, geom_logits, cont_targets, geom_targets


def train_one_epoch(model, loader, optimizer, reg_criterion, cls_criterion, device, alpha_reg=1.0, alpha_cls=1.0):
    model.train()

    total_loss = 0.0
    total_reg_loss = 0.0
    total_cls_loss = 0.0
    total_samples = 0
    total_correct = np.zeros(model.num_rules, dtype=np.int64)

    for batch in loader:
        seq, cont_targets, vel_geom_target, ori_geom_target, lengths = batch
        seq = seq.to(device, non_blocking=True)
        cont_targets = cont_targets.to(device, non_blocking=True)
        geom_targets = [
            vel_geom_target.to(device, non_blocking=True),
            ori_geom_target.to(device, non_blocking=True),
        ]
        lengths = lengths.to(device, non_blocking=True)

        optimizer.zero_grad()

        cont_preds, geom_logits = model(seq, lengths)

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


def test_rnn(output_dir="outputs/L_shape_rnn_test", headless=False, eval_only=False):
    dataset_path = "datasets/L_shape"

    train_csv = os.path.join(dataset_path, "train.csv")
    val_csv = os.path.join(dataset_path, "val.csv")
    test_csv = os.path.join(dataset_path, "test.csv")

    class_names = CLASS_NAMES
    os.makedirs(output_dir, exist_ok=True)

    ckpt_path = os.path.join(output_dir, "best_rnn_model.pt")

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
        class_names = ckpt.get("class_names", CLASS_NAMES)
        rule_names = ckpt.get("rule_names", ACTIVE_RULES)

        test_dataset = TrajDataset(test_csv, feature_stats=feature_stats, target_stats=target_stats)
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )

        model = RNNHeadModel(num_rules=len(rule_names), num_geom_classes=len(class_names)).to(device)
        try:
            model.load_state_dict(ckpt["model_state_dict"])
        except RuntimeError as exc:
            raise RuntimeError(
                "The checkpoint was created by an old RNN head/encoder layout. "
                "Run without --eval-only once to train a compatible checkpoint."
            ) from exc

    else:
        train_dataset = TrajDataset(train_csv)
        feature_stats = train_dataset.feature_stats
        target_stats = train_dataset.target_stats

        val_dataset = TrajDataset(val_csv, feature_stats=feature_stats, target_stats=target_stats)
        test_dataset = TrajDataset(test_csv, feature_stats=feature_stats, target_stats=target_stats)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        rule_names = ACTIVE_RULES
        model = RNNHeadModel(num_rules=len(rule_names), num_geom_classes=len(class_names)).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        reg_criterion = nn.HuberLoss(delta=1.0)
        cls_criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        epochs = 200

        for epoch in range(epochs):
            train_stats = train_one_epoch(
                model, train_loader, optimizer, reg_criterion, cls_criterion, device,
            )

            val_metrics = evaluate(
                model, _rnn_forward, val_loader, device,
                target_mean=target_stats[0], target_std=target_stats[1],
                rule_names=rule_names, class_names=class_names,
            )

            val_loss = val_metrics["combined_cont_mae"]

            if (epoch + 1) % 10 == 0:
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
                    "class_names": class_names,
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
        class_names = ckpt.get("class_names", CLASS_NAMES)
        rule_names = ckpt.get("rule_names", ACTIVE_RULES)
        model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(
        model, _rnn_forward, test_loader, device,
        target_mean=target_stats[0], target_std=target_stats[1],
        rule_names=rule_names, class_names=class_names,
    )

    print("\n=== TEST RESULTS ===")
    print(f"Velocity MAE:         {test_metrics['vel_mae']:.4f}")
    print(f"Velocity RMSE:        {test_metrics['vel_rmse']:.4f}")
    print(f"Geom accuracy:        {test_metrics['geom_accuracy']:.4f}")
    print(f"Geom macro F1:        {test_metrics['geom_macro_f1']:.4f}")
    print(f"Geom macro recall:    {test_metrics['geom_macro_recall']:.4f}")
    print(test_metrics["geom_classification_report"])
    if len(rule_names) > 1:
        for rule_name in rule_names[1:]:
            rule_metrics = test_metrics["per_rule"][rule_name]
            print(f"[ Classification Metrics ({rule_name}) ]")
            print(f"Accuracy:  {rule_metrics['geom_accuracy']:.4f}")
            print(f"Precision: {rule_metrics['geom_weighted_precision']:.4f}")
            print(f"Recall:    {rule_metrics['geom_weighted_recall']:.4f}")
            print(f"F1 Score:  {rule_metrics['geom_weighted_f1']:.4f}")
        print("[ Classification Metrics (All Rules Combined) ]")
        print(f"Accuracy:  {test_metrics['combined_geom_accuracy']:.4f}")
        print(f"Precision: {test_metrics['combined_geom_precision']:.4f}")
        print(f"Recall:    {test_metrics['combined_geom_recall']:.4f}")
        print(f"F1 Score:  {test_metrics['combined_geom_f1']:.4f}")

    plot_scatter_plot(
        test_metrics["vel_true_denorm"], test_metrics["vel_pred_denorm"],
        test_metrics["geom_true"], test_metrics["geom_pred"],
        headless=headless, output_dir=output_dir,
    )
    plot_confusion_matrix(test_metrics["geom_confusion_matrix"], class_names=class_names, normalize=False, headless=headless, output_dir=output_dir)
    plot_confusion_matrix(test_metrics["geom_confusion_matrix"], class_names=class_names, normalize=True, headless=headless, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test RNN model on trajectory data with regression and classification heads."
    )
    parser.add_argument("--output-dir", type=str, default="outputs/L_shape_rnn_test")
    parser.add_argument("--headless", action="store_false", help="Run in headless mode (no plots shown, only saved)")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation on test set using best model (no training)")

    args = parser.parse_args()
    test_rnn(output_dir=args.output_dir, headless=args.headless, eval_only=args.eval_only)
