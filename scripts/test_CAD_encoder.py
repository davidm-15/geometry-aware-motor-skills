import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import glob
import trimesh
import numpy as np
import json
from collections import Counter
import os
from model.CAD_encoder import CADEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

"""
    This file is for testing the CADEncoder on a simple classification task of predicting the number of crossings in the window skeletons.
"""


# python -m scripts.test_CAD_encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CLFModel(nn.Module):
    def __init__(self, embed_dim=256, num_classes=7) -> None:
        super().__init__()
        
        self.encoder = CADEncoder(embed_dim=embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        out,_ = self.encoder(x)
        return self.head(out)



class CAD_dataset(Dataset):
    def __init__(self, root="datasets/windows-v2", split_json=None, n_points=1024):
        self.samples = []
        self.n_points = n_points
        skeletons = f"{root}/0_skeletons"

        with open(split_json, "r") as f:
            split_names = set(json.load(f))

        window_folders = glob.glob(f"{root}/**/*_wr1fr_1", recursive=True)
        window_folders.sort(key=lambda x: int(x.split("/")[-1].split('_')[0]))



        for folder in window_folders:
            file_name = folder.split("/")[-1]

            if file_name not in split_names:
                continue

            obj_file = f"{folder}/{file_name}.obj"
            json_file = f"{skeletons}/{file_name}.json"

            if not os.path.exists(obj_file) or not os.path.exists(json_file):
                continue

            label = count_crossings_from_json(json_file)
            self.samples.append((obj_file, label, file_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj_file, label, file_name = self.samples[idx]
        points = load_obj_as_pointcloud(obj_file, self.n_points, use_cache=True)
        points = torch.from_numpy(points).transpose(0, 1)
        label = torch.tensor(label, dtype=torch.long)
        return points, label, file_name


            

def count_crossings_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    edges = data["edges"]

    degree = Counter()

    for i, j in edges:
        degree[i] += 1
        degree[j] += 1

    crossings = sum(1 for d in degree.values() if d == 4)

    return crossings


def load_obj_as_pointcloud(path, n_points=1024, use_cache=True):
    cache_path = path.replace(".obj", f"_{n_points}pts.npy")

    if use_cache and os.path.exists(cache_path):
        return np.load(cache_path).astype(np.float32)

    mesh = trimesh.load(path, force='mesh', process=False)
    points, _ = trimesh.sample.sample_surface(mesh, n_points)

    points = points - points.mean(axis=0)
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points = points / scale

    points = points.astype(np.float32)

    if use_cache:
        np.save(cache_path, points)

    return points

def test_encoder():
    test_split_path = "datasets/windows-v2/test_split_original.json"
    train_split_path = "datasets/windows-v2/train_split_original.json"

    train_dataset = CAD_dataset(split_json=train_split_path)
    test_dataset = CAD_dataset(split_json=test_split_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    model = CLFModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 50

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for pointcloud, nodes, _ in train_loader:
            pointcloud = pointcloud.to(device, non_blocking=True)
            nodes = nodes.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(pointcloud)
            loss = criterion(pred, nodes)
            loss.backward()
            optimizer.step()

            batch_size = pointcloud.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size
            correct += (pred.argmax(dim=1) == nodes).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"epoch: {epoch}, loss {epoch_loss:.4f}, train_acc {epoch_acc:.4f}")

    metrics = evaluate(model, test_loader, device=device, class_names=None)

    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1:      {metrics['macro_f1']:.4f}")
    print(f"Macro recall:  {metrics['macro_recall']:.4f}")
    print(metrics["classification_report"])

    plot_confusion_matrix(metrics["confusion_matrix"], normalize=False)
    plot_confusion_matrix(metrics["confusion_matrix"], normalize=True)

    good, bad = select_good_bad_samples(metrics["sample_records"], n_each=10)
    print("Best examples:")
    for r in good:
        print(r)
    print("\nWorst examples:")
    for r in bad:
        print(r)


def evaluate(model, loader, device, class_names=None):
    model.eval()

    all_logits = []
    all_preds = []
    all_targets = []
    all_paths = []

    with torch.no_grad():
        for pointcloud, targets, file_names in loader:
            pointcloud = pointcloud.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(pointcloud)                 # [B, C]
            probs = torch.softmax(logits, dim=1)       # [B, C]
            preds = logits.argmax(dim=1)               # [B]

            all_logits.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_paths.extend(file_names)

    y_prob = np.concatenate(all_logits, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    # confidence of predicted class
    pred_conf = y_prob[np.arange(len(y_pred)), y_pred]

    # confidence assigned to true class
    true_conf = y_prob[np.arange(len(y_true)), y_true]

    sample_records = []
    for i in range(len(y_true)):
        sample_records.append({
            "file_name": all_paths[i],
            "y_true": int(y_true[i]),
            "y_pred": int(y_pred[i]),
            "correct": bool(y_true[i] == y_pred[i]),
            "pred_conf": float(pred_conf[i]),
            "true_conf": float(true_conf[i]),
        })

    metrics = {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "per_class_support": support,
        "confusion_matrix": cm,
        "classification_report": report,
        "sample_records": sample_records,
    }

    return metrics


def plot_confusion_matrix(cm, class_names=None, normalize=False, save_path=None):
    if normalize:
        cm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(8, 8))
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
        title="Confusion Matrix" + (" (Normalized)" if normalize else "")
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

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def select_good_bad_samples(sample_records, n_each=10):
    bad = [r for r in sample_records if not r["correct"]]
    good = [r for r in sample_records if r["correct"]]

    # worst: confidently wrong
    bad = sorted(bad, key=lambda r: r["pred_conf"], reverse=True)[:n_each]

    # best: confidently correct
    good = sorted(good, key=lambda r: r["true_conf"], reverse=True)[:n_each]

    return good, bad

def plot_pointcloud(points, title="", save_path=None):
    # points expected as [N, 3]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2)
    ax.set_title(title)
    ax.set_axis_off()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    test_encoder()