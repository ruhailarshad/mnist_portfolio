
import glob
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps, ImageFilter
from scipy import ndimage
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# ============== CONFIG ==============
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ASSETS_DIR = PROJECT_ROOT / "assets"
CUSTOM_TEST_FOLDERS = {"assets": ASSETS_DIR}

LAYER_SIZES = [(784, 256), (256, 256), (256, 128), (128, 128), (128, 64), (64, 32)]
DROPOUT_RATE = 0.2
SEED = 42
EPOCHS = 15
BATCH_SIZE = 128
LR = 1e-3

ACTIVATIONS = ["sigmoid", "tanh", "relu"]

COLORS = {"sigmoid": "#E63946", "tanh": "#457B9D", "relu": "#2A9D8F"}
DEFAULT_COLOR = "#6C757D"


# ============== MODEL ==============
def get_activation(name: str) -> nn.Module:
    name = name.lower().strip()
    activations = {"sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "relu": nn.ReLU()}
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]


class MLP(nn.Module):
    def __init__(self, activation: str = "sigmoid", activations: Optional[List[str]] = None, dropout: float = DROPOUT_RATE):
        super().__init__()
        sizes = LAYER_SIZES
        acts = activations or [activation] * len(sizes)
        if len(acts) != len(sizes):
            raise ValueError("activations length must match layers")
        blocks = []
        for (in_f, out_f), act_name in zip(sizes, acts):
            blocks.extend([nn.Linear(in_f, out_f), get_activation(act_name), nn.Dropout(dropout)])
        blocks.extend([nn.Linear(sizes[-1][1], 10), nn.Softmax(dim=1)])
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============== DATA ==============
def _load_idx_images(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype=">i4")
        assert magic == 2051
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols).astype(np.float32) / 255.0


def _load_idx_labels(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, num = np.frombuffer(f.read(8), dtype=">i4")
        assert magic == 2049
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist_local(root: Path) -> Optional[tuple]:
    variants = [
        ("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
        ("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte"),
    ]
    for tr_img, tr_lbl, te_img, te_lbl in variants:
        if all((root / f).exists() for f in [tr_img, tr_lbl, te_img, te_lbl]):
            return (_load_idx_images(root / tr_img), _load_idx_labels(root / tr_lbl),
                    _load_idx_images(root / te_img), _load_idx_labels(root / te_lbl))
    return None


def get_dataloaders(batch_size: int = BATCH_SIZE):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local = load_mnist_local(DATASET_DIR)
    if local:
        X_tr, y_tr, X_te, y_te = local
        train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
        test_ds = TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long))
    else:
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
        train_ds = datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=tfm)
        test_ds = datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=tfm)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0))


# ============== TRAINING ==============
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def nll_from_softmax(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return nn.NLLLoss()(torch.log(probs.clamp(min=eps)), targets)


@dataclass
class TrainResult:
    model_name: str
    model: nn.Module
    history: Dict[str, List[float]]
    best_test_acc: float
    total_time_sec: float
    epoch_times_sec: List[float]
    final_state_dict: dict


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = nll_from_softmax(out, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def train_model(activation: str, activations=None, train_loader=None, test_loader=None,
                epochs: int = EPOCHS, lr: float = LR, dropout: float = 0.2) -> TrainResult:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(activation=activation, activations=activations, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    epoch_times, best_acc, best_state = [], -1.0, None
    model_name = "mixed" if activations else activation
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        t_ep = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, device, True)
        te_loss, te_acc = run_epoch(model, test_loader, optimizer, device, False)
        epoch_times.append(time.perf_counter() - t_ep)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        print(f"[{model_name.upper()}] Epoch {ep:02d}/{epochs} | train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"test loss={te_loss:.4f} acc={te_acc:.4f} | time={epoch_times[-1]:.2f}s")
        if te_acc > best_acc:
            best_acc, best_state = te_acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return TrainResult(model_name, model, history, best_acc, time.perf_counter() - t0, epoch_times, best_state)


# ============== VISUALIZATION ==============
def _color(name: str) -> str:
    return COLORS.get(name.lower().split("_")[0] if "_" in name else name.lower(), DEFAULT_COLOR)


def plot_learning_curves(results: Dict[str, dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name, data in results.items():
        h, e = data["history"], range(1, len(data["history"]["train_loss"]) + 1)
        c = _color(name)
        ax1.plot(e, h["train_loss"], "-", color=c, alpha=0.8)
        ax1.plot(e, h["test_loss"], "--", color=c, alpha=0.9, linewidth=1.5)
        ax2.plot(e, h["train_acc"], "-", color=c, alpha=0.8)
        ax2.plot(e, h["test_acc"], "--", color=c, alpha=0.9, linewidth=1.5)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss Curves")
    ax1.legend(results.keys(), fontsize=9); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy Curves")
    ax2.legend(results.keys(), fontsize=9); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 1.02)
    plt.tight_layout(); plt.savefig(out_dir / "comparison_learning_curves.png", dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved: {out_dir / 'comparison_learning_curves.png'}")


def plot_comparison_summary(results: Dict[str, dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bars1 = ax1.bar(names, [results[n]["best_acc"] * 100 for n in names], color=[_color(n) for n in names], edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Best Test Accuracy (%)"); ax1.set_title("Model Accuracy"); ax1.set_ylim(0, 105)
    for b in bars1: ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}%", ha="center", va="bottom")
    bars2 = ax2.bar(names, [results[n]["total_time_sec"] for n in names], color=[_color(n) for n in names], edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Training Time (sec)"); ax2.set_title("Training Speed")
    for b in bars2: ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}s", ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(out_dir / "comparison_summary.png", dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved: {out_dir / 'comparison_summary.png'}")


def plot_timing_per_epoch(results: Dict[str, dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in results.items():
        if data.get("epoch_times_sec"):
            ax.plot(range(1, len(data["epoch_times_sec"]) + 1), data["epoch_times_sec"], "o-", color=_color(name), label=name, markersize=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Epoch Duration (sec)"); ax.set_title("Per-Epoch Training Time"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "timing_per_epoch.png", dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved: {out_dir / 'timing_per_epoch.png'}")


def plot_inference_benchmark(inference_times_ms: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(inference_times_ms.keys())
    bars = plt.bar(names, list(inference_times_ms.values()), color=[_color(n) for n in names], edgecolor="black", linewidth=0.5)
    plt.ylabel("Inference Time (ms/128 imgs)"); plt.title("Inference Speed")
    for b in bars: plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}", ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(out_dir / "inference_benchmark.png", dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved: {out_dir / 'inference_benchmark.png'}")


def plot_speed_summary(results: Dict[str, dict], inference_times_ms: dict,
                      fastest_train: str, fastest_inference: str, overall_fastest: str, out_dir: Path):
    """Combined speed chart: training time + inference time + winners."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    train_times = [results[n]["total_time_sec"] for n in names]
    infer_times = [inference_times_ms[n] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = [_color(n) for n in names]
    edge_fast = "gold"
    lw_fast = 2.5

    # Training time
    bars1 = ax1.bar(names, train_times, color=colors, edgecolor="black", linewidth=0.5)
    for i, (bar, n) in enumerate(zip(bars1, names)):
        if n == fastest_train:
            bar.set_edgecolor(edge_fast)
            bar.set_linewidth(lw_fast)
    ax1.set_ylabel("Time (seconds)"); ax1.set_title("Training Time")
    ax1.set_ylim(0, max(train_times) * 1.15)
    for bar, n in zip(bars1, names):
        lbl = f"{bar.get_height():.1f}s\n★ Fastest" if n == fastest_train else f"{bar.get_height():.1f}s"
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, lbl, ha="center", va="bottom",
                 fontsize=9, fontweight="bold" if n == fastest_train else "normal")
    ax1.grid(axis="y", alpha=0.3)

    # Inference time
    bars2 = ax2.bar(names, infer_times, color=colors, edgecolor="black", linewidth=0.5)
    for bar, n in zip(bars2, names):
        if n == fastest_inference:
            bar.set_edgecolor(edge_fast)
            bar.set_linewidth(lw_fast)
    ax2.set_ylabel("Time (ms per 128 imgs)"); ax2.set_title("Inference Time")
    ax2.set_ylim(0, max(infer_times) * 1.2)
    for bar, n in zip(bars2, names):
        lbl = f"{bar.get_height():.2f} ms\n★ Fastest" if n == fastest_inference else f"{bar.get_height():.2f} ms"
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, lbl, ha="center", va="bottom",
                 fontsize=9, fontweight="bold" if n == fastest_inference else "normal")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Speed Summary — Overall Fastest: {overall_fastest.upper()}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "speed_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir / 'speed_summary.png'}")


def plot_mnist_samples(loader, out_path: Path, n: int = 9):
    np.random.seed(42)
    images, labels = [], []
    for x, y in loader:
        for i in range(x.size(0)):
            images.append(x[i].view(28, 28).numpy()); labels.append(y[i].item())
            if len(images) >= n: break
        if len(images) >= n: break
    idxs = np.random.choice(len(images), min(n, len(images)), replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for ax, idx in zip(axes.flatten(), idxs):
        ax.imshow(images[idx], cmap="gray"); ax.set_title(f"Label: {labels[idx]}"); ax.axis("off")
    plt.suptitle("MNIST Sample Images"); plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
    print(f"Saved: {out_path}")


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, title: str = "Confusion Matrix"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(10), yticks=np.arange(10), xticklabels=range(10), yticklabels=range(10),
           xlabel="Predicted", ylabel="True", title=title)
    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=9)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
    print(f"Saved: {out_path}")


def plot_mnist_summary_table(results: Dict[str, TrainResult], out_dir: Path):
    """Renders MNIST training results as a qualitative, explanatory PNG table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    best_acc = max(r.best_test_acc for r in results.values())
    fastest_time = min(r.total_time_sec for r in results.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    headers = ["Activation", "Best Test Accuracy", "Training Time", "Notes"]
    rows = []
    for name, r in results.items():
        acc_pct = r.best_test_acc * 100
        time_str = f"{r.total_time_sec:.2f} s"
        notes = []
        if r.best_test_acc >= best_acc:
            notes.append("Highest accuracy")
        if r.total_time_sec <= fastest_time:
            notes.append("Fastest training")
        notes_str = "; ".join(notes) if notes else "—"
        rows.append([name.capitalize(), f"{acc_pct:.2f}%", time_str, notes_str])

    col_widths = [0.18, 0.28, 0.22, 0.32]
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.8)

    for i, header in enumerate(headers):
        cell = table[(0, i)]
        cell.set_text_props(fontsize=12, fontweight="bold", ha="center")
        cell.set_facecolor("#2A9D8F")
        cell.set_text_props(color="white")

    for r_idx, row in enumerate(rows):
        for c_idx, cell_text in enumerate(row):
            cell = table[(r_idx + 1, c_idx)]
            cell.set_text_props(ha="center")
            if c_idx == 0:
                cell.set_facecolor(_color(row[0].lower()))
                cell.set_text_props(color="white", fontweight="bold")
            elif c_idx == 1:
                acc_val = results[row[0].lower()].best_test_acc
                cell.set_facecolor("#E8F5E9" if acc_val >= best_acc else "#FAFAFA")
            elif c_idx == 2:
                time_val = results[row[0].lower()].total_time_sec
                cell.set_facecolor("#E3F2FD" if time_val <= fastest_time else "#FAFAFA")
            else:
                cell.set_facecolor("#FAFAFA")

    ax.set_title(
        "MNIST Test Set Performance — Activation Function Comparison\n"
        "Best validation accuracy and total training time (15 epochs, 128 batch size)",
        fontsize=13, fontweight="bold", pad=20
    )
    fig.text(0.5, 0.02,
             "Green highlight = highest accuracy | Blue highlight = fastest training\n"
             "Metrics from held-out MNIST test set (~10k images)",
             ha="center", fontsize=9, color="#555", wrap=True)
    plt.tight_layout(rect=[0, 0.04, 1, 0.92])
    plt.savefig(out_dir / "mnist_summary_table.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir / 'mnist_summary_table.png'}")


def _friendly_folder_name(key: str) -> str:
    """Human-readable labels for custom test folder names."""
    names = {
        "online": "Online samples",
        "word_blackbg": "Word (black bg)",
        "word_whitebg": "Word (white bg)",
        "paint_whitebg": "Paint (white bg)",
        "handwritten": "Handwritten",
    }
    return names.get(key, key.replace("_", " ").title())


def plot_custom_accuracy_table(accuracy_results: Dict[str, Dict[str, float]], out_dir: Path):
    """Renders custom digit accuracy per model as a qualitative, explanatory PNG table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(accuracy_results.keys())
    folders = sorted({f for m in accuracy_results.values() for f in m.keys()})
    if not folders:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    headers = ["Model"] + [_friendly_folder_name(f) for f in folders] + ["Mean"]
    rows = []
    for model in models:
        row = [model.capitalize()]
        vals = []
        for f in folders:
            v = accuracy_results[model].get(f)
            if v is not None and isinstance(v, float):
                row.append(f"{v * 100:.1f}%")
                vals.append(v)
            else:
                row.append("—")
        mean_val = np.mean(vals) if vals else 0
        row.append(f"{mean_val * 100:.1f}%" if vals else "—")
        rows.append(row)

    n_cols = len(headers)
    col_widths = [0.12] + [0.14] * (n_cols - 2) + [0.12]
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.4)

    for i, header in enumerate(headers):
        cell = table[(0, i)]
        cell.set_text_props(fontsize=11, fontweight="bold", ha="center")
        cell.set_facecolor("#457B9D")
        cell.set_text_props(color="white")

    for r_idx, row in enumerate(rows):
        for c_idx, cell_text in enumerate(row):
            cell = table[(r_idx + 1, c_idx)]
            cell.set_text_props(ha="center")
            if c_idx == 0:
                cell.set_facecolor(_color(row[0].lower()))
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#FAFAFA")

    ax.set_title(
        "Custom Digit Recognition — Accuracy by Source and Model\n"
        "Fraction of correctly classified digits from each custom image folder",
        fontsize=13, fontweight="bold", pad=20
    )
    fig.text(0.5, 0.02,
             "Each folder contains digit images named {label}_{name}.png (e.g. 3_my_digit.png). "
             "Models trained on MNIST; custom images may differ in style and preprocessing.",
             ha="center", fontsize=9, color="#555", wrap=True)
    plt.tight_layout(rect=[0, 0.04, 1, 0.9])
    plt.savefig(out_dir / "custom_accuracy_table.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir / 'custom_accuracy_table.png'}")


# ============== CUSTOM IMAGE EVALUATION ==============
def preprocess_digit(path: str):
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img)
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.mean() > 0.5:
        img = ImageOps.invert(img)
        arr = np.array(img).astype(np.float32) / 255.0
    img = img.filter(ImageFilter.MaxFilter(3))
    arr = np.array(img).astype(np.float32) / 255.0
    mask = arr > 0.15
    labeled, n = ndimage.label(mask)
    if n == 0:
        return np.zeros((1, 784), dtype=np.float32), np.zeros((28, 28), dtype=np.float32), img
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest = 1 + np.argmax(sizes)
    mask = labeled == largest
    arr = arr * mask
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = arr[y0:y1, x0:x1]
    h, w = cropped.shape
    scale = 20.0 / max(h, w, 1)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    cropped_img = Image.fromarray((cropped * 255).astype(np.uint8)).resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(cropped_img, ((28 - new_w) // 2, (28 - new_h) // 2))
    arr28 = np.array(canvas).astype(np.float32) / 255.0
    cy, cx = ndimage.center_of_mass(arr28)
    arr28 = ndimage.shift(arr28, shift=(int(round(14 - cy)), int(round(14 - cx))), order=1, mode="constant", cval=0.0)
    return arr28.reshape(1, -1).astype(np.float32), arr28, img


def predict_and_save(model, path: str, out_dir: Path, device) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    flat, img28, img_orig = preprocess_digit(path)
    x = torch.tensor(flat, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        probs = model(x)[0].detach().cpu().numpy()
    pred = int(np.argmax(probs))
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(np.array(img_orig), cmap="gray"); axes[0].set_title("Loaded"); axes[0].axis("off")
    axes[1].imshow(img28, cmap="gray"); axes[1].set_title("Preprocessed 28×28"); axes[1].axis("off")
    axes[2].bar(range(10), probs, color="#457B9D"); axes[2].set_xticks(range(10)); axes[2].set_title(f"Pred: {pred}")
    plt.tight_layout(); plt.savefig(out_dir / f"{Path(path).stem}_prediction.png", dpi=200); plt.close()
    return pred


def evaluate_folder(model, folder_path: str, out_dir: Path, device) -> float:
    paths = [p for p in sorted(glob.glob(os.path.join(folder_path, "*.*"))) if not os.path.basename(p).startswith(".")]
    if not paths:
        print(f"WARNING: No images in {folder_path}")
        return 0.0
    y_true, y_pred = [], []
    for p in paths:
        try:
            true_label = int(os.path.basename(p).split("_")[0])
        except ValueError:
            continue
        y_true.append(true_label)
        y_pred.append(predict_and_save(model, p, out_dir, device))
    if not y_true:
        return 0.0
    acc = (np.array(y_true) == np.array(y_pred)).mean()
    cm = confusion_matrix(y_true, y_pred)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, digits=4, zero_division=0))
    plot_confusion_matrix(cm, out_dir / "confusion_matrix.png", title=Path(folder_path).name)
    return acc


# ============== MAIN ==============
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_dataloaders()
    data_src = "local" if (DATASET_DIR / "train-images-idx3-ubyte").exists() else "download"
    print(f"\nData source: {data_src}\n")

    plot_mnist_samples(train_loader, OUTPUT_DIR / "mnist_sample_grid.png")

    results, plot_data = {}, {}
    print("\n" + "=" * 60 + "\nTRAINING: Sigmoid, Tanh, ReLU\n" + "=" * 60)
    for act in ACTIVATIONS:
        res = train_model(act, train_loader=train_loader, test_loader=test_loader)
        results[act], plot_data[act] = res, {"history": res.history, "best_acc": res.best_test_acc,
                                             "total_time_sec": res.total_time_sec, "epoch_times_sec": res.epoch_times_sec}
        plot_learning_curves({act: plot_data[act]}, OUTPUT_DIR / f"mnist_{act}")

    print("\n" + "=" * 60 + "\nGENERATING CHARTS\n" + "=" * 60)
    cmp_dir = OUTPUT_DIR / "comparison"
    plot_learning_curves(plot_data, cmp_dir)
    plot_comparison_summary(plot_data, cmp_dir)
    plot_timing_per_epoch(plot_data, cmp_dir)

    plot_mnist_summary_table(results, OUTPUT_DIR)

    print("\n" + "=" * 60 + "\nMNIST RESULTS\n" + "=" * 60)
    for name, r in sorted(results.items(), key=lambda x: -x[1].best_test_acc):
        print(f"  {name:20s}  {r.best_test_acc:.4f}  ({r.total_time_sec:.1f}s)")
    fastest_train = min(results.items(), key=lambda x: x[1].total_time_sec)
    print(f"\n  Fastest training: {fastest_train[0]} ({fastest_train[1].total_time_sec:.1f}s)")

    print("\n" + "=" * 60 + "\nINFERENCE BENCHMARK\n" + "=" * 60)
    device = next(results["sigmoid"].model.parameters()).device
    x_sample = next(iter(test_loader))[0][:128].to(device)
    inference_times = {}
    for name, res in results.items():
        model = res.model.eval()
        with torch.no_grad():
            for _ in range(10): _ = model(x_sample)
            t0 = time.perf_counter()
            for _ in range(50): _ = model(x_sample)
            inference_times[name] = round((time.perf_counter() - t0) / 50 * 1000, 2)
        print(f"  {name:20s}  {inference_times[name]:.2f} ms/batch")
    fastest_inference = min(inference_times.items(), key=lambda x: x[1])
    print(f"\n  Fastest inference: {fastest_inference[0]} ({fastest_inference[1]:.2f} ms/batch)")
    plot_inference_benchmark(inference_times, cmp_dir)

    # Overall fastest (combined rank: 1st=3pts, 2nd=2pts, 3rd=1pt; highest wins)
    train_rank = {n: i + 1 for i, (n, _) in enumerate(sorted(results.items(), key=lambda x: x[1].total_time_sec))}
    infer_rank = {n: i + 1 for i, (n, _) in enumerate(sorted(inference_times.items(), key=lambda x: x[1]))}
    combined = {n: (4 - train_rank[n]) + (4 - infer_rank[n]) for n in results}
    overall_fastest = max(combined.items(), key=lambda x: x[1])[0]
    print("\n" + "=" * 60 + "\nSPEED SUMMARY\n" + "=" * 60)
    print(f"  Fastest training:   {fastest_train[0]} ({fastest_train[1].total_time_sec:.1f}s)")
    print(f"  Fastest inference:  {fastest_inference[0]} ({fastest_inference[1]:.2f} ms/batch)")
    print(f"  Overall fastest:   {overall_fastest} (best combined rank)")
    plot_speed_summary(plot_data, inference_times, fastest_train[0], fastest_inference[0], overall_fastest, cmp_dir)

    print("\n" + "=" * 60 + "\nCUSTOM DIGIT EVALUATION\n" + "=" * 60)
    accuracy_results = {}
    models_to_test = {n: r.model for n, r in results.items()}
    for model_name, model in models_to_test.items():
        accuracy_results[model_name] = {}
        print(f"\n--- {model_name.upper()} ---")
        for folder_name, folder_path in CUSTOM_TEST_FOLDERS.items():
            if not folder_path.exists():
                print(f"  {folder_name}: (skipped, not found)")
                continue
            acc = evaluate_folder(model, str(folder_path), OUTPUT_DIR / "custom_tests" / model_name / folder_name, device)
            accuracy_results[model_name][folder_name] = acc
            print(f"  {folder_name}: {acc:.4f}")

    plot_custom_accuracy_table(accuracy_results, OUTPUT_DIR)
    print("\n" + "=" * 60 + f"\nDONE. Outputs: {OUTPUT_DIR}\n" + "=" * 60)


if __name__ == "__main__":
    main()
