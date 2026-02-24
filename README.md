# MNIST Activation Function Comparison

Compare **Sigmoid**, **Tanh**, and **ReLU** activation functions using a 6-layer MLP on MNIST digit classification. The project trains three identical architectures (only the activation differs), benchmarks training and inference speed, and evaluates on your own digit images from the `assets/` folder.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
# or: python3 main.py
```

## What it does

1. **Trains** three MLPs (Sigmoid, Tanh, ReLU) for 15 epochs on MNIST
2. **Benchmarks** training time per epoch and total, plus inference speed (ms per 128 images)
3. **Generates** comparison charts and PNG summary tables
4. **Evaluates** each model on custom digit images in `assets/`

## Code overview

| Section | Purpose |
|--------|---------|
| **Model** | 6-layer MLP (784→256→256→128→128→64→32→10) with dropout. Same architecture for all three; only the non-linear activation changes. |
| **Data** | Loads MNIST from local `dataset/` (IDX files) or downloads via torchvision. Images flattened to 784-dim vectors. |
| **Training** | Adam optimizer, NLL loss from softmax. Best checkpoint (by test accuracy) saved; no early stopping. |
| **Custom digits** | Images from `assets/` are preprocessed (grayscale, invert if light bg, crop largest connected component, resize to 28×28, center). Predictions and confusion matrices saved per model. |

## Dataset

- Uses local `dataset/` if MNIST IDX files are present
- Otherwise downloads via torchvision

## Outputs

All saved to `outputs/`:

| Output | Description |
|--------|-------------|
| `mnist_sample_grid.png` | Sample MNIST images |
| `mnist_sigmoid/`, `mnist_tanh/`, `mnist_relu/` | Per-model loss & accuracy curves |
| `comparison/` | Combined charts: learning curves, timing, inference speed |
| `mnist_summary_table.png` | Best accuracy & training time (qualitative PNG table) |
| `custom_accuracy_table.png` | Accuracy on assets images (PNG table) |
| `custom_tests/{model}/assets/` | Per-image predictions, confusion matrix |

## Custom digit format

Place digit images in `assets/` with filenames: `{label}_{name}.png`

Example: `3_scan_001.png` → true label 3. The script preprocesses each image (crop, resize, center) to match MNIST style before running inference.
