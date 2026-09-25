# CNN MNIST — PyTorch

A from-scratch learning project for understanding and building a Convolutional Neural Network with PyTorch on the MNIST handwritten-digit dataset.

## Goal

Understand CNNs by building the complete pipeline:

```
Image → Convolution → ReLU → Pooling → Convolution → ReLU → Pooling → Flatten → Dense → Classification
```

This project is also used to inspect what happens inside the network during forward propagation, training, and evaluation.

## Files

| File | Description |
|---|---|
| `data.py` | Downloads MNIST via `torchvision`, wraps it in `DataLoader`s for training and testing |
| `model.py` | The `MNISTCNN` architecture |
| `train.py` | Training loop (5 epochs, Adam) + evaluation on the test set |
| `requirements.txt` | Dependencies |
| `.gitignore` | Excludes `data/`, `__pycache__/`, and virtual environments from version control |

## The Data

MNIST: 60,000 training images and 10,000 test images, each a 28×28 grayscale handwritten digit (0–9). `data.py` downloads it automatically on first run via `torchvision.datasets.MNIST`, converts each image to a float32 tensor scaled to `[0, 1]`, and serves both splits through `DataLoader`s with `batch_size=32` (shuffled for training, not shuffled for testing).

## The Model

```python
nn.Sequential(
    nn.Conv2d(in_channels=1,  out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Flatten(),
    nn.Linear(in_features=32*7*7, out_features=10)
)
```

**Verified shape trace** (confirmed by running a dummy batch through the model):

| Layer | Output Shape |
|---|---|
| Input | `(32, 1, 28, 28)` |
| Conv2d (1→16) | `(32, 16, 28, 28)` |
| ReLU | `(32, 16, 28, 28)` |
| MaxPool2d | `(32, 16, 14, 14)` |
| Conv2d (16→32) | `(32, 32, 14, 14)` |
| ReLU | `(32, 32, 14, 14)` |
| MaxPool2d | `(32, 32, 7, 7)` |
| Flatten | `(32, 1568)` |
| Linear (1568→10) | `(32, 10)` |

`padding=1` with a `3×3` kernel keeps spatial size unchanged after each convolution — all the size reduction (`28→14→7`) comes from the two `MaxPool2d` layers, each halving height and width. That's why `Linear` needs exactly `32*7*7 = 1568` input features — it's not a magic number, it's `(final channel count) × (final height) × (final width)`.

**Total trainable parameters: 20,490**

## Training Setup

| Setting | Value |
|---|---|
| Loss function | `CrossEntropyLoss` |
| Optimizer | `Adam`, `lr=0.001` |
| Epochs | 5 |
| Batch size | 32 |

The training loop follows the standard PyTorch pattern: forward pass → compute loss → `zero_grad()` → `backward()` → `optimizer.step()`, accumulating average loss per epoch. Evaluation runs afterward under `torch.no_grad()`, computing test loss and accuracy over the full test set.

## Requirements

```
numpy
torch
torchvision
matplotlib
```

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python train.py
```

This will download MNIST into `data/` on first run (may take a moment), train for 5 epochs, then print test loss and accuracy. Expect output in this shape:

```
Epoch 1/5 | Average Loss: ...
Epoch 2/5 | Average Loss: ...
...
Test Loss: ...
Test Accuracy: ...%
```

## What This Project Demonstrates

- The complete CNN pipeline: convolution (local feature extraction) → ReLU (non-linearity) → pooling (downsampling + spatial invariance), repeated, then flattened into a dense layer for final classification
- Why the `Linear` layer's input size is a direct, computable consequence of the architecture (channel count × final spatial dimensions), not a value to guess or hardcode blindly
- How `padding=1` on a `3×3` convolution preserves spatial dimensions, isolating all downsampling to the pooling layers
- The standard PyTorch training loop pattern, and the `model.train()` / `model.eval()` distinction between training and evaluation mode