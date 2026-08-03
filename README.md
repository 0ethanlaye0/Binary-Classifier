# pytorch-classification

Binary and multi-class classification models built in PyTorch, demonstrating how neural networks learn to separate non-linearly separable data.

## What's implemented

- **`data_exploration.py`** — Data setup and visualization for the circles dataset using `make_circles` from scikit-learn. Converts raw numpy data to PyTorch tensors, splits into train/test sets, and visualizes the two concentric circle classes using matplotlib.

- **`binary_classifier.py`** — MLP trained on `make_moons` using `BCEWithLogitsLoss` and Adam optimizer. Uses ReLU activations for non-linearity. Achieves ~95%+ test accuracy.

- **`multi_classifier.py`** — Multi-class MLP trained on `make_blobs` (5 classes) using `CrossEntropyLoss`. Uses `torch.argmax(torch.softmax(...))` to convert logits to class predictions. Achieves ~95%+ test accuracy.

## Key concepts demonstrated

- Binary classification — `BCEWithLogitsLoss`, `torch.sigmoid` + `torch.round` for predictions
- Multi-class classification — `CrossEntropyLoss`, `torch.softmax` + `torch.argmax` for predictions
- Why `CrossEntropyLoss` takes raw logits, not probabilities
- ReLU non-linearity enabling separation of non-linearly separable data
- Adam optimizer with adaptive learning rates
- Train/eval mode and `torch.inference_mode()` for evaluation
- Accuracy metric alongside loss

## Quick demo

```python
import torch
from torch import nn
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split

# --- Binary Classification ---
X, y = make_moons(n_samples=1000, noise=0.2)
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float).unsqueeze(1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model_bin = nn.Sequential(
    nn.Linear(2, 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, 1)
)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model_bin.parameters(), lr=0.001)

for epoch in range(1000):
    model_bin.train()
    logits = model_bin(X_train)
    loss = loss_fn(logits, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

model_bin.eval()
with torch.inference_mode():
    preds = torch.round(torch.sigmoid(model_bin(X_test)))

# --- Multi-class Classification ---
X, y = make_blobs(n_samples=1200, centers=5, cluster_std=1.8, random_state=99)
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.long)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model_multi = nn.Sequential(
    nn.Linear(2, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 5)
)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model_multi.parameters(), lr=0.001)

for epoch in range(3000):
    model_multi.train()
    logits = model_multi(X_train)
    loss = loss_fn(logits, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

model_multi.eval()
with torch.inference_mode():
    preds = torch.argmax(torch.softmax(model_multi(X_test), dim=1), dim=1)
```

## Credit

Built following Daniel Bourke's [Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/) course.
