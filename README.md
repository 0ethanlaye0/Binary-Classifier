# binary-classification

Binary classification models built in PyTorch, demonstrating how neural networks learn to separate non-linearly separable data.

## What's implemented

- **`data_exploration.py`** — Data setup and visualization for the circles dataset using `make_circles` from scikit-learn. Converts raw numpy data to PyTorch tensors, splits into train/test sets, and visualizes the two concentric circle classes using matplotlib.

- **`binary_classifier.py`** — A multi-layer perceptron (MLP) trained on the `make_moons` dataset using PyTorch. Uses ReLU activations for non-linearity, BCEWithLogitsLoss for binary cross-entropy, and Adam optimizer. Achieves ~95%+ test accuracy.

## Quick demo

```python
from torch import nn
import torch

model = nn.Sequential(
    nn.Linear(2, 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, 1)
)

loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1000):
    model.train()
    logits = model(X_train)
    preds = torch.round(torch.sigmoid(logits))
    loss = loss_fn(logits, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Evaluation loop
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test)
        test_preds = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
```

## Key concepts demonstrated

- Binary classification with `BCEWithLogitsLoss` (combines sigmoid + cross-entropy for numerical stability)
- Non-linearity via ReLU enabling separation of non-linearly separable data (make_moons)
- Adam optimizer — with a learning rate for parameters
- Converting logits to probabilities (`torch.sigmoid`) and class predictions (`torch.round`)
- Train/eval mode (`model.train()` / `model.eval()`) and `torch.inference_mode()` for evaluation
- Accuracy metric alongside loss for monitoring training

## Credit

Built following Daniel Bourke's [Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/) course.
