import torch
import sklearn
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


X, y = make_moons(n_samples=1000, noise=0.2)
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float).unsqueeze(dim = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 42)
def plot():
    plt.scatter(x = X[:, 0], y = X[:, 1], c = y.squeeze())
    plt.show()
#plot()

def accuracy(pred, target):
    summ = torch.eq(pred, target).sum()
    return 100 * summ/len(target)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features = 2, out_features = 32),
            nn.ReLU(),
            nn.Linear(in_features = 32, out_features = 32),
            nn.ReLU(),
            nn.Linear(in_features = 32, out_features = 32),
            nn.ReLU(),
            nn.Linear(in_features = 32, out_features = 1),
        )

    def forward(self, x):
        return self.layer_stack(x)

model = Model()

loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(params = model.parameters(),
                       lr = 0.001)
epochs = 1000

for epoch in range(epochs):
    model.train()
    logits = model(X_train)
    preds = torch.round(torch.sigmoid(logits))
    acc = accuracy(preds, y_train)
    loss = loss_fn(logits, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test)
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy(test_pred, y_test)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.4f} | Acc: {acc:.4f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}%")