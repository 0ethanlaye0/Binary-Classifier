import torch
import sklearn
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


X, y = make_blobs(n_samples=1200,
                  centers=5,
                  cluster_std=1.0,
                  random_state=99)
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.long)

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size = 0.2,
                                                    random_state = 42)

def plot():
    plt.scatter(x = X[:, 0], y = X[:, 1], c = y)
    plt.show()
#plot()

def accuracy(preds, target):
    summ = torch.eq(preds, target).sum()
    return 100 * summ / len(preds)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features = 2, out_features = 64),
            nn.ReLU(),
            nn.Linear(in_features = 64, out_features = 64),
            nn.ReLU(),
            nn.Linear(in_features = 64, out_features = 5),
        )

    def forward(self,x):
        return self.layer_stack(x)

model = Model()

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(params = model.parameters(),
                       lr = 0.001)

epochs = 1000

for epoch in range(epochs):
    model.train()
    logits = model(X_train)
    preds = torch.argmax(torch.softmax(logits, dim = 1), dim = 1)
    acc = accuracy(preds, y_train)
    loss = loss_fn(logits, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test)
        test_preds = torch.argmax(torch.softmax(test_logits, dim = 1), dim = 1)
        test_acc = accuracy(test_preds, y_test)
        test_loss = loss_fn(test_logits, y_test)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.4f} | Acc: {acc:.4f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}%")
