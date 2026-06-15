import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def train_one_seed(seed):
    set_seed(seed)
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = LSTMPredictor(input_dim=X_train.shape[2])
    criterion = nn.CrossEntropyLoss()  # 无类别权重
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    epochs = 150
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    pre = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    return acc, rec, pre, f1


# 运行30个种子
seeds = list(range(1, 31))
results = []
for s in seeds:
    acc, rec, pre, f1 = train_one_seed(s)
    results.append([s, acc, rec, pre, f1])
    print(f"Seed {s}: Acc={acc:.3f}, Rec={rec:.3f}, Pre={pre:.3f}, F1={f1:.3f}")

# 统计
results = np.array(results)
print("\n===== Summary over 30 runs (without class weights) =====")
print(
    f"Accuracy   : mean={results[:, 1].mean():.3f}, std={results[:, 1].std():.3f}, min={results[:, 1].min():.3f}, max={results[:, 1].max():.3f}")
print(
    f"Recall     : mean={results[:, 2].mean():.3f}, std={results[:, 2].std():.3f}, min={results[:, 2].min():.3f}, max={results[:, 2].max():.3f}")
print(
    f"Precision  : mean={results[:, 3].mean():.3f}, std={results[:, 3].std():.3f}, min={results[:, 3].min():.3f}, max={results[:, 3].max():.3f}")
print(
    f"Macro F1   : mean={results[:, 4].mean():.3f}, std={results[:, 4].std():.3f}, min={results[:, 4].min():.3f}, max={results[:, 4].max():.3f}")