# threshold_stability.py
# 30次运行，每次评估多个阈值(0.1~0.9)下的召回率与精确率，输出均值和标准差表

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import recall_score, precision_score
from sklearn.utils.class_weight import compute_class_weight
import random
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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

def train_once(seed, X_train, y_train, X_test, y_test):
    set_seed(seed)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = torch.tensor([cw[0], cw[1]], dtype=torch.float32)

    model = LSTMPredictor(input_dim=X_train.shape[2])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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

    thresholds = np.arange(0.1, 1.0, 0.1)
    recs = []
    pres = []
    for th in thresholds:
        preds = (probs >= th).astype(int)
        recs.append(recall_score(y_test, preds))
        pres.append(precision_score(y_test, preds))
    return recs, pres

# 加载数据
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

seeds = list(range(1, 31))
all_recs = []
all_pres = []
for seed in seeds:
    recs, pres = train_once(seed, X_train, y_train, X_test, y_test)
    all_recs.append(recs)
    all_pres.append(pres)
    print(f"Finished seed {seed}")

all_recs = np.array(all_recs)  # shape: (30, 9)
all_pres = np.array(all_pres)

thresholds = np.arange(0.1, 1.0, 0.1)
results = []
for i, th in enumerate(thresholds):
    rec_mean = np.mean(all_recs[:, i])
    rec_std = np.std(all_recs[:, i])
    pre_mean = np.mean(all_pres[:, i])
    pre_std = np.std(all_pres[:, i])
    results.append([th, rec_mean, rec_std, pre_mean, pre_std])

df = pd.DataFrame(results, columns=['Threshold', 'Recall_mean', 'Recall_std', 'Precision_mean', 'Precision_std'])
print("\n===== Threshold stability over 30 runs =====")
print(df.to_string(index=False, float_format='%.3f'))

# 保存为CSV
df.to_csv('threshold_stability.csv', index=False)
print("\n结果已保存到 threshold_stability.csv")