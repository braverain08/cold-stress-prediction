# stability_analysis.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import random
import pandas as pd


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_once(seed, X_train, y_train, X_test, y_test):
    set_seed(seed)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = torch.tensor([cw[0], cw[1]], dtype=torch.float32)

    class LSTMPredictor(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out)

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
        preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    pre = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    return acc, rec, pre, f1


# 加载数据
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

seeds = list(range(1, 31))  # 30次运行
results = []
for seed in seeds:
    acc, rec, pre, f1 = train_once(seed, X_train, y_train, X_test, y_test)
    results.append([seed, acc, rec, pre, f1])
    print(f"Seed {seed}: Acc={acc:.3f}, Rec={rec:.3f}, Pre={pre:.3f}, F1={f1:.3f}")

df = pd.DataFrame(results, columns=['seed', 'accuracy', 'recall', 'precision', 'macro_f1'])
print("\n===== Summary over 30 runs =====")
print(
    f"Accuracy  : mean={df['accuracy'].mean():.3f}, std={df['accuracy'].std():.3f}, min={df['accuracy'].min():.3f}, max={df['accuracy'].max():.3f}")
print(
    f"Recall    : mean={df['recall'].mean():.3f}, std={df['recall'].std():.3f}, min={df['recall'].min():.3f}, max={df['recall'].max():.3f}")
print(
    f"Precision : mean={df['precision'].mean():.3f}, std={df['precision'].std():.3f}, min={df['precision'].min():.3f}, max={df['precision'].max():.3f}")
print(
    f"Macro F1  : mean={df['macro_f1'].mean():.3f}, std={df['macro_f1'].std():.3f}, min={df['macro_f1'].min():.3f}, max={df['macro_f1'].max():.3f}")
import matplotlib.pyplot as plt

# 假设 df 是包含30次结果的DataFrame
metrics = ['accuracy', 'precision', 'macro_f1']
titles = ['Accuracy', 'Precision (cold)', 'Macro F1']
colors = ['skyblue', 'salmon', 'lightgreen']

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
    axes[i].boxplot(df[metric], patch_artist=True, boxprops=dict(facecolor=color))
    axes[i].set_title(title)
    axes[i].set_ylabel('Score')
    axes[i].set_ylim(0, 1)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('stability_boxplot.png', dpi=300)
plt.show()

# 专门画一个召回率的图（因为所有值都是1.0）
plt.figure(figsize=(4, 4))
plt.boxplot(df['recall'], patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title('Recall (cold)')
plt.ylabel('Score')
plt.ylim(0.95, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('stability_recall.png', dpi=300)
plt.show()
print("箱线图已保存: stability_boxplot.png, stability_recall.png")