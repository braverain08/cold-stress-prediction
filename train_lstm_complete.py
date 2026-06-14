# train_lstm_complete.py
# 稳定版：固定随机种子 + 学习率0.0005 + 梯度裁剪 + 阈值实验

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import random
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 固定随机种子 ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# ========== 加载数据 ==========
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# ========== 类别权重 ==========
classes = np.unique(y_train)
cw = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = torch.tensor([cw[0], cw[1]], dtype=torch.float32)
print(f"类别权重: {class_weights.numpy()}")

# ========== 模型定义 ==========
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

input_dim = X_train.shape[2]
model = LSTMPredictor(input_dim)

# ========== 训练设置 ==========
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
epochs = 150

# ========== 训练循环 ==========
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
    if (epoch+1) % 30 == 0:
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# ========== 测试：默认阈值0.5 ==========
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)

acc = accuracy_score(y_test, preds)
rec = recall_score(y_test, preds)
pre = precision_score(y_test, preds)
f1 = f1_score(y_test, preds, average='macro')
print(f"\nTest Accuracy: {acc:.3f}")
print(f"Recall (Cold): {rec:.3f}")
print(f"Precision (Cold): {pre:.3f}")
print(f"Macro F1: {f1:.3f}")

# ========== 阈值实验（生成表3） ==========
thresholds = np.arange(0.1, 1.0, 0.1)
recalls = []
precisions = []
for th in thresholds:
    preds_th = (probs >= th).astype(int)
    recalls.append(recall_score(y_test, preds_th))
    precisions.append(precision_score(y_test, preds_th))

print("\nThreshold\tRecall\tPrecision")
for th, rec, prec in zip(thresholds, recalls, precisions):
    print(f"{th:.1f}\t\t{rec:.3f}\t{prec:.3f}")

# ========== 保存模型权重 ==========
torch.save(model.state_dict(), 'lstm_model_complete.pth')
print("\n模型权重已保存为 lstm_model_complete.pth")

# ========== 可选：生成论文所需图片 ==========
# 混淆矩阵
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Water','Cold'], yticklabels=['Water','Cold'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - LSTM')
plt.tight_layout()
plt.savefig('confusion_matrix_new.png', dpi=300)
plt.show()

# PR曲线
precision_curve, recall_curve, _ = precision_recall_curve(y_test, probs)
plt.figure(figsize=(5,4))
plt.plot(recall_curve, precision_curve, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.savefig('pr_curve_new.png', dpi=300)
plt.show()

# 阈值影响曲线
plt.figure(figsize=(6,4))
plt.plot(thresholds, recalls, 'o-', label='Recall')
plt.plot(thresholds, precisions, 's-', label='Precision')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Threshold Effect on Cold Stress Prediction')
plt.legend()
plt.grid(True)
plt.savefig('threshold_effect_new.png', dpi=300)
plt.show()

print("图片已保存: confusion_matrix_new.png, pr_curve_new.png, threshold_effect_new.png")