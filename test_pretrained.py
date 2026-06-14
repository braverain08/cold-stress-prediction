# test_pretrained.py
# 加载预训练模型权重，直接复现论文中的数值和图表

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix, precision_recall_curve)

# ========== 模型定义（必须与训练时一致） ==========
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# ========== 加载测试数据 ==========
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
X_test_t = torch.tensor(X_test, dtype=torch.float32)

# ========== 加载预训练权重 ==========
model = LSTMPredictor(input_dim=X_test.shape[2])
model.load_state_dict(torch.load('lstm_model_complete.pth', map_location=torch.device('cpu')))
model.eval()

# ========== 预测 ==========
with torch.no_grad():
    logits = model(X_test_t)
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)

# ========== 打印指标（应与论文表2一致） ==========
acc = accuracy_score(y_test, preds)
rec = recall_score(y_test, preds)
pre = precision_score(y_test, preds)
f1 = f1_score(y_test, preds, average='macro')
print(f"Test Accuracy : {acc:.3f}")   # 应约为 0.807
print(f"Recall (Cold) : {rec:.3f}")   # 应为 1.000
print(f"Precision (Cold): {pre:.3f}") # 应约为 0.352
print(f"Macro F1      : {f1:.3f}")   # 应约为 0.700

# ========== 混淆矩阵（图1） ==========
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Water','Cold'], yticklabels=['Water','Cold'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - LSTM')
plt.tight_layout()
plt.savefig('confusion_matrix_pretrained.png', dpi=300)
plt.show()

# ========== PR曲线（图2） ==========
prec_curve, rec_curve, _ = precision_recall_curve(y_test, probs)
plt.figure(figsize=(5,4))
plt.plot(rec_curve, prec_curve, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.savefig('pr_curve_pretrained.png', dpi=300)
plt.show()

# ========== 阈值实验（表3） ==========
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

# 阈值影响曲线（图3）
plt.figure(figsize=(6,4))
plt.plot(thresholds, recalls, 'o-', label='Recall')
plt.plot(thresholds, precisions, 's-', label='Precision')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Threshold Effect on Cold Stress Prediction')
plt.legend()
plt.grid(True)
plt.savefig('threshold_effect_pretrained.png', dpi=300)
plt.show()

print("\n图片已保存为: confusion_matrix_pretrained.png, pr_curve_pretrained.png, threshold_effect_pretrained.png")