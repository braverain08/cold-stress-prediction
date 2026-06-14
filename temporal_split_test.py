# temporal_split_test.py
# 按月份划分2025年测试数据，评估模型在不同季节的泛化性能

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# ========== 1. 加载原始数据，重新生成窗口日期 ==========
df = pd.read_csv('final_dataset_filled.csv')  # 由 fill_and_prepare.py 生成
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['year', 'date']).reset_index(drop=True)

feature_cols = ['temp_mean','temp_min','temp_max','hum_mean','light_mean','light_max',
                'st1','st2','st3','st4','sm1','sm2','sm3','sm4','ph','salt',
                'pest_count','lure_count','spore_count']
lookback = 7
future = 3

# 生成每个窗口的标签日期（窗口最后一天，即第7天）
window_end_dates = []
for i in range(lookback, len(df) - future + 1):
    last_date = df['date'].iloc[i-1]  # 窗口的最后一天
    window_end_dates.append(last_date)
window_end_dates = np.array(window_end_dates)

# 划分训练/测试 (按年份)
years = np.array([d.year for d in window_end_dates])
train_mask = years == 2024
test_mask = years == 2025

# 加载预处理后的 .npy 数据（与 fill_and_prepare.py 生成的完全一致）
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# 获取测试集对应的日期
test_dates = window_end_dates[test_mask]
test_months = pd.Series(test_dates).dt.month

# 按月份划分测试子集
mask_may_jun = (test_months >= 5) & (test_months <= 6)
mask_jul_aug = (test_months >= 7) & (test_months <= 8)
mask_sep_oct = (test_months >= 9) & (test_months <= 10)

X_test_mj = X_test[mask_may_jun]
y_test_mj = y_test[mask_may_jun]
X_test_ja = X_test[mask_jul_aug]
y_test_ja = y_test[mask_jul_aug]
X_test_so = X_test[mask_sep_oct]
y_test_so = y_test[mask_sep_oct]

print(f"May-Jun samples: {len(X_test_mj)}, cold ratio: {y_test_mj.mean():.3f}")
print(f"Jul-Aug samples: {len(X_test_ja)}, cold ratio: {y_test_ja.mean():.3f}")
print(f"Sep-Oct samples: {len(X_test_so)}, cold ratio: {y_test_so.mean():.3f}")

# ========== 2. 定义模型并加载预训练权重 ==========
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
model.load_state_dict(torch.load('lstm_model_complete.pth', map_location=torch.device('cpu')))
model.eval()

def evaluate_on_subset(X, y):
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    rec = recall_score(y, preds)
    pre = precision_score(y, preds)
    f1 = f1_score(y, preds, average='macro')
    return acc, rec, pre, f1

print("\n===== Performance on different temporal subsets =====")
acc_mj, rec_mj, pre_mj, f1_mj = evaluate_on_subset(X_test_mj, y_test_mj)
print(f"May-Jun     : Acc={acc_mj:.3f}, Rec={rec_mj:.3f}, Pre={pre_mj:.3f}, F1={f1_mj:.3f}")

acc_ja, rec_ja, pre_ja, f1_ja = evaluate_on_subset(X_test_ja, y_test_ja)
print(f"Jul-Aug     : Acc={acc_ja:.3f}, Rec={rec_ja:.3f}, Pre={pre_ja:.3f}, F1={f1_ja:.3f}")

acc_so, rec_so, pre_so, f1_so = evaluate_on_subset(X_test_so, y_test_so)
print(f"Sep-Oct     : Acc={acc_so:.3f}, Rec={rec_so:.3f}, Pre={pre_so:.3f}, F1={f1_so:.3f}")

# 保存结果
results = pd.DataFrame({
    'Subset': ['May-Jun', 'Jul-Aug', 'Sep-Oct'],
    'Samples': [len(X_test_mj), len(X_test_ja), len(X_test_so)],
    'Cold_ratio': [y_test_mj.mean(), y_test_ja.mean(), y_test_so.mean()],
    'Accuracy': [acc_mj, acc_ja, acc_so],
    'Recall': [rec_mj, rec_ja, rec_so],
    'Precision': [pre_mj, pre_ja, pre_so],
    'Macro_F1': [f1_mj, f1_ja, f1_so]
})
results.to_csv('temporal_performance.csv', index=False)
print("\n结果已保存至 temporal_performance.csv")