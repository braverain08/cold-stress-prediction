# run_window_ablation.py
# 窗口长度消融实验：对比 lookback = 3,5,7,10 天的预测性能
# 所有超参数、随机种子与主实验一致

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import random


# ========== 1. 固定随机种子 ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========== 2. 模型定义（与原论文相同） ==========
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# ========== 3. 训练和评估函数 ==========
def train_and_evaluate(lookback, future, df, feature_cols):
    print(f"\n--- 窗口长度 = {lookback} 天 ---")

    # 3.1 滑动窗口生成样本
    X, y = [], []
    for i in range(lookback, len(df) - future + 1):
        X_window = df[feature_cols].iloc[i - lookback:i].values
        y_label = df['cold_today'].iloc[i + future - 1]
        X.append(X_window)
        y.append(y_label)
    X = np.array(X)
    y = np.array(y)

    # 3.2 按年份划分训练/测试（窗口最后一天的年份）
    years = []
    for i in range(lookback, len(df) - future + 1):
        last_date = df['date'].iloc[i - 1]
        years.append(pd.to_datetime(last_date).year)
    years = np.array(years)
    train_mask = years == 2024
    test_mask = years == 2025
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
    print(f"冷害比例: 训练 {y_train.mean():.3f}, 测试 {y_test.mean():.3f}")

    # 3.3 标准化（仅用训练集）
    scaler = StandardScaler()
    n_train, seq_len, n_feat = X_train.shape
    X_train_flat = X_train.reshape(-1, n_feat)
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_train = X_train_flat.reshape(n_train, seq_len, n_feat)

    n_test = X_test.shape[0]
    X_test_flat = X_test.reshape(-1, n_feat)
    X_test_flat = scaler.transform(X_test_flat)
    X_test = X_test_flat.reshape(n_test, seq_len, n_feat)

    # 3.4 转换为Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # 3.5 类别权重
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = torch.tensor([cw[0], cw[1]], dtype=torch.float32)

    # 3.6 模型初始化
    set_seed(42)
    model = LSTMPredictor(input_dim=len(feature_cols))
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

    # 3.7 训练
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
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    # 3.8 评估
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    pre = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    print(f"结果: Acc={acc:.3f}, Rec={rec:.3f}, Pre={pre:.3f}, F1={f1:.3f}")
    return acc, rec, pre, f1, len(X_train), len(X_test)


# ========== 4. 主程序 ==========
if __name__ == "__main__":
    # 加载原始数据
    df = pd.read_csv('final_dataset.csv')
    df = df.sort_values(['year', 'date']).reset_index(drop=True)

    # 特征列
    feature_cols = ['temp_mean', 'temp_min', 'temp_max', 'hum_mean', 'light_mean', 'light_max',
                    'st1', 'st2', 'st3', 'st4', 'sm1', 'sm2', 'sm3', 'sm4', 'ph', 'salt',
                    'pest_count', 'lure_count', 'spore_count']

    # 填补缺失值
    df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(df[feature_cols].mean())


    # 生成冷害标签
    def cold_stress(row):
        return 1 if (row['temp_min'] < 0 or row['temp_mean'] < 5) else 0


    df['cold_today'] = df.apply(cold_stress, axis=1)

    # 测试的窗口长度
    lookbacks = [3, 5, 7, 10]
    future = 3
    results = []

    for lb in lookbacks:
        acc, rec, pre, f1, n_train, n_test = train_and_evaluate(lb, future, df, feature_cols)
        results.append({
            'lookback': lb,
            'accuracy': acc,
            'recall': rec,
            'precision': pre,
            'macro_f1': f1,
            'train_samples': n_train,
            'test_samples': n_test
        })

    # 打印结果表格
    print("\n===== 窗口长度消融实验结果 =====")
    print(f"{'Lookback':<8} {'Accuracy':<10} {'Recall (Cold)':<15} {'Precision':<12} {'Macro F1':<10} {'Train/Test'}")
    for r in results:
        print(
            f"{r['lookback']:<8} {r['accuracy']:<10.3f} {r['recall']:<15.3f} {r['precision']:<12.3f} {r['macro_f1']:<10.3f} {r['train_samples']}/{r['test_samples']}")

    # 保存结果到CSV
    df_res = pd.DataFrame(results)
    df_res.to_csv('window_ablation_results.csv', index=False)
    print("\n结果已保存到 window_ablation_results.csv")