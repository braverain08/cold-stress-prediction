# rule_baseline.py
# 传统农学规则基线：基于过去7天最低温累计下降和最后一天日均温

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def rule_predict(df, lookback=7, future=3):
    """
    df: 原始数据，需包含 'temp_min', 'temp_mean', 'date', 'year'
    返回 y_pred (list), y_true (list)
    """
    y_pred = []
    y_true = []
    for i in range(lookback, len(df) - future + 1):
        # 窗口数据
        window = df.iloc[i - lookback:i]
        # 实际标签：第 i+future-1 天是否为冷害（使用原数据中的 label 列）
        true_label = df['cold_today'].iloc[i + future - 1]
        y_true.append(true_label)

        # 规则判断
        temp_min_first = window['temp_min'].iloc[0]
        temp_min_last = window['temp_min'].iloc[-1]
        cumulative_drop = temp_min_first - temp_min_last  # 正数表示下降
        last_day_mean_temp = window['temp_mean'].iloc[-1]

        if cumulative_drop >= 3.0 and last_day_mean_temp < 8.0:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred, y_true


# 加载原始数据（已填充缺失值）
df = pd.read_csv('final_dataset.csv')
df = df.sort_values(['year', 'date']).reset_index(drop=True)

# 填充缺失值（与论文预处理一致）
feature_cols = ['temp_mean', 'temp_min', 'temp_max', 'hum_mean', 'light_mean', 'light_max',
                'st1', 'st2', 'st3', 'st4', 'sm1', 'sm2', 'sm3', 'sm4', 'ph', 'salt',
                'pest_count', 'lure_count', 'spore_count']
df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(df[feature_cols].mean())


# 生成冷害标签（与论文一致）
def cold_stress(row):
    return 1 if (row['temp_min'] < 0 or row['temp_mean'] < 5) else 0


df['cold_today'] = df.apply(cold_stress, axis=1)

# 使用规则预测
y_pred, y_true = rule_predict(df)

# 评估指标
acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
pre = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

print("===== Rule-based baseline results =====")
print(f"Accuracy  : {acc:.3f}")
print(f"Recall    : {rec:.3f}")
print(f"Precision : {pre:.3f}")
print(f"Macro F1  : {f1:.3f}")