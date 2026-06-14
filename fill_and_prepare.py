import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 读取原始数据
df = pd.read_csv('final_dataset.csv')
df = df.sort_values(['year', 'date']).reset_index(drop=True)

feature_cols = ['temp_mean', 'temp_min', 'temp_max', 'hum_mean', 'light_mean', 'light_max',
                'st1','st2','st3','st4','sm1','sm2','sm3','sm4','ph','salt',
                'pest_count','lure_count','spore_count']

# 填充缺失值：前向填充，再均值填充剩余的
df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(df[feature_cols].mean())

# 检查是否还有NaN
print("填充后缺失值总数:", df[feature_cols].isna().sum().sum())

# 保存填充后的数据
df.to_csv('final_dataset_filled.csv', index=False)
print("已保存 final_dataset_filled.csv")

# 定义冷胁迫标签
def cold_stress(row):
    return 1 if (row['temp_min'] < 0 or row['temp_mean'] < 5) else 0
df['cold_today'] = df.apply(cold_stress, axis=1)

# 滑动窗口参数
lookback = 7
future = 3
X, y = [], []
for i in range(lookback, len(df) - future + 1):
    X_window = df[feature_cols].iloc[i-lookback:i].values
    y_label = df['cold_today'].iloc[i+future-1]
    X.append(X_window)
    y.append(y_label)

X = np.array(X)
y = np.array(y)

# 划分年份
years = []
for i in range(lookback, len(df) - future + 1):
    last_date = df['date'].iloc[i-1]
    years.append(pd.to_datetime(last_date).year)
years = np.array(years)

train_mask = years == 2024
test_mask = years == 2025
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print("训练集样本数:", len(X_train))
print("测试集样本数:", len(X_test))
print("训练集正例比例: {:.3f}".format(y_train.mean()))
print("测试集正例比例: {:.3f}".format(y_test.mean()))

# 标准化
scaler = StandardScaler()
n_train, seq_len, n_feat = X_train.shape
X_train_2d = X_train.reshape(-1, n_feat)
X_train_2d = scaler.fit_transform(X_train_2d)
X_train = X_train_2d.reshape(n_train, seq_len, n_feat)

n_test = X_test.shape[0]
X_test_2d = X_test.reshape(-1, n_feat)
X_test_2d = scaler.transform(X_test_2d)
X_test = X_test_2d.reshape(n_test, seq_len, n_feat)

# 保存处理好的数据
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
print("已保存 X_train.npy, y_train.npy, X_test.npy, y_test.npy")