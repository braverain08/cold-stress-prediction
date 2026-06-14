import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# 1. 加载之前保存的数据
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

print("数据加载成功")
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 转换为 Tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# 类别权重（使用训练集计算）
classes = np.unique(y_train)
cw = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = torch.tensor([cw[0], cw[1]], dtype=torch.float32)


# 2. 定义 LSTM 模型（与之前相同）
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# 3. 训练函数（复用之前的超参数）
def train_eval(X_tr, y_tr, X_te, y_te, input_dim, epochs=150):
    model = LSTMPredictor(input_dim)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_te)
        preds = logits.argmax(dim=1).numpy()
    acc = accuracy_score(y_te.numpy(), preds)
    rec = recall_score(y_te.numpy(), preds)
    f1 = f1_score(y_te.numpy(), preds, average='macro')
    return acc, rec, f1


# 4. 定义特征分组
# 原始特征顺序（19维）：前16个是传感器，后3个是图像计数
sensor_cols = list(range(16))  # 索引 0-15
image_cols = list(range(16, 19))  # 索引 16,17,18

X_train_sensor = X_train_t[:, :, sensor_cols]
X_test_sensor = X_test_t[:, :, sensor_cols]
X_train_img = X_train_t[:, :, image_cols]
X_test_img = X_test_t[:, :, image_cols]

# 5. 运行消融实验
print("\n===== Sensor only =====")
acc_sen, rec_sen, f1_sen = train_eval(X_train_sensor, y_train_t, X_test_sensor, y_test_t, len(sensor_cols))

print("\n===== Image only =====")
acc_img, rec_img, f1_img = train_eval(X_train_img, y_train_t, X_test_img, y_test_t, len(image_cols))

print("\n===== All features =====")
acc_all, rec_all, f1_all = train_eval(X_train_t, y_train_t, X_test_t, y_test_t, 19)

# 6. 输出结果表格
print("\n" + "=" * 50)
print("Ablation Study Results")
print("=" * 50)
print(f"{'Modality':<15} {'Accuracy':<10} {'Recall (Cold)':<15} {'Macro F1':<10}")
print("-" * 50)
print(f"{'Sensor only':<15} {acc_sen:.3f}{'':>6} {rec_sen:.3f}{'':>11} {f1_sen:.3f}")
print(f"{'Image only':<15} {acc_img:.3f}{'':>6} {rec_img:.3f}{'':>11} {f1_img:.3f}")
print(f"{'All features':<15} {acc_all:.3f}{'':>6} {rec_all:.3f}{'':>11} {f1_all:.3f}")