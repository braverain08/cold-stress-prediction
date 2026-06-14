import numpy as np
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

# 加载数据
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print("数据加载完成")
print(f"训练集: {X_train_flat.shape}, 测试集: {X_test_flat.shape}")

# 1. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_flat, y_train)
y_pred_rf = rf.predict(X_test_flat)
acc_rf = accuracy_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='macro')
print(f"Random Forest: Acc={acc_rf:.3f}, Recall={rec_rf:.3f}, F1={f1_rf:.3f}")

# 2. XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_flat, y_train)
y_pred_xgb = xgb_model.predict(X_test_flat)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
rec_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb, average='macro')
print(f"XGBoost    : Acc={acc_xgb:.3f}, Recall={rec_xgb:.3f}, F1={f1_xgb:.3f}")

# 3. LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
lgb_model.fit(X_train_flat, y_train)
y_pred_lgb = lgb_model.predict(X_test_flat)
acc_lgb = accuracy_score(y_test, y_pred_lgb)
rec_lgb = recall_score(y_test, y_pred_lgb)
f1_lgb = f1_score(y_test, y_pred_lgb, average='macro')
print(f"LightGBM   : Acc={acc_lgb:.3f}, Recall={rec_lgb:.3f}, F1={f1_lgb:.3f}")

# 4. GRU
class GRUPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

classes = np.unique(y_train)
cw = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = torch.tensor([cw[0], cw[1]], dtype=torch.float32)

model = GRUPredictor(input_dim=X_train.shape[2])
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

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
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)
    if (epoch+1) % 50 == 0:
        print(f"GRU Epoch {epoch+1}, Loss: {avg_loss:.4f}")

model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    preds_gru = logits.argmax(dim=1).numpy()
acc_gru = accuracy_score(y_test, preds_gru)
rec_gru = recall_score(y_test, preds_gru)
f1_gru = f1_score(y_test, preds_gru, average='macro')
print(f"GRU         : Acc={acc_gru:.3f}, Recall={rec_gru:.3f}, F1={f1_gru:.3f}")

print("\n===== Summary =====")
print(f"Random Forest: Acc={acc_rf:.3f}, Recall={rec_rf:.3f}, F1={f1_rf:.3f}")
print(f"XGBoost     : Acc={acc_xgb:.3f}, Recall={rec_xgb:.3f}, F1={f1_xgb:.3f}")
print(f"LightGBM    : Acc={acc_lgb:.3f}, Recall={rec_lgb:.3f}, F1={f1_lgb:.3f}")
print(f"GRU         : Acc={acc_gru:.3f}, Recall={rec_gru:.3f}, F1={f1_gru:.3f}")