import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ==== Модель ====
class RadiusNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.model(x)


# ==== Загрузка данных ====
data = pd.read_csv("bpa_dataset.csv")
X = data[["num_points", "avg_dist"]].values
y = data[["r1", "r2", "r3"]].values

# Масштабируем входы
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)

# Разделение
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Torch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# ==== Обучение ====
model = RadiusNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 10
train_loss_log = []
val_loss_log = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    # Валидация
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)

    train_loss_log.append(loss.item())
    val_loss_log.append(val_loss.item())

    print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

# ==== Сохранение ====
torch.save({
    "model_state_dict": model.state_dict(),
    "scaler_mean": x_scaler.mean_,
    "scaler_scale": x_scaler.scale_
}, "model.pt")

print("\n✅ Модель сохранена как model.pt")

# ==== График ====
plt.plot(train_loss_log, label="Train")
plt.plot(val_loss_log, label="Val")
plt.title("MSE Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
