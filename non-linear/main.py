import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 设置随机种子
np.random.seed(0)
torch.manual_seed(0)

# 1. 数据生成
n_samples = 200
X = np.linspace(-2, 2, n_samples).reshape(-1, 1)
y_true = np.sin(2 * np.pi * X) + 0.3 * X**2
noise = np.random.normal(0, 0.1, size=y_true.shape)
y = y_true + noise

# 转为 PyTorch 张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 2. 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. 训练
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

# 4. 可视化结果
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor).numpy()

plt.figure(figsize=(8, 5))
plt.plot(X, y_true, label='True Function', color='green')
plt.scatter(X, y, label='Noisy Observations', s=10, color='blue')
plt.plot(X, y_pred, label='NN Prediction', color='red')
plt.title('Nonlinear Symbolic Regression (NN approximation)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('symbolic_regression_nn.png')

