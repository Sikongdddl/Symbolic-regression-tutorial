import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ==== 1. 构造数据（目标函数：y = sin(x0) + x1^2） ====
n_samples = 200
X = torch.rand(n_samples, 2) * 4 - 2  # x0, x1 
true_y = torch.sin(X[:, 0]) + X[:, 1] ** 2

# ==== 2. 定义 EQL 层（支持 sin, square, identity） ====
class EQLLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)  # 权重连接

    def forward(self, x):
        z = self.linear(x)  # shape: (batch, out_dim)
        return torch.cat([
            z,                        # linear
            torch.sin(z),            # sin(z)
            z ** 2,                  # square(z)
        ], dim=1)

# ==== 3. 构建完整网络 ====
class EQLNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = EQLLayer(input_dim, 10)
        self.final = nn.Linear(10 * 3, 1)  # 10 units × 3 transforms

    def forward(self, x):
        h = self.layer1(x)
        return self.final(h).squeeze(-1)

# ==== 4. 训练模型 ====
model = EQLNetwork(input_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    pred = model(X)
    loss = F.mse_loss(pred, true_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# ==== 5. 可视化结果 ====
pred_y = model(X).detach().numpy()
plt.scatter(true_y, pred_y, s=10)
plt.xlabel("True y")
plt.ylabel("Predicted y")
plt.title("EQL Regression")
plt.grid(True)
plt.savefig("EQL_regression.png")