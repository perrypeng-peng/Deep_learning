import torch
import torch.nn as nn
import numpy as np

# ===================== 1. 数据集（AND 逻辑）=====================
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)

# ===================== 2. 感知机模型（修复版！）=====================
class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        # 用 Sigmoid 代替不可导的阶跃函数（解决梯度报错）
        # 激活函数：阶跃函数 step（感知机专用）
        #self.step = lambda x: torch.where(x >= 0, 1.0, 0.0)        
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        #out = self.step(out)
        return out

# ===================== 3. 初始化 =====================
model = Perceptron(input_dim=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
criterion = nn.MSELoss()

# ===================== 4. 训练 =====================
epochs = 5000
for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()   # ✅ 现在不会报错了！
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# ===================== 5. 测试 =====================
print("\n==== 测试结果 ====")
with torch.no_grad():
    y_pred = model(X)
    # 最后再转成 0/1（推理阶段用阶跃）
    y_pred_bin = torch.where(y_pred >= 0.5, 1.0, 0.0)

print("输入\t预测值(0/1)\t真实值")
for i in range(4):
    print(f"{X[i].numpy()}\t{int(y_pred_bin[i].item())}\t\t{int(y[i].item())}")