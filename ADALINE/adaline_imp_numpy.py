import torch
import torch.nn as nn
import torch.optim as optim

# ===================== 1. 准备数据集（AND 逻辑，线性可分，适配二分类）=====================
# 输入：2维特征（x0=1 偏置项已融入模型nn.Linear的bias，此处无需手动添加）
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
# 目标标签：采用 ADALINE 常用的 1/-1 标签（区别于感知机的 0/1）
y = torch.tensor([[-1], [-1], [-1], [1]], dtype=torch.float32)

# ===================== 2. 定义 ADALINE 模型（核心：线性输出用于训练）=====================
class ADALINE(nn.Module):
    def __init__(self, input_dim):
        super(ADALINE, self).__init__()
        # 线性层：输入维度 → 1个连续输出（用于训练，计算误差）
        self.linear = nn.Linear(input_dim, 1)
        # 阶跃函数：仅用于推理阶段的分类输出，不参与训练（避免梯度报错）
        self.step = lambda x: torch.where(x >= 0, torch.tensor(1.0), torch.tensor(-1.0))

    def forward(self, x, train=True):
        # 前向传播：训练时返回线性输出（用于梯度计算），推理时返回阶跃分类结果
        linear_out = self.linear(x)
        if train:
            return linear_out  # 训练：输出连续值，用于计算均方误差、反向传播
        else:
            return self.step(linear_out)  # 推理：输出离散分类结果（1/-1）

# ===================== 3. 初始化模型、优化器、损失函数（贴合 LMS 规则）=====================
input_dim = 2  # 输入特征维度（AND逻辑为2维）
model = ADALINE(input_dim=input_dim)
# 优化器：SGD（贴合 LMS 梯度下降思想），学习率可调整（0~1）
optimizer = optim.SGD(model.parameters(), lr=0.01)
# 损失函数：MSELoss（均方误差，完全对应 ADALINE 的 LMS 规则）
criterion = nn.MSELoss()

# ===================== 4. 训练模型（LMS 梯度下降，无梯度报错）=====================
epochs = 10000  # 训练轮数，确保收敛
for epoch in range(epochs):
    # 1. 前向传播（训练模式，返回线性连续输出）
    y_pred_linear = model(X, train=True)
    
    # 2. 计算均方误差（基于线性输出，而非阶跃输出）
    loss = criterion(y_pred_linear, y)
    
    # 3. 反向传播 + 权重更新（LMS 规则的 PyTorch 实现，梯度正常传播）
    optimizer.zero_grad()  # 清空上一轮梯度
    loss.backward()        # 计算梯度（线性输出可导，无报错）
    optimizer.step()       # 沿负梯度更新权重
    
    # 打印训练日志（每1000轮打印一次，查看收敛情况）
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# ===================== 5. 测试模型（推理模式，输出离散分类结果）=====================
print("\n==== ADALINE 测试结果 ====")
with torch.no_grad():  # 推理时禁用梯度计算，节省资源
    y_pred = model(X, train=False)  # 切换为推理模式，返回阶跃分类结果
    print("输入\t\t预测值\t真实值")
    for i in range(len(X)):
        print(f"{X[i].numpy()}\t{int(y_pred[i].item())}\t{int(y[i].item())}")

# ===================== 6. 查看训练后的权重和偏置=====================
print("\n==== 训练后的参数（权重 + 偏置）====")
# 提取线性层的权重（w1, w2）和偏置（w0）
weights = model.linear.weight.detach().numpy()[0]
bias = model.linear.bias.detach().numpy()[0]
print(f"权重 w1, w2: {weights.round(4)}")
print(f"偏置 w0: {bias.round(4)}")
