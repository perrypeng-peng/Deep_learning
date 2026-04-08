import torch

# 数据
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([0,0,0,1], dtype=torch.float32)

# 初始化权重 + 偏置
w = torch.randn(2, requires_grad=False)
b = torch.tensor(0.0, requires_grad=False)
lr = 0.1

# 原始感知机学习规则
epochs = 20
for epoch in range(epochs):
    errors = 0
    for i in range(len(X)):
        x = X[i]
        target = y[i]
        z = torch.dot(x, w) + b
        y_pred = 1.0 if z >= 0 else 0.0

        # 更新规则
        if y_pred != target:
            w += lr * (target - y_pred) * x
            b += lr * (target - y_pred)
            errors += 1
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, 错误数量: {errors}")

# 测试
print("\n测试结果：")
for i in range(len(X)):
    x = X[i]
    z = torch.dot(x, w) + b
    pred = 1 if z >=0 else 0
    print(f"{x.numpy()} -> 预测：{pred}, 真实：{int(y[i])}")