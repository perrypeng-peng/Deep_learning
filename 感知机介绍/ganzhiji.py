import numpy as np

# 感知机类
class Perceptron:
    def __init__(self, input_size=2, lr=0.1):
        self.w = np.random.randn(input_size)
        self.b = np.random.randn()
        self.lr = lr

    def step(self, x):
        return 1 if x >= 0 else 0

    def forward(self, x):
        return self.step(np.dot(x, self.w) + self.b)

    def update(self, x, y, pred):
        # 误差更新（感知机学习规则）
        self.w += self.lr * (y - pred) * x
        self.b += self.lr * (y - pred)

# XOR 数据集
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([0, 1, 1, 0])

# 训练
model = Perceptron()
for epoch in range(100):
    for x, y in zip(X, Y):
        pred = model.forward(x)
        model.update(x, y, pred)

# 测试
print("XOR 测试结果：")
for x in X:
    print(x, "→", model.forward(x))