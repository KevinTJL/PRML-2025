import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(0)
x = np.linspace(-2, 2, 100)
y_true = x**3 - 2*x + 1 + 0.5 * np.random.randn(100)  # 目标函数加噪声

# 初始化参数
theta = np.random.randn(4)  # 3阶多项式拟合
alpha = 0.01  # 学习率
epochs = 1000  # 迭代次数
m = len(x)  # 数据量

# 设计矩阵（x^0, x^1, x^2, x^3）
X = np.vstack([x**i for i in range(4)]).T

# 训练
loss_history = []
for _ in range(epochs):
    y_pred = X @ theta  # 计算预测值
    loss = np.mean((y_pred - y_true)**2)  # 计算损失
    loss_history.append(loss)
    grad = (2/m) * X.T @ (y_pred - y_true)  # 计算梯度
    theta -= alpha * grad  # 梯度下降更新参数

# 绘图
plt.figure(figsize=(10, 5))
plt.scatter(x, y_true, label="Data", alpha=0.6)
plt.plot(x, X @ theta, color="red", label="Fitted Curve")
plt.legend()
plt.title("Gradient Descent for 1D Function Fitting")
plt.show()

# 绘制损失下降曲线
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Loss Curve")
plt.show()