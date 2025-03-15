import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

num_ = 1
# 读取CSV文件，仅导入第一列和第二列
file_path_train = "./PRML-2025/work_1/data_train.csv"  # 替换为你的CSV文件路径
file_path_test = "./PRML-2025/work_1/data_test.csv"
train_data = []
test_data = []
df = pd.read_csv(file_path_train, usecols=[0, 1])
train_data = df.values
df = pd.read_csv(file_path_test, usecols=[0, 1])
test_data = df.values.T
train_data = train_data.T

# 初始化参数
theta = np.random.randn(num_)  # n阶多项式拟合
alpha = 0.01  # 学习率
epochs = 1000  # 迭代次数
m = train_data.T.shape[0]  # 数据量

x = train_data[0]
X = np.vstack([x**i for i in range(num_)]).T
y_true = train_data[1]
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
plt.scatter(x, y_true, label="Train_Data", alpha=0.6)
plt.scatter(test_data[0], test_data[1],color = "blue", label="Test_Data", alpha=0.6)
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