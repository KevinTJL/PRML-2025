import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# 设计矩阵（x^0, x^1, x^2, x^3）
x = train_data[0]
y_true = train_data[1]

# 构造设计矩阵
X = np.vstack([np.ones_like(x), x]).T
y = y_true.reshape(-1, 1)

# 计算 Hessian 和梯度
H = 2 * X.T @ X  # Hessian 矩阵
grad = 2 * X.T @ (X @ np.random.randn(2, 1) - y)  # 随机初始化 theta

# 计算参数的更新步长
theta = np.linalg.inv(H) @ grad  # 牛顿法更新

# 计算最终的参数（实际只需要一步）
theta = np.linalg.inv(X.T @ X) @ X.T @ y  # 直接用最小二乘解

# 预测拟合曲线
x_fit = np.linspace(0, 10, 1000)
X_fit = np.vstack([np.ones_like(x_fit), x_fit]).T
y_fit = X_fit @ theta  # 预测曲线

# 绘图
plt.scatter(x, y_true, label="Train_Data", color="blue")
plt.scatter(test_data[0], test_data[1],color = "green", label="Test_Data", alpha=0.6)
plt.plot(x_fit, y_fit, color="red", label="Newton Fit (Least Squares)")
plt.legend()
plt.title("Linear Curve Fitting using Newton's Method")
plt.show()

# 输出拟合参数
print("Fitted parameters (theta):", theta.flatten())