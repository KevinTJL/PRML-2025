import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 读取CSV文件，仅导入第一列和第二列
file_path_train = "./work_1/data_train.csv"  # 替换为你的CSV文件路径
file_path_test = "./work_1/data_test.csv"

df_train = pd.read_csv(file_path_train, usecols=[0, 1])
df_test = pd.read_csv(file_path_test, usecols=[0, 1])

# 转换数据
train_data = df_train.values.T  # 转置
test_data = df_test.values.T  # 转置

# 设计矩阵（x^0, x^1）
x_train = train_data[0]
y_train = train_data[1]

# 构造设计矩阵 X（包含偏置项）
X_train = np.vstack([np.ones_like(x_train), x_train]).T  
y_train = y_train.reshape(-1, 1)

# 计算最小二乘解（解析解）
theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# **定义拟合函数**
def fit_function(x_input):
    """ 计算线性回归预测值 """
    return theta[0] + theta[1] * x_input

# **计算 Train Data 误差**
y_train_pred = fit_function(x_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)

# **计算 Test Data 误差**
x_test = test_data[0]
y_test = test_data[1]
y_test_pred = fit_function(x_test)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)

# **输出误差**
print(f"Train Data MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
print(f"Test Data MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

# **生成拟合曲线**
x_fit = np.linspace(min(x_train), max(x_train), 1000)
y_fit = fit_function(x_fit)

# **绘图**
plt.figure(figsize=(10, 5))
plt.scatter(x_train, y_train, label="Train Data", color="blue")
plt.scatter(test_data[0], test_data[1], color="green", label="Test Data", alpha=0.6)
plt.plot(x_fit, y_fit, color="red", label=f"Fitted Line: y = {theta[1, 0]:.4f} * x + {theta[0, 0]:.4f}")
plt.legend()
plt.title("Linear Curve Fitting using Newton's Method")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# **输出拟合直线方程**
print(f"拟合直线方程: y = {theta[1, 0]:.4f} * x + {theta[0, 0]:.4f}")

# **示例调用拟合函数**
x_test_values = np.array([1, 2, 3, 4, 5])  
y_test_values = fit_function(x_test_values)  
print("测试输入:", x_test_values)
print("拟合输出:", y_test_values.flatten())