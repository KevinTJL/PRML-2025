import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

# 读取CSV文件
file_path_train = "./work_1/data_train.csv"  # 替换为你的CSV文件路径
file_path_test = "./work_1/data_test.csv"

df_train = pd.read_csv(file_path_train, usecols=[0, 1])
df_test = pd.read_csv(file_path_test, usecols=[0, 1])

# 转换数据
train_data = df_train.values.T  # 转置
test_data = df_test.values.T  # 转置

# 训练数据
x_train = train_data[0].reshape(-1, 1)  # 转换为列向量
y_train = train_data[1].reshape(-1, 1)

# 训练随机森林回归模型
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf.fit(x_train, y_train.ravel())  # 训练模型

# **定义拟合函数**
def fit_function(x_input):
    """ 使用训练好的随机森林模型预测 y 值 """
    x_input = np.array(x_input).reshape(-1, 1)
    return rf.predict(x_input)

# **计算 Train Data 误差**
y_train_pred = fit_function(x_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)

# **计算 Test Data 误差**
x_test = test_data[0].reshape(-1, 1)
y_test = test_data[1].reshape(-1, 1)
y_test_pred = fit_function(x_test)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)

# **输出误差**
print(f"Train Data MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
print(f"Test Data MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

# **生成拟合曲线**
x_pred = np.linspace(min(x_train), max(x_train), 100).reshape(-1, 1)
y_pred = fit_function(x_pred)

# **绘图**
plt.figure(figsize=(10, 5))
plt.scatter(x_train, y_train, color="blue", label="Train Data", alpha=0.6)
plt.scatter(x_test, y_test, color="green", label="Test Data", alpha=0.6)
plt.plot(x_pred, y_pred, color="red", label="Random Forest Regression")
plt.legend()
plt.title("Random Forest Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# **示例调用拟合函数**
x_test_values = np.array([1, 2, 3, 4, 5])  # 示例输入
y_test_values = fit_function(x_test_values)  # 计算拟合值
print("测试输入:", x_test_values)
print("拟合输出:", y_test_values)