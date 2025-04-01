from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

num_ = 100
# 读取CSV文件，仅导入第一列和第二列
file_path_train = "./assignment_1/data_train.csv"  # 替换为你的CSV文件路径
file_path_test = "./assignment_1/data_test.csv"

df_train = pd.read_csv(file_path_train, usecols=[0, 1])
df_test = pd.read_csv(file_path_test, usecols=[0, 1])

# 转换数据
train_data = df_train.values.T
test_data = df_test.values.T

# 训练数据
x_train = train_data[0].reshape(-1, 1)
y_train = train_data[1].reshape(-1, 1)

# 训练 SVR 模型
svr = SVR(kernel='rbf', C=num_, gamma=0.3, epsilon=0.1)
svr.fit(x_train, y_train.ravel())

# 预测 Train Data
y_train_pred = svr.predict(x_train)

# 计算 Train Data 误差
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)

# 预测 Test Data
x_test = test_data[0].reshape(-1, 1)
y_test = test_data[1].reshape(-1, 1)
y_test_pred = svr.predict(x_test)

# 计算 Test Data 误差
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)

# 输出误差
print(f"Train Data MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
print(f"Test Data MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

# 生成拟合曲线
x_pred = np.linspace(min(x_train), max(x_train), num_).reshape(-1, 1)
y_pred = svr.predict(x_pred)

# 绘图
plt.figure(figsize=(10, 5))
plt.scatter(x_train, y_train, color="blue", label="Train Data", alpha=0.6)
plt.scatter(x_test, y_test, color="green", label="Test Data", alpha=0.6)
plt.plot(x_pred, y_pred, color="red", label="SVR (RBF Kernel)")
plt.legend()
plt.title("Support Vector Regression (SVR)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()