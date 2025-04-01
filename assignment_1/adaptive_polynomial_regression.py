import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

# 读取CSV文件
file_path_train = "./assignment_1/data_train.csv"  
file_path_test = "./assignment_1/data_test.csv"

df_train = pd.read_csv(file_path_train, usecols=[0, 1])
df_test = pd.read_csv(file_path_test, usecols=[0, 1])

# 转换数据
train_data = df_train.values.T  
test_data = df_test.values.T  

# 训练数据
x_train = train_data[0].reshape(-1, 1)
y_train = train_data[1]

# 交叉验证选择最佳多项式阶数
degrees = range(1, 50)
kf = KFold(n_splits=5, shuffle=True, random_state=0)
mse_list = []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(x_train)

    model = LinearRegression()
    scores = cross_val_score(model, X_poly, y_train, cv=kf, scoring='neg_mean_squared_error')
    mse = -scores.mean()
    mse_list.append(mse)

# 找到最优阶数
best_degree = degrees[np.argmin(mse_list)]
print(f"最优多项式阶数: {best_degree}")

# 训练最终模型
poly = PolynomialFeatures(degree=best_degree)
X_train_poly = poly.fit_transform(x_train)
model = LinearRegression()
model.fit(X_train_poly, y_train)

# **获取多项式回归模型的系数**
theta = model.coef_  
intercept = model.intercept_  

# **生成多项式方程**
equation_terms = [f"{theta[i]:.4f} * x^{i}" for i in range(1, best_degree + 1)]
equation = " + ".join(equation_terms)
equation = f"y = {intercept:.4f} + " + equation  

# **输出最终的多项式方程**
print("拟合多项式方程:")
print(equation)

# **定义拟合函数**
def fit_function(x_input):
    x_input = np.array(x_input).reshape(-1, 1)
    X_input_poly = poly.transform(x_input)
    return model.predict(X_input_poly)

# 计算 Train Data 误差
y_train_pred = fit_function(x_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)

# 计算 Test Data 误差
x_test = test_data[0].reshape(-1, 1)
y_test = test_data[1]
y_test_pred = fit_function(x_test)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)

# 输出误差
print(f"Train Data MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
print(f"Test Data MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

# 生成拟合曲线
x_fit = np.linspace(min(x_train), max(x_train), 100).reshape(-1, 1)
y_fit = fit_function(x_fit)

# 绘制拟合曲线
plt.figure(figsize=(10, 5))
plt.scatter(x_train, y_train, label="Train Data", color="blue", alpha=0.6)
plt.plot(x_fit, y_fit, color="red", label=f"Best Fit (degree={best_degree})")
plt.scatter(x_test, y_test, color="green", label="Test Data", alpha=0.6)
plt.legend()
plt.title("Polynomial Curve Fitting with Optimal Degree")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# **示例调用拟合函数**
x_test_values = np.array([1, 2, 3, 4, 5])  
y_test_values = fit_function(x_test_values)  
print("测试输入:", x_test_values)
print("拟合输出:", y_test_values)