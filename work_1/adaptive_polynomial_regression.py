import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

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

# 生成数据
x = train_data[0]
y = train_data[1]

x = x.reshape(-1, 1)  # 变成列向量

# 交叉验证参数
degrees = range(1, 50)  # 选择 1 到 10 阶
kf = KFold(n_splits=5, shuffle=True, random_state=0)  # 5 折交叉验证
mse_list = []

# 遍历不同的阶数
for d in degrees:
    poly = PolynomialFeatures(degree=d)  # 生成 d 阶多项式特征
    X_poly = poly.fit_transform(x)

    model = LinearRegression()
    scores = cross_val_score(model, X_poly, y, cv=kf, scoring='neg_mean_squared_error')
    mse = -scores.mean()  # 取反，MSE 越小越好
    mse_list.append(mse)

# 找到最优阶数
best_degree = degrees[np.argmin(mse_list)]
print(f"最优多项式阶数: {best_degree}")

# 画出 MSE 变化趋势
plt.figure(figsize=(8, 5))
plt.plot(degrees, mse_list, marker='o', linestyle='-')
plt.xlabel("Polynomial Degree")
plt.ylabel("Cross-Validation MSE")
plt.title("Degree Selection using Cross-Validation")
plt.show()

# 使用最优阶数训练最终模型
poly = PolynomialFeatures(degree=best_degree)
X_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

# 绘制拟合曲线
plt.scatter(x, y, label="Train_Data", color="blue", alpha=0.6)
plt.plot(x, y_pred, color="red", label=f"Best Fit (degree={best_degree})")
plt.scatter(test_data[0], test_data[1],color = "green", label="Test_Data", alpha=0.6)
plt.legend()
plt.title("Polynomial Curve Fitting with Optimal Degree")
plt.show()