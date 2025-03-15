import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

# 读取CSV文件，仅导入第一列和第二列
file_path = "./PRML-2025/work_1/data.csv"  # 替换为你的CSV文件路径
train_data = []
test_data = []
df = pd.read_csv(file_path, usecols=[0, 1])
train_data = df.values
# test_data = df.iloc[:, 1].values
train_data = train_data.T

lambda1 = train_data.shape[1]
lambda2 = train_data.sum()      
x2 = np.dot(train_data[0],train_data[0]) #sum x^2
sum_xy = train_data[1]@train_data[0]
n_xy = train_data[1].mean() * train_data[0].mean() * lambda1
n_x_head2 = train_data[0].mean() * train_data[0].mean() * lambda1

a = (sum_xy - n_xy)/(x2 - n_x_head2)
b = train_data[1].mean() - a * train_data[0].mean()
# A = np.array([[lambda1, lambda2],[lambda2,lambda3]])
# B = np.array([[train_data[1].sum()],[]])
# a,b = np.linalg.solve(A, B)


# 提取散点数据
x_train = train_data.T[:, 0]  # 取 x 坐标
y_train = train_data.T[:, 1]  # 取 y 坐标

# 生成用于绘制直线的 x 值
x_line = np.linspace(min(x_train) - 1, max(x_train) + 1, 100)
y_line = a * x_line + b

# 绘制散点
plt.scatter(x_train, y_train, color='red', label="Train Data", marker='o')

# 绘制直线
plt.plot(x_line, y_line, color='blue', linestyle='-', linewidth=2, label=f"y = {a}x + {b}")

# 设置图例
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Function with Scatter Data")

# 显示图像
plt.show()
