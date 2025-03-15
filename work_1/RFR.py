import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
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
x = train_data[0].reshape(len(test_data[0]),1)
y = train_data[1].reshape(len(test_data[1]),1)

# 训练随机森林回归模型
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf.fit(x, y)

# 预测
x_pred = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = rf.predict(x_pred)

# 绘图
plt.scatter(x, y, color="blue", label="Train_Data", alpha=0.6)
plt.scatter(test_data[0], test_data[1],color = "green", label="Test_Data", alpha=0.6)
plt.plot(x_pred, y_pred, color="red", label="Random Forest Regression")
plt.legend()
plt.title("Random Forest Regression")
plt.show()