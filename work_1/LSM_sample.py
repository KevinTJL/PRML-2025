import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

file_path = "./PRML-2025./work_1/data.csv"  # 替换为你的CSV文件路径
train_data = []
test_data = []
df = pd.read_csv(file_path, usecols=[0, 1])
data = df.values
# Define the x_samples and y_noisy values
x_samples = data.T[0]
y_noisy = data.T[1]

# Reshape x_samples for sklearn's LinearRegression
x_samples_reshaped = x_samples.reshape(-1, 1)

# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(x_samples_reshaped, y_noisy)

# Extracting the slope and intercept from the model
slope = model.coef_[0]
intercept = model.intercept_

# Predict y values using the fitted model
y_predicted = model.predict(x_samples_reshaped)

# Plotting the original noisy data
plt.figure(figsize=(8, 6))
plt.scatter(x_samples, y_noisy, color='red', label='Noisy Data')

# Plotting the linear regression line
plt.plot(x_samples, y_predicted, color='blue', label='Linear Regression Line')

# Annotate the linear equation on the plot
plt.text(1, 8, f'y = {slope:.2f}x + {intercept:.2f}', fontsize=12, color='blue')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression on Noisy Data')
plt.legend()
plt.grid(True)
plt.show()