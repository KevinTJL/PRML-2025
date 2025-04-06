import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# 假设下面这部分代码已经运行过，生成了训练数据 X 和 labels（共1000个数据点）
# ------------------------------------------------------
def make_moons_3d(n_samples=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # 第三维的正弦变化
    X = np.vstack([np.column_stack([x, y, z]),
                   np.column_stack([-x, y - 1, -z])])
    y_labels = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    X += np.random.normal(scale=noise, size=X.shape)
    return X, y_labels

# 原先生成的训练数据（1000个数据点，每类500个）
X, labels = make_moons_3d(n_samples=1000, noise=0.2)

# 绘制训练数据（可选）
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', marker='o')
# legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
# ax.add_artist(legend1)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title('3D Make Moons (Training Data)')
# plt.show()
# ------------------------------------------------------

# 直接使用原先生成的训练数据
X_train, y_train = X, labels

# 为测试生成新数据：500个数据点（每类250个）
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)

# ----------------------- 模型训练 -----------------------

# 1. 决策树
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

# 2. AdaBoost + 决策树（基础分类器采用深度为1的决策树）
ada_clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    random_state=42
)
ada_clf.fit(X_train, y_train)
y_pred_ada = ada_clf.predict(X_test)
acc_ada = accuracy_score(y_test, y_pred_ada)

# 3. SVM 分类器（选用三种不同的核函数）
# 3.1 SVM - 线性核
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
acc_linear = accuracy_score(y_test, y_pred_linear)

# 3.2 SVM - RBF核（径向基函数核）
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
acc_rbf = accuracy_score(y_test, y_pred_rbf)

# 3.3 SVM - 多项式核（采用3次多项式）
svm_poly = SVC(kernel='poly', degree=3, random_state=42)
svm_poly.fit(X_train, y_train)
y_pred_poly = svm_poly.predict(X_test)
acc_poly = accuracy_score(y_test, y_pred_poly)

# ----------------------- 可视化函数 -----------------------
def visualize_classification(X_test, y_test, y_pred, title):
    """
    绘制测试数据的3D散点图，其中：
      - 根据预测标签着色
      - 用红色“x”标记出预测错误的样本
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 根据预测标签上色
    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2],
                         c=y_pred, cmap='viridis', marker='o', label='Predicted')
    
    # 找出误分类的样本，并用红色"X"标记
    misclassified = (y_pred != y_test)
    if np.sum(misclassified) > 0:
        ax.scatter(X_test[misclassified, 0],
                   X_test[misclassified, 1],
                   X_test[misclassified, 2],
                   c='red', marker='x', s=100, label='Misclassified')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    ax.legend()
    plt.show()

# ----------------------- 输出结果及可视化 -----------------------
print("决策树准确率:", acc_dt)
print("AdaBoost + 决策树准确率:", acc_ada)
print("SVM (线性核) 准确率:", acc_linear)
print("SVM (RBF核) 准确率:", acc_rbf)
print("SVM (多项式核) 准确率:", acc_poly)

# # 可视化各模型在测试数据上的预测结果
# visualize_classification(X_test, y_test, y_pred_dt, f"决策树预测 (准确率: {acc_dt:.2f})")
# visualize_classification(X_test, y_test, y_pred_ada, f"AdaBoost+决策树预测 (准确率: {acc_ada:.2f})")
# visualize_classification(X_test, y_test, y_pred_linear, f"SVM (线性核) 预测 (准确率: {acc_linear:.2f})")
# visualize_classification(X_test, y_test, y_pred_rbf, f"SVM (RBF核) 预测 (准确率: {acc_rbf:.2f})")
# visualize_classification(X_test, y_test, y_pred_poly, f"SVM (多项式核) 预测 (准确率: {acc_poly:.2f})")