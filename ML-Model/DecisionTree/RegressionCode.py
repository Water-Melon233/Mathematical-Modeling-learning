from sklearn.tree import DecisionTreeRegressor  # 导入决策树回归器模块
from sklearn.model_selection import train_test_split  # 导入数据划分函数模块
from sklearn.datasets import load_diabetes  # 导入糖尿病数据集模块（回归数据集）
from sklearn.metrics import mean_squared_error, r2_score  # 导入回归评估指标

# 决策树回归函数 可直接调用
def DecisionTreeRegressorFunction(X_train, y_train):
    model = DecisionTreeRegressor(
        max_depth=3,           # 最大深度
        min_samples_split=5,   # 节点分裂最少样本数
        criterion='squared_error',  # 分裂标准（回归常用：squared_error, absolute_error, friedman_mse）
        random_state=42         # 随机种子
        # etc.
    )  # 创建决策树回归器对象
    model.fit(X_train, y_train)  # 训练模型
    return model

# 加载数据（使用回归数据集）
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练模型
model = DecisionTreeRegressorFunction(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 回归评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"均方误差 (MSE): {mse:.2f}")
print(f"均方根误差 (RMSE): {rmse:.2f}")
print(f"R² 决定系数: {r2:.3f}")

# 显示前5个真实值与预测值的对比
print("\n前5个样本对比：")
for i in range(5):
    print(f"真实值: {y_test[i]:.1f}, 预测值: {y_pred[i]:.1f}")