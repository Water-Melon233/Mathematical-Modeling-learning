from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类器模块
from sklearn.model_selection import train_test_split  # 导入数据划分函数模块
from sklearn.datasets import load_iris  # 导入鸢尾花数据集模块

#决策树分类函数 可直接调用
def DecisionTreeClassifierFuction(X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=3,           # 最大深度
        min_samples_split=5,   # 节点分裂最少样本数
        criterion='gini',      # 分裂标准
        random_state=42         # 随机种子
        #etc.
    )  # 创建决策树分类器对象
    model.fit(X_train, y_train)  # 训练模型
    return model

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练模型
model = DecisionTreeClassifierFuction(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 准确率
print(f"准确率: {sum(y_pred == y_test) / len(y_test):.2f}")