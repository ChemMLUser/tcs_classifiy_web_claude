import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1. 加载你自己的数据集
# 👇 修改这里的路径为你自己的文件位置
data = pd.read_csv("./dataset.txt", sep="\s+", engine='python')  # 例如: data.csv 或 ./data/train_data.csv

# 2. 假设最后一列是标签，前12列是特征
X = data.iloc[:, :-1].values  # 前12列作为特征
y = data.iloc[:, -1].values   # 最后一列作为标签

# 3. 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 初始化并训练 KNN 分类器
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 5. 模型评估
y_pred = knn.predict(X_test)
print("✅ 准确率:", accuracy_score(y_test, y_pred))
print("📊 分类报告:\n", classification_report(y_test, y_pred))

# 6. 保存模型
# 创建模型保存目录（如果不存在）
os.makedirs("models", exist_ok=True)
joblib.dump(knn, "models/knn_model.pkl")
print("📁 模型已保存到 models/knn_model.pkl")