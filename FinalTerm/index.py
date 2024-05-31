import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer

# 加载训练集和测试集数据
train_data = pd.read_csv('../Data/train_data.csv', index_col='id')
train_labels = pd.read_csv('../Data/train_label.csv', index_col='id')
test_data = pd.read_csv('../Data/test_data.csv', index_col='id')
test_labels = pd.read_csv('../Data/test_label.csv', index_col='id')

def compute_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# 定义K-means算法
class MyKMeans:
    def __init__(self, n_clusters, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        random_indices = np.random.permutation(X.shape[0])
        self.centroids = X[random_indices[:self.n_clusters]]

        for _ in range(self.max_iters):
            self.labels = self.assign_clusters(X)
            old_centroids = self.centroids
            self.centroids = self.update_centroids(X)

            if np.all(old_centroids == self.centroids):
                break

    def assign_clusters(self, X):
        labels = np.zeros(X.shape[0])
        for i, point in enumerate(X):
            distances = [compute_distance(point, centroid) for centroid in self.centroids]
            labels[i] = np.argmin(distances)
        return labels

    def update_centroids(self, X):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            points = X[self.labels == i]
            if len(points) > 0:
                centroids[i] = np.mean(points, axis=0)
        return centroids

    def predict(self, X):
        labels = self.assign_clusters(X)
        return labels

# 使用随机森林分类器对所有样本进行分类
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(train_data, train_labels.values.ravel())
test_predictions_proba = classifier.predict_proba(test_data)

# 定义阈值来确定未知类别的样本
threshold = 0.75

# 根据概率确定未知类别的样本
unknown_indices = np.where(np.max(test_predictions_proba, axis=1) < threshold)[0]
test_predictions = classifier.predict(test_data)
test_predictions[unknown_indices] = 'Unknown'

# 提取未知类别的样本
unknown_data = test_data.iloc[unknown_indices]

# 将新增的两个类别名称添加到原有的类别列表中

classes = np.append(train_labels['Class'].unique(), ['COAD', 'PRAD'])

# 使用K均值聚类对未知类别中的样本进行分组
if not unknown_data.empty:
    kmeans = MyKMeans(n_clusters=len(classes), random_state=42)
    kmeans.fit(train_data.values)  # 使用训练集数据进行聚类
    unknown_predictions = kmeans.predict(unknown_data.values)
    test_predictions[unknown_indices] = [classes[int(label)] for label in unknown_predictions]

# 输出结果
print("預測結果：")
print(test_predictions)

# 输出与测试标签对比的准确性
test_labels_array = test_labels.values.ravel()
accuracy = np.mean(test_predictions == test_labels_array)
print("\預測正確率：", accuracy)
