import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 讀取資料
train_data = pd.read_csv('data/train_data.csv', index_col='id')
train_labels = pd.read_csv('data/train_label.csv', index_col='id')
test_data = pd.read_csv('data/test_data.csv', index_col='id')
test_labels = pd.read_csv('data/test_label.csv', index_col='id')

def dis_cal(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class K_means: # K-means
    def __init__(self, clusters, iters=100, state=42):
        self.clusters = clusters
        self.iters = iters
        self.state = state

    def cluster_assign(self, X):
        labels = np.zeros(X.shape[0])
        for i, point in enumerate(X):
            distance = [dis_cal(point, centroid) for centroid in self.centroids]
            labels[i] = np.argmin(distance)
        return labels

    def centroids_update(self, X):
        centroids = np.zeros((self.clusters, X.shape[1]))
        for i in range(self.clusters):
            points = X[self.labels == i]
            if len(points) > 0:
                centroids[i] = np.mean(points, axis=0)
        return centroids
    
    def fit(self, X): # 訓練
        np.random.seed(self.state)
        indices = np.random.permutation(X.shape[0])
        self.centroids = X[indices[:self.clusters]]

        for _ in range(self.iters): # 找中心點
            self.labels = self.cluster_assign(X)
            old_cen = self.centroids
            self.centroids = self.centroids_update(X)

            if np.all(old_cen == self.centroids): # 找到結果，中心沒再改變
                break

    def predict(self, X): # 預測結果
        labels = self.cluster_assign(X)
        return labels

# 分類
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(train_data, train_labels.values.ravel())
test_predictions_proba = classifier.predict_proba(test_data)

# 定義 unknown 門檻
threshold = 0.75

# 將低於0.75的設為 unknown
unknown_indices = np.where(np.max(test_predictions_proba, axis=1) < threshold)[0]
test_predictions = classifier.predict(test_data)
test_predictions[unknown_indices] = 'Unknown'

# 將 unknown 資料拿出
unknown_data = test_data.iloc[unknown_indices]

# 新標籤名稱
classes = ['PRAD', 'COAD']

# 將 unknown 資料分成兩群
if not unknown_data.empty:
    kmeans = K_means(2, state=42)
    kmeans.fit(train_data.values)
    unknown_predictions = kmeans.predict(unknown_data.values)
    test_predictions[unknown_indices] = [classes[int(label)] for label in unknown_predictions]

print("預測結果：")
print(test_predictions)

test_labels_array = test_labels.values.ravel()
accuracy = np.mean(test_predictions == test_labels_array)
print("\預測正確率：", accuracy)
