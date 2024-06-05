import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer

train_data = pd.read_csv('data/train_data.csv', index_col='id')
train_labels = pd.read_csv('data/train_label.csv', index_col='id')
test_data = pd.read_csv('data/test_data.csv', index_col='id')
test_labels = pd.read_csv('data/test_label.csv', index_col='id')

def compute_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def fill_zeros_with_knn(data, n_neighbors=5):
    # 记录哪些列是全零的
    zero_cols = data.columns[(data == 0).all()]
    non_zero_data = data.drop(columns=zero_cols)

    imputer = KNNImputer(n_neighbors=n_neighbors, missing_values=0)
    data_imputed = imputer.fit_transform(non_zero_data)

    # 将填补后的数据转换回 DataFrame
    data_imputed_df = pd.DataFrame(data_imputed, columns=non_zero_data.columns, index=data.index)

    # 创建一个全零的 DataFrame
    zero_cols_df = pd.DataFrame(0, index=data.index, columns=zero_cols)

    # 将全零的列加回去
    data_imputed_df = pd.concat([data_imputed_df, zero_cols_df], axis=1)

    # 按照原始列的顺序重新排序
    data_imputed_df = data_imputed_df.reindex(columns=data.columns)

    return data_imputed_df

# # 填补训练数据中的0值
train_data = fill_zeros_with_knn(train_data)

# #填补测试数据中的0值
test_data = fill_zeros_with_knn(test_data)

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

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(train_data, train_labels.values.ravel())
test_predictions_proba = classifier.predict_proba(test_data)

threshold = 0.65

unknown_indices = np.where(np.max(test_predictions_proba, axis=1) < threshold)[0]
test_predictions = classifier.predict(test_data)
test_predictions[unknown_indices] = 'Unknown'
test_labels_array = test_labels.values.ravel()

a = 0
b = 0

for i in range(len(test_predictions)):
    if test_labels_array[i] != 'PRAD' and test_labels_array[i] != 'COAD':
        if test_predictions[i] == test_labels_array[i]:
            a += 1
        b += 1

print(test_predictions)
print(test_labels_array)
print(a/b*100, "%")

unknown_data = test_data.iloc[unknown_indices]

classes = ['PRAD', 'COAD']

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