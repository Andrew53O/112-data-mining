import numpy as np
import pandas as pd
from collections import Counter
import csv

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) 
                     for x_train in self.X_train 
                     if not np.any(x_train == 0)]  # 排除任何特徵值為 0 的樣本
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = [self.y_train[i] for i in nearest_indices]
        most_common = Counter(nearest_labels).most_common(1)
        return most_common[0][0]

def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    X = data[:, :-1]  # 所有特徵
    y = data[:, -1]   # 最後一列是目標變量
    return X, y

# 讀取資料
X, y = load_data('Data/A/train_data.csv')  # 請替換成你的CSV檔案名稱

# 初始化並訓練KNN模型
knn = KNN(k=3)
knn.fit(X, y)

# 預測
# 這裡我假設你有另一組測試數據，你可以根據需要替換或加入你的測試數據
X_test, _ = load_data('Data/A/test_data.csv')

predictions = knn.predict(X_test)
#print("Predictions:", predictions)