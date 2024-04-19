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
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = [self.y_train[i] for i in nearest_indices]
        most_common = Counter(nearest_labels).most_common(1)
        return most_common[0][0]


#def read_csv(file_path):
    #data = []
    #with open(file_path, 'r') as file:
        #csv_reader = csv.reader(file)
        #for row in csv_reader:
            #data.append(row)
    #return data

#file_path = "A/test_data.csv"
#csv_data = read_csv(file_path)

#print(csv_data)

def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    X = data[:, :-1]  # 所有特徵
    y = data[:, -1]   # 最後一列是目標變量
    return X, y

# 讀取資料
X, y = load_data('A/train_data.csv')  # 請替換成你的CSV檔案名稱

# 初始化並訓練KNN模型
knn = KNN(k=3)
knn.fit(X, y)

# 預測
# 這裡我假設你有另一組測試數據，你可以根據需要替換或加入你的測試數據
X_test = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50],
                   [1, 85, 66, 29, 0, 26.6, 0.351, 31],
                   [8, 183, 64, 0, 0, 23.3, 0.672, 32]])

predictions = knn.predict(X_test)
print("Predictions:", predictions)