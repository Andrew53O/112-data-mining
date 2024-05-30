import numpy as np
import csv

class KNN:
    def __init__(self, k=3):
        self.k = k

    # 計算兩點之間的歐氏距離
    def euclidean_distance(self, x1, x2):
        # 排除零值特徵
        mask = (x1 != 0) & (x2 != 0)
        return np.sqrt(np.sum(((x1 - x2) * mask)**2))

    # 訓練方法
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # 預測方法
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    # 預測單一樣本
    def _predict(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common


# 計算正確率
def accuracy(y_true, y_pred):
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        return correct / total

# 讀取CSV文件
def load_data(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳過標題行
        data = []
        for row in reader:
            data.append(row)
    return data

# 處理資料
def preprocess_data(data):
    data = np.array(data, dtype=float)
    X = data[:, :-1]  # 所有特徵
    y = data[:, -1]   # 最後一列是目標值
    return X, y

if __name__ == "__main__":
    # 讀取資料
    data = load_data('Data/A/train_data.csv')
    
    # 預處理資料
    X, y = preprocess_data(data)

    # 初始化KNN分類器
    knn = KNN(k=9)

    # 訓練模型
    knn.fit(X, y)

    # 假設我們有一個新樣本
    test_data = load_data('Data/A/test_data.csv')  # 這只是一個示例
    X1, y1 = preprocess_data(test_data)  # 不需要目標值y

    predictions = knn.predict(X1)
    print("預測結果:", predictions)

    acc = accuracy(y1, predictions)
    print(f"正確率: {acc * 100:.2f}%")