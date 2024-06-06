import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('Data/train_data.csv', index_col='id')
train_labels = pd.read_csv('Data/train_label.csv', index_col='id')
test_data = pd.read_csv('Data/test_data.csv', index_col='id')
test_labels = pd.read_csv('Data/test_label.csv', index_col='id')

threshold = 0.9 * len(train_data)
columns_to_drop = train_data.columns[(train_data == 0).sum() > threshold]

train_data = train_data.drop(columns=columns_to_drop)
test_data = test_data.drop(columns=columns_to_drop)

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

threshold = 0.75

unknown_indices = np.where(np.max(test_predictions_proba, axis=1) < threshold)[0]
test_predictions = classifier.predict(test_data)
test_predictions[unknown_indices] = 'Unknown'
test_labels_array = test_labels.values.ravel()

a = 0
b = 0
unknown = 0
BRCA_test = 0
KIRC_test = 0
LUAD_test = 0
BRCA_predict = 0
KIRC_predict = 0
LUAD_predict = 0


for i in range(len(test_predictions)):
    if test_labels_array[i] != 'PRAD' and test_labels_array[i] != 'COAD':
        if test_labels_array[i] == 'BRCA':
            BRCA_test += 1
        elif test_labels_array[i] == 'KIRC':
            KIRC_test += 1
        elif test_labels_array[i] == 'LUAD':
            LUAD_test += 1
            
        if test_predictions[i] == test_labels_array[i]:
            a += 1
            
            if test_predictions[i] == 'BRCA':
                BRCA_predict += 1
            elif test_predictions[i] == 'KIRC':
                KIRC_predict += 1
            elif test_predictions[i] == 'LUAD':
                LUAD_predict += 1
           
        else:
            unknown += 1
            
        b += 1

#print(test_predictions)
#print(test_labels_array)
print(a/b*100, "%")
print(unknown/b*100, "%")
print(a, b, unknown)
print("test data total a b c", b)
print("BRCA_test", BRCA_test)
print("KIRC_test", KIRC_test)
print("LUAD_test", LUAD_test)
print("BRCA_predict", BRCA_predict)
print("KIRC_predict", KIRC_predict)
print("LUAD_predict", LUAD_predict)
print("uknown", unknown)


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

import matplotlib.pyplot as plt
import numpy as np

# Values for each group
group1 = [12, 28, 1]
group2 = [28, 7, 16, 20]

# Create the x positions of the bars
x1 = np.arange(len(group1))
x2 = np.arange(len(group2))

# Create the plot
fig, ax = plt.subplots()

# # Plot the bars
# bar_width = 0.35
# rects1 = ax.bar(x1, group1, bar_width, label='Group 1')
# rects2 = ax.bar(x2 + len(group1), group2, bar_width, label='Group 2')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Group')
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group')
# ax.set_xticks(np.append(x1, x2 + len(group1)))
# ax.set_xticklabels(['A', 'B', 'C', 'A_after', 'B_after', 'C_after', 'D_after'])
# ax.legend()

# # Show the plot
# plt.show()

