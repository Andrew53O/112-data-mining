import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer

# 加载训练集和测试集数据
train_data = pd.read_csv('data/train_data.csv', index_col='id')
train_labels = pd.read_csv('data/train_label.csv', index_col='id')
test_data = pd.read_csv('data/test_data.csv', index_col='id')
test_labels = pd.read_csv('data/test_label.csv', index_col='id')

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

# 填补训练数据中的0值
#train_data = fill_zeros_with_knn(train_data)

# 填补测试数据中的0值
#test_data = fill_zeros_with_knn(test_data)

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
classes = np.append(train_labels['Class'].unique(), ['PRAD', 'COAD'])

# 使用K均值聚类对未知类别中的样本进行分组
if not unknown_data.empty:
    kmeans = KMeans(n_clusters=len(classes), random_state=42)
    kmeans.fit(train_data)  # 使用训练集数据进行聚类
    unknown_predictions = kmeans.predict(unknown_data)
    test_predictions[unknown_indices] = [classes[label] for label in unknown_predictions]

# 输出结果
print("預測結果：")
print(test_predictions)

# 输出与测试标签对比的准确性
test_labels_array = test_labels.values.ravel()
accuracy = np.mean(test_predictions == test_labels_array)
print("\預測正確率：", accuracy)
