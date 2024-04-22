import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv('A/train_data.csv')

test_data = pd.read_csv('A/test_data.csv')

X_train = train_data.drop('Outcome', axis=1)
Outcome_train = train_data['Outcome']

X_test = test_data.drop('Outcome', axis=1)
Outcome_test = test_data['Outcome']

total = 0 # 總正確率
times = 10 # 計算次數

for i in range(times):
    # 決策樹
    clf = DecisionTreeClassifier()
    clf.fit(X_train, Outcome_train)

    # 預測
    y_pred = clf.predict(X_test)

    total += np.sum(Outcome_test == y_pred) / len(y_pred)

print(f"Average Accuracy in {times} times: {total * 100 / times:.2f}%")
