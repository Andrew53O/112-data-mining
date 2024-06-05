import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv('B/train_data.csv')
test_data = pd.read_csv('B/test_data.csv')

X_train = train_data.drop('Outcome', axis=1)
Outcome_train = train_data['Outcome']

X_test = test_data.drop('Outcome', axis=1)
Outcome_test = test_data['Outcome']

# 決策樹
clf = DecisionTreeClassifier()
clf.fit(X_train, Outcome_train)

# 預測
y_pred = clf.predict(X_test)

print(f"Accuracy: {np.sum(y_pred == Outcome_test) / len(Outcome_test) * 100:.3f}%")
