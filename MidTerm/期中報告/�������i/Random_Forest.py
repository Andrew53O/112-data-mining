import pandas as pd
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier

columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

train_data = pd.read_csv('B/train_data.csv')
test_data = pd.read_csv('B/test_data.csv')

X_train = train_data.drop('Outcome', axis=1)
Outcome_train = train_data['Outcome']

X_test = test_data.drop('Outcome', axis=1)
Outcome_test = test_data['Outcome']

times = 15 # 計算次數
num = 250 #隨機森林的數量

result = 0

for j in range(times):
    random_indices = X_train.sample(n = num, random_state = random.randint(0, 100)).index # data set
    random_data = X_train.loc[random_indices]
    random_out = Outcome_train.loc[random_indices]

    selected_columns = random.sample(random_data.columns.tolist(), random.randint(4, 8))
    random_data = random_data[selected_columns]
    random_test_data = X_test[selected_columns]

    clf = DecisionTreeClassifier()
    clf.fit(random_data, random_out)
        
    y_pred = clf.predict(random_test_data)
    result += y_pred

for i in range(len(result)):
    if result[i] > times/2:
        result[i] = 1
    else:
        result[i] = 0

accuracy = np.sum(Outcome_test == result) / len(result)
print(f"Accuracy: {accuracy * 100:.2f}%")