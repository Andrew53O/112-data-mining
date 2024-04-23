import pandas as pd 
from sklearn import metrics
from sklearn.preprocessing import StandardScaler # For scaling the data
from sklearn.svm import SVC # For SVM model

# Load data from CSV file
training_df = pd.read_csv('Data/A/train_data.csv')
test_df = pd.read_csv('Data/A/test_data.csv')

# Split to parameters and outcome data 
train_pr_df = training_df.iloc[:, :7]
train_out_df = training_df.iloc[:, 8]
test_pr_df = test_df.iloc[:, :7]
test_out_df = test_df.iloc[:, 8]

# Scale the data
scaler = StandardScaler()
train_pr_df = scaler.fit_transform(train_pr_df)
test_pr_df = scaler.transform(test_pr_df) # Transform the test data using the same scaler learned from the training data

# Create a SVM model
model = SVC(kernel='rbf', random_state=0) # random_state -> Ensure the same result every time 
model.fit(train_pr_df, train_out_df)

# Predict the test data
predictions = model.predict(test_pr_df)

# Calculate the accuracy
print("Accuracy: ", round(metrics.accuracy_score(test_out_df, predictions) * 100, 3), "%")
