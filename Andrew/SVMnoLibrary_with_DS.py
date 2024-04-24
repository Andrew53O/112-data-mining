import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class SVM():
  # Initialize the parameters
  def __init__(self, lrate, iterate_count, lambdaa):
    self.lrate = lrate
    self.iterate_count = iterate_count
    self.lambdaa = lambdaa

  # Function for training the model
  def fit(self, X, Y):
    # Init some variable
    self.X = X
    self.Y = Y
    self.m = X.shape[0] # rows
    self.n = X.shape[1] # columns

    self.Weight = np.zeros(self.n) # numpy array filled with zero
    self.Bias = 0

    # Gradient Descent iteration
    for _ in range(self.iterate_count):
      outcome = np.array([1 if y > 0 else -1 for y in self.Y]) # make it for binary classification
    
      for i, row_data in enumerate(self.X):
        # check if data is outside the margin
        if (outcome[i] * (np.dot(row_data, self.Weight) - self.Bias) >= 1):
          dweight = 2 * self.lambdaa * self.Weight
          dbias = 0
        else:
          # if it's inside the margin, increase the margin
          dweight = 2 * self.lambdaa * self.Weight - np.dot(row_data, outcome[i])
          dbias = outcome[i]
      
        self.Weight -= self.lrate * dweight
        self.Bias -= self.lrate * dbias

  # Predict the label for a given input value
  def predict(self, test_data):
    output = np.dot(test_data, self.Weight) - self.Bias
    predicted = np.array([helper(x) for x in output])
    result = np.where(predicted <= -1, 0, 1)

    return result  

def helper(x):
  if x < 0:
    return -1
  elif x > 0:
    return 1
  else:
    return 0

# Load data from CSV file
Train_data = pd.read_csv('Data/B/train_data.csv')
Test_data = pd.read_csv('Data/B/test_data.csv')

# Initialize parameters
lrate = 0.01
iterate_count = 1000
lambdaa = 0.01

# Split to parameters and outcome data 
X_train = Train_data.iloc[:, :7]
Y_train = Train_data.iloc[:, 8]
X_test = Test_data.iloc[:, :7]
Y_test = Test_data.iloc[:, 8]

# Data Standardization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# SVM model
model = SVM(lrate = lrate, iterate_count = iterate_count, lambdaa = lambdaa)

# Train the model with training data
model.fit(X_train, Y_train)

X_test = model.predict(X_test)
test_acc = accuracy_score(Y_test, X_test)

print("Testing Accuracy: ", round(test_acc * 100, 3), "%")