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
    # m -> rows, n -> columns
    self.m = X.shape[0]
    self.n = X.shape[1]
    
    # Initializen the weight value and bias value
    self.weight = np.zeros(self.n)
    self.bias = 0
    self.X = X
    self.Y = Y

    # Gradient Descent
    for _ in range(self.iterate_count):
      self.update_weights_and_bias()

  # Function for updating the weight and bias value
  def update_weights_and_bias(self):
    # Label the data
    outcome = np.array([1 if y > 0 else -1 for y in self.Y])
    
    # Slope dweight and dbias
    for index, x_i in enumerate(self.X):
      # check the condition using hinge loss function
      if (outcome[index] * (np.dot(x_i, self.weight) - self.bias) >= 1):
        dweight = 2 * self.lambdaa * self.weight
        dbias = 0
      else:
        dweight = 2 * self.lambdaa * self.weight - np.dot(x_i, outcome[index])
        dbias = outcome[index]
    
      self.weight -= self.lrate * dweight
      self.bias -= self.lrate * dbias

  # Predict the label for a given input value
  def predict(self, X):
    output = np.dot(X, self.weight) - self.bias
    predicted_labels = np.sign(output)
    result = np.where(predicted_labels <= -1, 0, 1)

    return result  

# Load data from CSV file
Train_data = pd.read_csv('Data/A/train_data.csv')
Test_data = pd.read_csv('Data/A/test_data.csv')

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

# Check the Accuracy on Training and Testing data
# X_train_pred = model.predict(X_train)
# train_acc = accuracy_score(Y_train, X_train_pred)

# print("Training Accuracy: ", round(train_acc * 100, 2), "%")

X_test = model.predict(X_test)
test_acc = accuracy_score(Y_test, X_test)

print("Testing Accuracy: ", round(test_acc * 100, 3), "%")