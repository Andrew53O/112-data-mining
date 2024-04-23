import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV file
Train_data = pd.read_csv('Data/A/train_data.csv')
Test_data = pd.read_csv('Data/A/test_data.csv')

# Split to parameters and outcome data 
X_train = Train_data.iloc[:, :7].values.T # Transpose data
Y_train = Train_data.iloc[:, 8].values.reshape(1, X_train.shape[1]) # Reshape data to (1, n)
X_test = Test_data.iloc[:, :7].values.T
Y_test = Test_data.iloc[:, 8].values.reshape(1, X_test.shape[1])

# Initialize parameters
iterate_count = 100000
lrate = 0.00027


# Sigmoid function
def sigmoid(x):
    if np.any(x < -100): # if x is too negative 
        return np.where(x < -100, 0, 1 / (1 + np.exp(-x)))
    else:
        return 1 / (1 + np.exp(-x))

# Calculate sigmoid or probabilistic predictions between 0 and 1
def calc_sigmoid(X, Weight, Bias):
    res = np.dot(Weight.T, X) + Bias
    return sigmoid(res)

def Logistic_Regression(X, Y, lrate, iterate_count):
    m = X_train.shape[1]
    n = X_train.shape[0]
    
    Weight = np.zeros((n,1)) # numpy array filled with zero
    Bias = 0
    cost_all = []
    
    for _ in range(iterate_count):
        # Calculate the sigmoid
        A = calc_sigmoid(X, Weight, Bias)
        
        # Cost function -> Error representation
        cost = -(1/m)*np.sum(Y * np.log(A + 1e-9) + (1-Y) * np.log(1 - A + 1e-9))
        
        # Gradient Descent
        dWeight = (1/m) * np.dot(A - Y, X.T)
        dBias = (1/m) * np.sum(A - Y)
        
        Weight = Weight - lrate * dWeight.T
        Bias = Bias - lrate * dBias
        
        cost_all.append(cost)
        
    return Weight, Bias, cost_all

def accuracy(X, Actual_value, Weight, Bias):
    # Calculate the sigmoid
    Predicted_value = calc_sigmoid(X, Weight, Bias)
    Predicted_value = np.array(Predicted_value > 0.5, dtype='int64') # Change to 0 or 1
        
    acc = (1 - np.sum(np.absolute(Predicted_value - Actual_value)) / Actual_value.shape[1]) * 100
    
    print("Accuracy: ", round(acc, 2), "%") 


Weight, Bias, cost_all = Logistic_Regression(X_train, Y_train, lrate = lrate, iterate_count = iterate_count)

# Draw the cost function
plt.figure(num="Cost Function", figsize=(10, 6))
plt.plot(np.arange(iterate_count), cost_all)
plt.title("Cost Function Plot")
plt.show()

accuracy(X_test, Y_test, Weight, Bias)