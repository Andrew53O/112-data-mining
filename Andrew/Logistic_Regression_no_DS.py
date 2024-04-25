import pandas as pd
import numpy as np

# Load data from CSV file
Train_data = pd.read_csv('Data/B/train_data.csv')
Test_data = pd.read_csv('Data/B/test_data.csv')

# Split to parameters and outcome data 
X_train = Train_data.iloc[:, :7].values.T # Transpose data
Y_train = Train_data.iloc[:, 8].values.reshape(1, X_train.shape[1]) # Reshape data to (1, n)
X_test = Test_data.iloc[:, :7].values.T
Y_test = Test_data.iloc[:, 8].values.reshape(1, X_test.shape[1])

# Initialize parameters
iterate_count = 100000
lrate = 0.00027 # manually find the best learning rate 

class Logistic_Regression():
    # Initialize the parameters
    def __init__(self, lrate, iterate_count):
        self.lrate = lrate
        self.iterate_count = iterate_count
    
    # Train the model 
    def fit(self, X, Y):
        # Init some variable
        self.X = X
        self.Y = Y
        self.m = X_train.shape[1]
        self.n = X_train.shape[0]
        
        weight = np.zeros((self.n, 1)) # numpy array filled with zero
        bias = 0
        
        for _ in range(iterate_count):
            # Calculate the sigmoid
            res_sigmoid = calc_sigmoid(self.X, weight, self.bias)
            
            # adjusting the weight and bias using cost function + gradient descent
            dWeight = (1/self.m) * np.dot(res_sigmoid - self.Y, self.X.T)
            dBias = (1/self.m) * np.sum(res_sigmoid - self.Y)
            
            weight -= lrate * dWeight.T
            bias -= lrate * dBias
        
        # Lastly change the weight and the bias 
        self.weight = weight
        self.bias = bias
            
    
    def accuracy(self, test_data, Actual_value):
        # Calculate the sigmoid with our weight andd bias
        Predicted_value = calc_sigmoid(test_data, self.Weight, self.Bias)
        Predicted_value = np.array(Predicted_value > 0.5, dtype='int64') # Change to 0 or 1
            
        acc = (1 - np.sum(np.absolute(Predicted_value - Actual_value)) / Actual_value.shape[1])
        
        return acc
        
       
def helper_sigmoid(x):
    if np.any(x < -100): # if x is too negative 
        return np.where(x < -100, 0, 1 / (1 + np.exp(-x)))
    else:
        return 1 / (1 + np.exp(-x)) # sigmoid function
        
# Calculate sigmoid or probabilistic predictions between 0 and 1
def calc_sigmoid(X, Weight, Bias):
    res = np.dot(Weight.T, X) + Bias
    return helper_sigmoid(res)

model = Logistic_Regression(lrate, iterate_count)
model.fit(X_train, Y_train)
test_acc = model.accuracy(X_test, Y_test)

print("Accuracy: ", round(test_acc * 100, 3), "%")

