import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import pandas as pd
import re
import sys
from contextlib import redirect_stdout
import io

# For Classification
from sklearn.ensemble import RandomForestClassifier

# For dbscan algorithm
from scipy.spatial import distance

# for finding the best hyperparameters
from sklearn.metrics import silhouette_score
from bayes_opt import BayesianOptimization


class AndrewDBSCANwrapper:
    # Initialize the DBSCAN object
    def __init__(self, eps, minimalPts, metric = distance.euclidean):
        self.eps = eps # eps -> radius of each point to find neighbors
        self.minimalPts = minimalPts # minimalPts -> minimum number of points to form a cluster
        self.metric = metric # metric -> distance metric to calculate distance between points
    
    def fit_predict(self, data):
        # create an array of zeros to store cluster labels
        clusters = [0] * data.shape[0]
        DBSCAN(data, clusters, self.eps, self.minimalPts, self.metric)
        return clusters

# DBSCAN algorithm
def DBSCAN(data, clusters, eps, minimalPts, metric = distance.euclidean):
    currentPointLabel = 0
    
    for point in range(0, data.shape[0]):
        # if point is already labeled,
        if clusters[point] != 0: 
            continue
    
        # Find the neighbors of the current point
        neighbors = neighborsFind(data, point, eps, metric)

        # if point is noise
        if len(neighbors) < minimalPts:
            clusters[point] = -1

        else:
            # keep expanding the cluster 
            currentPointLabel += 1
            expandCluster(data, clusters, point, neighbors, currentPointLabel, eps, minimalPts, metric)
    
    return clusters

# Find the neighbors of a given point
def neighborsFind(data, point, eps, metric):
    neighborsList = []
    
    for i in range(data.shape[0]):
        # If the distance to the point is less than eps, add it to the list of neighbors
        if metric(data[point], data[i]) < eps:
            neighborsList.append(i)
    
    return neighborsList

# This function expands the cluster from a given `point` until the boundaries of the neighborhood are reached. It assigns the `point` and all points in `neighbors` to the current cluster (`currentPointLabel`).
def expandCluster(data, clusters, point, neighbors, currentPointLabel, eps, minimalPts, metric):
    clusters[point] = currentPointLabel
    
    i = 0
    while i < len(neighbors):
        # Get the next neighbor
        nextPoint = neighbors[i]
        # It's still a the cluster point, if the NEXT neighbor is a noise, but no need to find the neighbors of the noise
        if clusters[nextPoint] == -1: 
            clusters[nextPoint] = currentPointLabel
        
        # Assign the point to the cluster if it hasn't been assigned yet
        elif clusters[nextPoint] == 0:
            clusters[nextPoint] = currentPointLabel
            
            nextNeighbrs = neighborsFind(data, nextPoint, eps, metric)
            
            # if the point is a core point, add its neighbors to the list of neighbors 
            if len(nextNeighbrs) >= minimalPts:
                neighbors = neighbors + nextNeighbrs
        
        # Move to the next neighbor  
        i += 1

# Load data
train_data = pd.read_csv('../Data/train_data.csv', index_col='id')
train_labels = pd.read_csv('../Data/train_label.csv', index_col='id')
test_data = pd.read_csv('../Data/test_data.csv', index_col='id')
test_labels = pd.read_csv('../Data/test_label.csv', index_col='id')

# Using RandomForestClassifier classifier, train the model using train data and predict the test
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(train_data, train_labels.values.ravel()) # Convert labels into 1D array
test_predictions_proba = classifier.predict_proba(test_data) # Predict each data belongs to what class label

# Get the probabilities of the first class
class1_probabilities = test_predictions_proba[:, 0]
class2_probabilities = test_predictions_proba[:, 1]
class3_probabilities = test_predictions_proba[:, 2]

# Plot the histogram
plt.hist(class1_probabilities, bins=10, alpha=0.5, label='Class 1')
plt.hist(class2_probabilities, bins=10, alpha=0.5, label='Class 2')
plt.hist(class3_probabilities, bins=10, alpha=0.5, label='Class 3')

plt.title('Distribution Probabilities of 3 known classes')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.legend(loc='upper right')

# plt.savefig('All3Class.jpg', format='jpg', dpi=600) # save the plot
# plt.show() # show the plot 

# Find the maximum value of each row
max_values = np.max(test_predictions_proba, axis=1) # row-wise
plt.hist(max_values, bins=10, alpha=0.5, label='Classes')
plt.title('Distribution Probabilities of max 3 known classes')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.legend(loc='upper right')

# plt.savefig('Max3Class.jpg', format='jpg', dpi=600) # save the plot
# plt.show()  # show the plot 

# Settings for the threshold, we found 0.75 is the best threshold after trying different values
threshold = [0.75]

# Find the indices of the rows that have maximum value less than threshold
unknown_indices = np.where(np.max(test_predictions_proba, axis=1) < threshold[0])[0]
test_predictions = classifier.predict(test_data) # Using test data to predict  
test_predictions[unknown_indices] = 'Unknown' # set the index of unknown_indices to 'Unknown'

# Extract the unknown data
unknown_data = test_data.iloc[unknown_indices]

# 4 classes: Dummy, PRAD, COAD, Noise
# Dbscan will return -1 for the noise data
# Dbscan will not return 0, so index 0 is for "dump"
# return 1 for the first class, 2 for the second class
classes = ['Dump', 'PRAD', 'COAD', 'Noise']
# 2 different permuation possibilities
newclasses = [['Dump', 'PRAD', 'COAD', 'Noise'], ['Dump', 'COAD', 'PRAD', 'Noise']]

# Define the range of the hyperparameters
pbounds = {'eps': (0.1, 200), 'minPts': (1, 20)}

# Define the function to optimize
def dbscan_func(eps, minPts):
    minPts = int(minPts)
    dbscan = AndrewDBSCANwrapper(eps=eps, minimalPts=minPts) # call our dbscan clustering
    clusters = dbscan.fit_predict(unknown_data.values)
    # Check if more than one cluster is formed
    if len(set(clusters)) > 1:
        score = silhouette_score(unknown_data.values, clusters)
    else:
        score = -1  # return a low score
    return score

# Initialize the optimizer we use BayesianOptimization
optimizer = BayesianOptimization(
    f=dbscan_func,
    pbounds=pbounds,
    random_state=1,
)

# Create a string buffer to capture the output
buffer = io.StringIO()

# Perform optimization
with redirect_stdout(buffer):
    optimizer.maximize(
        init_points=2, # perform 2 random steps before starting bayesian optimization
        n_iter=5, # perform 3 steps of bayesian optimization 
    )


# Function to strip ANSI codes
def strip_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

# Get the output and strip ANSI codes
output = strip_ansi_codes(buffer.getvalue())
print(output) # output after process the ansi code

# # Best epsilons and minpoints
epsilons = optimizer.max['params']['eps']
minpoints = optimizer.max['params']['minPts']

# Use DBSCAN clustering to cluster the unknown data
dbscan_clustering = AndrewDBSCANwrapper(eps=epsilons, minimalPts=minpoints)
clusters = dbscan_clustering.fit_predict(unknown_data.values)

# print the count of -1, 1, 1 after clustering 
num_minus_one = clusters.count(-1)
print("-1:", num_minus_one)
num_ones = clusters.count(1)
print("1:", num_ones)         
num_zero = clusters.count(2)
print("2:", num_zero)
        

max_accuracy = 0
for classesis in newclasses: # for finding the best permutation of classes
    # Get the number of clusters 
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    if n_clusters <= 2:
        # Assign the label to the unknown data
        test_predictions[unknown_indices] = [classesis[int(label)] for label in clusters]

    # Handle the noise data
    noise_indices = np.where(np.array(test_predictions) == 'Noise')[0]
    if(noise_indices.size > 0): # if there is noise data
        noise_data = test_data.iloc[noise_indices]
        # Get the All Known data (All 5 class)
        known_indices = np.where(np.array(test_predictions) != 'Noise')[0]
        known_data = test_data.iloc[known_indices]
        known_label = test_predictions[known_indices]

        # Using classification to classify uknown data after clustering 
        classifier_noise = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier_noise.fit(known_data, known_label) 
        classifier_noise_predictions = classifier_noise.predict(noise_data)
    
    # Assign the result to the overall test predictions
    test_predictions[noise_indices] = classifier_noise_predictions
    test_labels_array = test_labels.values.ravel() # Convert the test_labels into 1D array
    local_accuracy = np.mean(test_predictions == test_labels_array)
    if (local_accuracy > max_accuracy): 
        max_accuracy = local_accuracy
        
# Print the test labels and test predictions
print("Test Labels     : ", ' '.join(map(str, test_labels_array)))
print("Test Predictions: ", ' '.join(map(str, test_predictions.ravel())))
print("\n預測正確率：", max_accuracy)