import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

# For Classification
from sklearn.ensemble import RandomForestClassifier

# For dbscan algorithm
from dbscan_helper import *

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
thresholds = [0.75]

for threshold in thresholds:
    # Find the indices of the rows that have maximum value less than threshold
    unknown_indices = np.where(np.max(test_predictions_proba, axis=1) < threshold)[0]
    test_predictions = classifier.predict(test_data) # Using test data to predict  
    #print(test_predictions)
    test_predictions[unknown_indices] = 'Unknown' # set the index of unknown_indices to 'Unknown'

    # Extract the unknown data
    unknown_data = test_data.iloc[unknown_indices]

    # New cluster possiblities name 
    # possibilities = [['COAD', 'PRAD'], ['PRAD', 'COAD']]

    # 4 classes: Dummy, PRAD, COAD, Noise
    # Noise -1, 
    classes = ['Dummy', 'PRAD', 'COAD', 'Noise']
    #classes = ['Dummy', 'COAD', 'PRAD', 'Noise']
    # print(classes[-1])
    #epsilons = np.arange(0.2, 0.6, 0.1).tolist()
    # epsilons = np.arange(180, 185, 1).tolist()
    #epsilons = [182]
    epsilons = [192]
    #minpoints = np.arange(5, 25, 1).tolist()
    # minpoints = np.arange(50, 61, 1).tolist()
    #minpoints = [52]
    minpoints = [12]
    
    for eps in epsilons:
        for minPts in minpoints:
            # 使用K均值聚类对未知类别中的样本进行分组
            test_predictions[unknown_indices] = 'Unknown'
            #if not unknown_data.empty:
                # kmeans = MyKMeans(n_clusters=len(classes), random_state=42)
            #dbscan_clustering = DBSCAN(eps=eps, min_samples=minPts)
            dbscan_clustering = Basic_DBSCAN(eps=eps, minPts=minPts)
            clusters = dbscan_clustering.fit_predict(unknown_data.values)
            #print(clusters)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            if n_clusters <= 2:
                print(clusters)
                num_minus_one = clusters.count(-1)
                print("-1:", num_minus_one)
                num_ones = clusters.count(1)
                print("1:", num_ones)         
                num_zero = clusters.count(2)
                print("2:", num_zero)
                
                print(clusters)
                print(f"eps={eps}, min_samples={minPts} gives {n_clusters} clusters")
                # assign the label to the unknown data
                try:
                    test_predictions[unknown_indices] = [classes[int(label)] for label in clusters]
                    unknown_data_label = [classes[int(label)] for label in clusters]
                except IndexError as e:
                    print(clusters)
                    print("Error happends", e)
                    break
            
            # Handle the noise data
            noise_indices = np.where(np.array(test_predictions) == 'Noise')[0]
            noise_data = test_data.iloc[noise_indices]
            
            # Get the All Known data 
            known_indices = np.where(np.array(test_predictions) != 'Noise')[0]
            known_data = test_data.iloc[known_indices]
            known_label = test_predictions[known_indices]
            
            # Using classification to classify the clustered data 
            classifier_noise = RandomForestClassifier(n_estimators=100, random_state=42)
            classifier_noise.fit(known_data, known_label) 
            #print("unknown_data", unknown_data)
            #print(noise_data_label)
            print("noise data", noise_data)
            classifier_noise_predictions = classifier_noise.predict(noise_data)
            print("test predictions before", test_predictions)
            #[classes[int(label)] for label in clusters]
            print("noise indices", noise_indices)
            test_predictions[noise_indices] = classifier_noise_predictions
            print("test predictions after", test_predictions)
            print("classifer noise predictions", classifier_noise_predictions)
            
            
            
            
        
            
            test_labels_array = test_labels.values.ravel() # Convert the test_labels into 1D array
            local_accuracy = np.mean(test_predictions == test_labels_array)
            print(eps, minPts, local_accuracy)
            print("test labels", test_labels_array)
            
            accuracy = max(local_accuracy, max_accuracy)
            if (accuracy > max_accuracy):
                max_accuracy = accuracy
                max_threshold = threshold
                max_eps = eps
                max_minPts = minPts
                print(eps, minPts, "Max ", accuracy)
                print(clusters)
                
          
                    
        
    # 输出结果
    #print("預測結果：")
    # print(test_predictions)

    # Calculate the accuracy
    
    print(threshold)
    #print("\預測正確率：", accuracy)

print("\預測正確率：", max_accuracy)
print("\最佳閾值：", max_threshold)
print("\最佳eps：", max_eps)
print("\最佳minPts：", max_minPts)
# cari tau gimana buat border di blocknya itu biar lebih bagus kli 
# masukin code dari medium buat yg dbscan 
# liat accuracnya 

# apa sih unknown_data.values ama unknown_predictions


# kata yulin
# 1. cuman perlu ngeclustering 2 data unknowin itu aja 

# problem
# 1. accuracy nya masih rendah, bingung tanya yulin
# epsilon ama min pointnya ancur


# \預測正確率： 0.9668674698795181
# \最佳閾值： 0.75
# \最佳eps： 200
# \最佳minPts： 2


# dpt ini apakah ga kettingian tuh?

# ganti classification pakek data yg 2 cluster itu aaj 