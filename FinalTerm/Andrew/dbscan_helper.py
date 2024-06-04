from scipy.spatial import distance
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

def simple_DBSCAN(X, clusters, eps, minPts, metric=distance.euclidean):
    """
    Driver; 
    iterates through neighborsGen for every point in X
    expands cluster for every point not determined to be noise
    """
    currentPoint = 0
    
    for i in range(0, X.shape[0]):
        if clusters[i] != 0:
            continue
    
        neighbors = neighborsGen(X, i, eps, metric)

        if len(neighbors) < minPts:
            clusters[i] = -1

        else:
            currentPoint += 1
            expand(X, clusters, i, neighbors, currentPoint, eps, minPts, metric)
    
    return clusters


def neighborsGen(X, point, eps, metric):
    """
    Generates neighborhood graph for a given point
    """
    
    neighbors = []
    
    for i in range(X.shape[0]):
        if metric(X[point], X[i]) < eps:
            neighbors.append(i)
    
    return neighbors


def expand(X, clusters, point, neighbors, currentPoint, eps, minPts, metric):
    """
    Expands cluster from a given point until neighborhood boundaries are reached
    """
    clusters[point] = currentPoint
    
    i = 0
    while i < len(neighbors):
        
        nextPoint = neighbors[i]
        
        if clusters[nextPoint] == -1:
            clusters[nextPoint] = currentPoint
        
        elif clusters[nextPoint] == 0:
            clusters[nextPoint] = currentPoint
            
            nextNeighbors = neighborsGen(X, nextPoint, eps, metric)
            
            if len(nextNeighbors) >= minPts:
                neighbors = neighbors + nextNeighbors
        
        i += 1
        
class AndrewDBSCAN:
    """
    Parameters:
    
    eps: Radius of neighborhood graph
    
    minPts: Number of neighbors required to label a given point as a core point.
    
    metric: Distance metric used to determine distance between points; 
            currently accepts scipy.spatial.distance metrics for two numeric vectors
    
    """
    
    def __init__(self, eps, minPts, metric=distance.euclidean):
        self.eps = eps
        self.minPts = minPts
        self.metric = metric
    
    def fit_predict(self, X):
        """
        Parameters:
        
        X: An n-dimensional array of numeric vectors to be analyzed
        
        Returns:
        
        [n] cluster labels
        """
    
        clusters = [0] * X.shape[0]
        
        simple_DBSCAN(X, clusters, self.eps, self.minPts, self.metric)
        
        return clusters
    

# df = pd.read_csv('../Data/test_data.csv') # ~10% of the original dataset
# cols = df.columns


# X = df[[cols[0], cols[1]]]
# X = StandardScaler().fit_transform(X)

# clusters = scanner.fit_predict(X)