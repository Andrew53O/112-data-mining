from scipy.spatial import distance

class AndrewDBSCAN:
    # Initialize the DBSCAN object
    def __init__(self, eps, minimalPts, metric=distance.euclidean):
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
            expand(data, clusters, point, neighbors, currentPointLabel, eps, minimalPts, metric)
    
    return clusters

# Find the neighbors of a given point
def neighborsFind(data, point, eps, metric):
    neighbors = []
    
    for i in range(data.shape[0]):
        # If the distance to the point is less than eps, add it to the list of neighbors
        if metric(data[point], data[i]) < eps:
            neighbors.append(i)
    
    return neighbors

# This function expands the cluster from a given `point` until the boundaries of the neighborhood are reached. It assigns the `point` and all points in `neighbors` to the current cluster (`currentPointLabel`).
def expand(data, clusters, point, neighbors, currentPointLabel, eps, minimalPts, metric):
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
            
            nextNeighbors = neighborsFind(data, nextPoint, eps, metric)
            
            # if the point is a core point, add its neighbors to the list of neighbors 
            if len(nextNeighbors) >= minimalPts:
                neighbors = neighbors + nextNeighbors
        
        # Move to the next neighbor  
        i += 1