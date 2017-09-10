import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

class K_Means:
    # tol - how much centroid is going to move
    # iter - total number of iteration to find the centroid
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        
    def fit(self, data):
        self.centroids = {}
        # simply intialize the centroids with first two points - can be random as well
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            # will contains centroids and classifications
            # for every iterations the clusters and centroids are changing - we need to clear it out
            self.classifications = {}
            for i in range(self.k):
                self.classifications[i] = [] # simply store the data sets in centroids
            for featureset in data:
                # for a particular featureset in data - find the norm distance from all the centroids
                # store the distances in a list
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                # get min distance - get the index to the mindistance
                classification = distances.index(min(distances))
                
                # bucket of all the featureset belonging to a centroid
                self.classifications[classification].append(featureset)

            # compare the two centroids
            # copy as a dictionary
            prev_centroids = dict(self.centroids)
            
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid - original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break
        
    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances)) # find the min distance with all the centroids
        return classification


#### dataset logic
X = np.array([[1, 2],
            [1.5, 1.8],
            [5,8],
            [8,8],
            [1,.6],
            [9,11], [1,3], [8,9], [0,13], [5,4], [6,4]])

plt.scatter(X[:,0], X[:,1], s=100, color='b')
#plt.show()
colors = ["g", "r", "c", 'b', 'k', 'y']

#### simple program logic
clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker='o', color='k', s=150, linewidth=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidth=5)

"""
unknowns = np.array([[1,3], [8,9], [0,13], [5,4], [6,4]])

for unk in unknowns:
    classification = clf.predict(unk)
    plt.scatter(unk[0], unk[1], marker="*", color=colors[classification], s=100, linewidth=5)
"""
plt.show()
