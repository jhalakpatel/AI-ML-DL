import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

# Algorithm:
# step 1: make every feature set as cluster center 
# step 2: take all the feature set bandwidth or radius and take the mean of feature set and assign or update new cluster center
# step 3: repeat step2 until convergence , many cluster center will converge and few them will stop moving


class Mean_Shift:
    def __init__(self, radius=2):
        self.radius = radius

    def fit(self, data):
        centroids = {}
        
        # set feature set equal to the centroid - initialize centroids
        for i in range(len(data)):
            centroids[i] = data[i]

        while True:
            new_centroids = []
            for i in centroids:
                in_radius = []
                centroid = centroids[i]
                
                # iterate through the data and decide feature set in bw
                for featureset in data:
                    if np.linalg.norm(featureset-centroid) < self.radius:
                       in_radius.append(featureset)
                
                # get mean of the all the feature set
                new_centroid = np.average(in_radius, axis=0)

                # convert the centroid array to tuple
                new_centroids.append(tuple(new_centroid))

            # get unique centroids
            # we can get set of tuples - we want to get a unique of a tuple
            # in numpy we will get unique of the array not the whole value
            # thus we are converting array to a tuple
            uniques = sorted(list(set(new_centroids)))
            
            # simply copy the older centroids
            prev_centroids = dict(centroids)

            centroids = {}
            # iterate through all the unique centroids
            for i in range(len(uniques)):
                # for each centroid upadate - i.e. convert teh unqiue[i]
                # to an array and store them in the centroids dict
                centroids[i] = np.array(uniques[i])

            optimized = True

            # as centroids are sorted, we can compare the current centroids with 
            # the prevoius centroids
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break

            if optimized:
                break

        self.centroids = centroids

    def predict(self, data):
        pass


# create the input data set
X = np.array([[1,2],
            [1.5,1.8],
            [5,8],
            [8,8],
            [1,0.6],
            [9,11],
            [8,2],
            [10,2],
            [9,3]])

colors = 10*['g', 'r', 'c', 'b', 'k', 'm']

# create the classifier - hierarchical classifier 
clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids
plt.scatter(X[:,0], X[:,1],s=100)
for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=100)

plt.show()
