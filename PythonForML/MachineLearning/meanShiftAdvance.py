import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import style
style.use('ggplot')

# Algorithm:
# step 1: make every feature set as cluster center 
# step 2: take all the feature set bandwidth or radius and take the mean of feature set and assign or update new cluster center
# step 3: repeat step2 until convergence , many cluster center will converge and few them will stop moving


# advance:
# for simple version with different radius, the clustering logic might r might not work
# for advance version, we can start with large radius and penalize points - for 
# how much further the data or featureset are from the centroid in question
# closer to the centroid more weight to the data point
class Mean_Shift:
    def __init__(self, radius=None, radius_norm_step = 100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):
       
        # simply get the centroid for all the data points
        # get the norm and calculate the distance
        if self.radius == None:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step

        centroids = {}

        # set feature set equal to the centroid - initialize centroids
        for i in range(len(data)):
            centroids[i] = data[i]

        # define weights and reverse them
        weights = [i for i in range(self.radius_norm_step)][::-1]
        
        while True:
            new_centroids = []
            for i in centroids:
                in_radius = []
                centroid = centroids[i]
            
                # iterate through the data and decide feature set in bw
                for featureset in data:
                    distance = np.linalg.norm(featureset-centroid)
                    if distance == 0:
                        distance = 0.000000001
            
                    # how many radius step we need to take
                    # if the distance is greater than radius - penalize
                    # if distance is less than radius - then weight_index = 0
                    weight_index = int(distance/self.radius)
                 
                    # wight index = 3, weights[3] = 96?
                    if weight_index > self.radius_norm_step-1:
                        weight_index = self.radius_norm_step - 1

                    # for weight_index = 0, we will have 99 times more pref to the featureset and add it to the to_radius
                    # add the weighted values times ** 2 of the feature set
                    to_add = (weights[weight_index])*[featureset]
                   
                    #print('to_add :', to_add)

                    # add two lists, add updated list with weighted features to the in radius
                    in_radius += to_add

                # get mean of the all the feature set
                new_centroid = np.average(in_radius, axis=0)

                # convert the centroid array to tuple
                new_centroids.append(tuple(new_centroid))

            # get unique centroids
            # we can get set of tuples - we want to get a unique of a tuple
            # in numpy we will get unique of the array not the whole value
            # thus we are converting array to a tuple
            uniques = sorted(list(set(new_centroids)))

            to_pop = []
            # iterate overall the pairs of unique centroids
            # we need to get rid of centroids which are almost equal
            # similar to tolerance
            for i in uniques:
                for ii in  uniques:
                    if i == ii:
                        pass    
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius:
                        # if they are in one step of each other
                        # should not modify the list when we are iterating though it
                        to_pop.append(ii)
                        break;
            
            # simply remove those multiple centroids
            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass

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

        # create simple dictionary with keys are centroid and values as list of featureset
        self.classifications = {}

        for i in range(len(self.centroids)):
            self.classifications[i] = []

        for featureset in data:
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)

    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        # for the feature set, find out the distance between the all the centroid and feature set

        # get the classification i.e. cluster - i.e. one with least distance from teh centroid
        classification = distances.index(min(distances))
        return classfication

# create the input data set

centers = random.randrange(2,5)

X, y = make_blobs (n_samples=50, centers=centers, n_features=2) 
#X = np.array([[1,2],
#            [1.5,1.8],
#            [5,8],
#            [8,8],
#            [1,0.6],
#            [9,11],
#            [8,2],
#            [10,2],
#            [9,3]])
colors = 10*['g', 'r', 'c', 'b', 'k', 'm']

# create the classifier - hierarchical classifier 
clf = Mean_Shift()
clf.fit(X)

# iterate over the classification dictionary
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=100)

centroids = clf.centroids
for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=100)


plt.show()
