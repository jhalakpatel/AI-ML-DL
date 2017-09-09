import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

# Support Vector Machine Class - we want it to be object - so we can save it

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # training
    def fit(self, data):
        self.data = data
        # { ||w|| : [w,b] }
        opt_dict = {}
        transforms =[[1,1],
                        [-1,1],
                        [-1,-1],
                        [1,-1]]

        # need to get maximum and minimum in dataset
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)


        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # support vectors y1(xi.w + b) == 1
        # 1.01

        # different step sizes
        step_sizes = [self.max_feature_value * 0.1,
                        self.max_feature_value * 0.01,
                        self.max_feature_value * 0.001
                     ]

        # extremly exprensive - b did not need to take smaller steps
        b_range_multiple = 1
        b_multiple = 5
        
        latest_optimum = self.max_feature_value * 10
        
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # we can do this because convex
            optimized = False

            while not optimized:
                # this loop can be threaded to populate the dictionary
                for b in np.arange(-1*self.max_feature_value*b_range_multiple,
                        self.max_feature_value*b_range_multiple, step*b_multiple):
                    
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # weakest link in SVM fundamentally
                        # SMO attempts to fix this bit
                        # run this calculation in all the data
                        # yi(xi*w+b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False

                                #print(xi, ':', yi*(np.dot(w_t,xi)+b))

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                # iterate and change 'b' till the w is optimized i.e. w[0] < 0
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step
            
            # get the list of sorted norms
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step*2
            # we are using linear kernel for this optimization problem
        
    # prediction
    def predict(self, feature):
        # sign (x.w + b)
        classification = np.sign(np.dot(np.array(feature), self.w)+self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(feature[0], feature[1], s=200, marker='*', c=self.colors[classification])
        return classification


    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplan = x.w + b
        def hyperplane(x,w,b,v):
            # v = values of hyperplane we want
            # v = x.w + b
            # we care for v in psv = 1
            # nsv = -1
            # decision boundary= 0
            return (-w[0]*x-b+v) / w[1]
            
        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        # psv1 and psv2 are - y coordinate
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        # nsv1 and nsv2 are - y coordinate
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max], [nsv1, nsv2], 'k')
    
        # (w.x+b) = 0
        # Decision boundary support vector hyperplane
        # psv and nsv are - y coordinate
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2], 'y--')
        plt.show()

# simple data dictionary with two classes -1 and +1
data_dict = {-1:np.array([[1,7],
                          [2,8], 
                          [3,8]]), 
              1:np.array([[5,1],
                          [6,-1],
                          [7,3]])}


# initialize SVM class - instance
# SVM - wide street algorithm
svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us = [[0,10],
                [1,3],
                [3,5],
                [5,5],
                [5,6],
                [6,-5],
                [5,8]]

for p in predict_us:
    svm.predict(p)

svm.visualize()
