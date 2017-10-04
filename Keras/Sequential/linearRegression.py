import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# from -1 to 1, get 101 points - 0.01 for each point delta
train_X = np.linspace(-1, 1, 101)
print('train_X shape : ', train_X.shape)
train_Y = 3 * train_X + np.random.randn(*train_X.shape) * 0.33

model = Sequential()

# we only need a single connection - use dense layer with linear activation
model.add(Dense(units=1, activation="linear", input_dim=1, kernel_initializer="uniform"))
# get the initial weights - it might be using Xavier Initialization - 
# weights variance = 1/(n_out + n_in)
weights = model.layers[0].get_weights()

# get the initial weights
w_init = weights[0][0][0]
b_init = weights[1][0]

print('Linear Regression model is initialized with weights w: %2f, b: %2f' %(w_init, b_init))

# compile function will specify the loss function and optimizer
model.compile(optimizer='sgd', loss='mse')

# do the actual training
model.fit(train_X, train_Y, epochs=200, verbose=1)

# there is only one layer
weights = model.layers[0].get_weights()

w_final = weights[0][0][0]
b_final = weights[1][0]

print('Linear regression model is trained to have weight w: %.2f, b: %.2f' %(w_final, b_final))

# store the model file as hdf5 format - we can store multi terabyte datasets on 
# disk - use them as real numpy arrays
# can store multiple datasets in a single file - iterate over them or checkout
# .shape and .dtype attributes

# save weights in hdf5 format
model.save_weights('linearmodel.h5')

# load the weights
model.load_weights('linearmodel.h5')



