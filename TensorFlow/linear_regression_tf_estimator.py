import tensorflow as tf

# numpy is used load, manipulate and preprocess the data
import numpy as np

# list of features, we have only one feature
# there are other types of columns which are more complicated and useful
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# estimator - front end to invoke training (fitting) and evaluation(inference)
#. predefined estimators - linear regression, linear classfication, neural 
# network classifiers and regressors
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

x_train = np.array([1.,2.,3.,4.])
y_train = np.array([0.,-1.,-2.,-3.])
x_eval = np.array([2.,5.,8.,1.])
y_eval = np.array([-1.01,-4.1,-7,0.])

input_fn = tf.estimator.inputs.numpy_input_fn({"x" : x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn({"x" : x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x" : x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

estimator.train(input_fn=input_fn, steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)

