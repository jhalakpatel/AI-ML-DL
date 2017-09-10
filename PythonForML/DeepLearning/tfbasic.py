import tensorflow as tf
# TF - can define the model in abstract terms. then run the graph - reason of optimization
# computation graph - we can have number of nodes, layers, and graph
# running the session - will go through the optimizer - will modify the weights 
# we just have tell tensorflow - tf will take care of how to update the weight
x1 = tf.constant(5)
x2 = tf.constant(6)
# simple multiplication in the graph
result = tf.multiply(x1, x2)

# need to create a session to run the graph
#sess = tf.Session()
#print(sess.run(result))
#sess.close()

# will close autmatically
with tf.Session() as sess:
    # output is a simple python variable - not a tf variable
    output = sess.run(result)
    print(output)

print(output)
