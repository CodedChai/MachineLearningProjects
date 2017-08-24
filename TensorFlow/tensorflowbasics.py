# Tensorflow takes all of the stuff in chunks, runs it in the background, then gives the results
# Going to usually just give some things to tensorflow then it'll give back results
# Can use an interactive session if really desired but not recommended
# Think of TensorFlow as an array manipulation library
# A tensor is like an array
# Any function that can be done on an array can be done on a tensor
# Still a deep learning library though
# Python is a slow language so define the model in abstract terms, then run the session
#   and everything is done in the background with tensorflow and then it comes back to python stuff

import tensorflow as tf

# First thing you do is construct the graph

x1 = tf.constant(5)
x2 = tf.constant(6)

# result = x1*x2 You can do this but it isn't as efficient as the official way of doing it
# Official way
result = tf.scalar_mul(x1, x2)      # Only defines model
# If using arrays use tf.matmul

print(result)

# Begins the session, where it actually runs the graph
'''
sess = tf.Session()
print(sess.run(result))
sess.close()    # Make sure to close it
'''
# Alternatively this will automatically close it
with tf.Session() as sess:
    output = sess.run(result)   # This will be a python variable
    print(output)

print(output)
# print(sess.run(result))   This would be an error since it's outside of the session



# There's a computation graph, run the session with an optimizer based on cost function that we use
#   and that will all run and modify the weights to where we don't have to modify the weights, TF will
#   do that for us automatically. TF is magic.


# Think of TF programs as 2 parts
#   1. Build the computation graph
#   2. Build what's supposed to happen in the session