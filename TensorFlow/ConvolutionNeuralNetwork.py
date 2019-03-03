import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time


# If you want to limit the GPU mem usage then use this. This is useful for running multiple models at the same time or ensuring your pc doesn't die when being used
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# We're loading in the data from "LoadingInOutsideDataset.py"
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0 # Scaling image data from 0-255 to 0-1

dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
            print(NAME)
            model = Sequential()
            model.add(Conv2D(layer_size, (4,4), input_shape=X.shape[1:])) # Add the type of layer, the window size and then the shape. Remember that the shape is X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1), we want to skip the first value as it is always -1 and was just used to determine how many feature sets there were
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(3,3)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (4,4))) # input shape is already defined, don't need to add a new one
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(3,3)))

            model.add(Flatten()) # Make the 2D data 1D

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))

            model.add(Dense(1)) # Output layer
            model.add(Activation("sigmoid"))

            model.compile(loss="binary_crossentropy",
                            optimizer="adam",
                            metrics=['accuracy'])


            model.fit(X, y, batch_size = 32, epochs=10, validation_split=0.3, callbacks=[tensorboard])
            model.save("{}.model".format(NAME))
