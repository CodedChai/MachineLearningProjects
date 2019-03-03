import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "/home/codedchai/Datasets/PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 150 # Choose a size that is small enough to also help processing time and be generic but big enough that it actually has data

#for category in CATEGORIES:
 #   path = os.path.join(DATADIR, category) # path to cats or dog dir
 #   for img in os.listdir(path):
#        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
   #     IMG_SIZE = 150 # Choose a size that is small enough to also help processing time and be generic but big enough that it actually has data
  #      new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # resize the images so they are all the same size. This will help since images are different sizes
    #    plt.imshow(new_array, cmap='gray')
        #plt.show()

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to cats or dog dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # resize the images so they are all the same size. This will help since images are different sizes
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

print(len(training_data))

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # This one should be a 3 for color

# Save data set to prevent us from having to reprocess it each time
# Save features
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

# Save label
pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# Load in features
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

#Load in label
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

