import cv2
import tensorflow as tf

CATEGORIES = ["Dog", "Cat"]

def prepare(filepath):
    IMG_SIZE = 150
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("3-conv-64-nodes-0-dense-1551647259.model")

prediction = model.predict([prepare('dog.jpg')])
print(CATEGORIES[int(prediction[0][0])])

prediction = model.predict([prepare('cat.jpg')])
print(CATEGORIES[int(prediction[0][0])])