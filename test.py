import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


model = tf.keras.models.load_model('TrainedSet.h5')

path = input("Test image path: ")
img = image.load_img(path, target_size=(128, 128))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images)
print("################################################################################")
print(int(classes[0][-1]))
print("################################################################################")
if classes[0]<0.5:
    print("Given image contains a sample 1")
else:
    print("Given image contains a sample 2")