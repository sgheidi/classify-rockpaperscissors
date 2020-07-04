import numpy as np
import keras
import tensorflow

from keras.models import load_model
from keras_preprocessing import image
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

pic = r"test.png"

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
  model = load_model(
    r"model/rps.h5",
    custom_objects=None,
    compile=False
)


img = image.load_img(pic, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread(pic)


if classes[0][0]:
  plt.title("Paper")
elif classes[0][1]:
  plt.title("Rock")
elif classes[0][2]:
  plt.title("Scissor")
plt.axis('off')
imgplot = plt.imshow(img)
plt.show()
