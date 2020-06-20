import numpy as np
import keras
import tensorflow
from keras.models import load_model
from keras_preprocessing import image

import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

pic = "rock.png"

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
  model = load_model(
    "rps.h5",
    custom_objects=None,
    compile=False
)


img = image.load_img(pic, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
if classes[0][0]:
  print("Paper")
elif classes[0][1]:
  print("Rock")
else:
  print("Scissor")
