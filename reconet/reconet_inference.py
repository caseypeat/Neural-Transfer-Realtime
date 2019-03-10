import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, UpSampling2D

from reconet import init_reconet_model


image_dim = 512

reconet_model = init_reconet_model()

reconet_model.load_weights('./models/starry_night_1_eb_2.h5')



content_image = cv2.imread('../images/content_images/european_building.jpg').astype(np.float32) / 255
content_image_resized = cv2.resize(content_image, (image_dim, image_dim))
combined_image = reconet_model.predict(np.expand_dims(content_image_resized, axis=0))

fig, ax = plt.subplots(1, 1)
ax.imshow((combined_image[0]))
#ax[1].imshow(content_image_resized)
plt.show()