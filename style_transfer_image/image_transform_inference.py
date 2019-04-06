import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, UpSampling2D

from transform_networks import init_transform_network


image_dim = 512

transform_model = init_transform_network()

transform_model.load_weights('./models/sketch_1_512_3.h5')


style_image = cv2.imread('../images/style_images/sketch.jpg').astype(np.float32) / 255
style_image_resized = cv2.resize(style_image, (image_dim, image_dim))

content_image = cv2.imread('../images/content_images/european_building.jpg').astype(np.float32) / 255
content_image_resized = cv2.resize(content_image, (image_dim, image_dim))

combined_image = transform_model.predict(np.expand_dims(content_image_resized, axis=0))

fig, ax = plt.subplots(1, 3)
ax[0].imshow(cv2.cvtColor(combined_image[0], cv2.COLOR_BGR2RGB))
ax[1].imshow(cv2.cvtColor(content_image_resized, cv2.COLOR_BGR2RGB))
ax[2].imshow(cv2.cvtColor(style_image_resized, cv2.COLOR_BGR2RGB))
plt.show()