import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import imageio

from tqdm import tqdm

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, UpSampling2D

from transform_networks import init_transform_network



transform_model = init_transform_network()

transform_model.load_weights('./models/a_muse_1_256_3_reflect.h5')

content_video_filepath = '../images/content_videos/SUNP0025.AVI'
combined_video_filepath = '../images/combined_videos/SUNP0025_a_muse_1_256_3_reflect.AVI'


reader = imageio.get_reader(content_video_filepath)
writer = imageio.get_writer(combined_video_filepath, fps=reader.get_meta_data()['fps'])

for frame in tqdm(range(reader.get_length())):

	image = reader.get_data(frame).astype(np.float32) / 255

	stylized_image = cv2.cvtColor(transform_model.predict(np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0))[0], cv2.COLOR_BGR2RGB)
	stylized_image_rescaled = (stylized_image * 255).astype(np.uint8)
	stylized_image_resized = stylized_image_rescaled[40:-40,40:-40]

	writer.append_data(stylized_image_resized)

writer.close()