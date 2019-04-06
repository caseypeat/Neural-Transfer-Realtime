import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import imageio

from tqdm import tqdm


base_image_dirpath = '../images/mountain_biking/style_images'
image_filenames = ['a_muse.jpg', 'candy.png', 'picasso.jpg', 'sketch.jpg']

base_video_dirpath = '../images/mountain_biking'
input_video_filenames = ['SUNP0025.AVI', 'SUNP0025_a_muse_3_256_3.AVI', 'SUNP0025_candy_1_256_3.AVI', 'SUNP0025_picasso_3_256_3.AVI', 'SUNP0025_sketch_3_256_3.AVI']

output_filepath = '../images/mountain_biking/SUNP0025_mixed.mp4'

writer = imageio.get_writer(output_filepath, fps=60)

readers = []
style_images = []

for i in range(5):
	readers.append(imageio.get_reader(os.path.join(base_video_dirpath, input_video_filenames[i])))

style_images.append(np.zeros((256,256,3), dtype=np.uint8))
for i in range(4):
	style_images.append(cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(base_image_dirpath, image_filenames[i])), (256,256)), cv2.COLOR_BGR2RGB))

for k in range(5):
	for i in tqdm(range(500)):

		frame = i + k*500

		image = readers[k].get_data(frame)

		image_extended = np.zeros((480, 640+256, 3), dtype=np.uint8)
		image_extended[:,:640,:] = image
		image_extended[:256,640:,:] = style_images[k]

		writer.append_data(image_extended)

writer.close()
