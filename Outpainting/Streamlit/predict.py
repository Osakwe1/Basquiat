import numpy as np
import pandas as pd
from PIL import Image


import matplotlib.pyplot as plt
import matplotlib.image as img

import tensorflow as tf

from preprocessing import preprocess
from load_model import load_model
from postprocessing import reduce_colors

path = '/Users/olisa/Downloads/IMG_0985.jpg'

expand_side = 'right'

preprocessed_image = preprocess(path,expand_side)
#print(preprocessed_image.shape)

#test = Image.fromarray(((preprocessed_image[0,:,:,:])*256).astype(np.uint8))

#test.save('/home/krishinipatel/code/krishinipatel/trial_project/mountain_test.jpg')

#this can be commented out at the end
#st.image(preprocessed_image)

#update these with compute engine paths

#model_path_dis_r = 'xxx'
model_path_gen_r = '/Users/olisa/Downloads.h5'
model_uploaded = load_model(model_path_gen=model_path_gen_r)


prediction = model_uploaded(preprocessed_image, training = True)

#tf.keras.preprocessing.image.save_img( "tensor.png", (prediction[0]))
# prediction = prediction.numpy()

# test = (prediction[0,:,:,:]*255).astype(np.uint8())
# print(test.shape)
# test = Image.fromarray(test)

# test.save('/home/krishinipatel/code/krishinipatel/trial_project/mountain_predict.jpg')

#now we need to do post-processing
output = reduce_colors(prediction[0],80).astype(np.uint8)
#print(output.shape)
output_ready = Image.fromarray(output)
output_ready.save('/Users/olisa/Downloads/mountain_output.png')

#print(output_ready)
