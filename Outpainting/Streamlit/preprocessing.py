import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image




def load_image(path):
    image = Image.open(path)
    return image


def resize(image):
    image = image.convert('RGB')
    width, height = image.size
    if width == height:
        if width ==256:
            return image
        elif width <256:
            image = image.resize((256,256))
            return image
        elif width >256:
            image = image.resize((256,256))
            return image
    elif width > height:
        rescale_size = height/256
        width_rescaled = width/rescale_size
        image_scaled = image.resize((int(width_rescaled),256))
        return image_scaled
    elif height > width:
        return 'Only accept landscape or sqaure images'

def convert_to_np(image):
    image_np = np.asarray(image)
    if image_np.shape[0] == 256:
        return image_np
    else:
        return 'Error with re-shaping'
    return image_np

def left_right_expand(image,expand_side):
    if expand_side == 'left':
        image_sliced = image[:,:192,:]
    elif expand_side == 'right':
        width_ = image.shape[1] - 192
        image_sliced = image[:,width_:,:]
    return image_sliced

def normalize(image):
    image = image/255
    return image


def masked_image(image,expand_side):
    mask = np.full((256,64,3),(0))
    if expand_side == 'left':
        image_out = np.concatenate((mask,image),axis=1)
    if expand_side == 'right':
        image_out = np.concatenate((image,mask),axis=1)
    return image_out


def flip_reshape(image,expand_side):
    if expand_side == 'left':
        image = np.fliplr(image)

    image_to_predict = image.reshape((-1,256,256,3))
    return image_to_predict

def preprocess(path,expand_side):
    '''image_jpg needs to be in the form of a jpg path, limited to landscape and square images'''

    #load image of either landscape or square shape, image is loaded in using Pillow
    image = load_image(path)

    #then resize image to be height 256
    image = resize(image)

    #image is in Pillow object, convert to an np array
    image_np = convert_to_np(image)

    #slicing the image to be the 192 pixel section, if we want a left expand this takes the left side, if we want a right expand this takes the right side
    image_sliced = left_right_expand(image_np,expand_side)

    #normalize image to be in range from 0 to 1 instead of 0 to 255
    image_norm = normalize(image_sliced)

    #mask image so we create a black mask of 64 pixels wide that our model will predict on
    image_masked = masked_image(image_norm,expand_side)

    #the model we have built is train on right masks, so if the 'left' option has been selected we need to flip this image horizantally
    #we then also need to add a 4th dimension to the image so it runs in the model, will now be shape(1,256,256,3)
    image_ready = flip_reshape(image_masked,expand_side)

    #shape of image should now be (1,256,256,3) and ready to predict on
    return image_ready, image
