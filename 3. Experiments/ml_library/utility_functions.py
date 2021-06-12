import cv2
import numpy as np
from keras import applications

def pre_process_image_for_vgg(img,input_size):
    """
        Resizes the image to input of VGGInputSize specified in the config dictionary
        Normalises the image
        Reshapes the image to an array of images e.g. [[img],[img],..]

        Inputs:
            img: a numpy array or array of numpy arrays that represent an image
            input_size: a tuple of (width, height )
    """
    if type(img) == np.ndarray: # Single image 
        resized_img = cv2.resize(img,input_size,interpolation = cv2.INTER_AREA)
        normalised_image = applications.vgg16.preprocess_input(resized_img)
        reshaped_to_array_of_images = np.array([normalised_image])
        return reshaped_to_array_of_images
    elif type(img) == list: # list of images
        img_list = img
        resized_img_list = [cv2.resize(image,input_size,interpolation = cv2.INTER_AREA) for image in img_list]
        resized_img_array = np.array(resized_img_list)
        normalised_images_array = applications.vgg16.preprocess_input(resized_img_array)
        return normalised_images_array