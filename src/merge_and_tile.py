"""
This file contains the methods for merging and tiling the images.
"""
import tensorflow as tf
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import params as params
from PIL import Image


@tf.function
def tiling(img):
    """
    Tile input image with a given tile size

    Params:
        - img : np.ndarray
            Image to be tiled

    Outputs:
        - tiles : list of Tensorflow tensors
            List with tiles

    """
    tiles = []

    for i in range(int(math.ceil(img.shape[0]/(params.tile_size * 1.0)))):
        for j in range(int(math.ceil(img.shape[1]/(params.tile_size * 1.0)))):
            cropped_img = img[params.tile_size*i:min(params.tile_size*i+params.tile_size, img.shape[0]), params.tile_size*j:min(params.tile_size*j+params.tile_size, img.shape[1])]
            tiles.append(tf.cast(cropped_img, tf.uint16))

    tiles = tf.stack(tiles, axis=0)

    return tiles




@tf.function
def create_tiles_in_tensorflow(input_image):
    """
     This function adapts the "get_tiles_array" method to be used during training on tensorflow

     inputs:
     image: image tensor

     outputs:
     tiles: numpy stack containing the image tiles
     """

    tiles = tiling(input_image)

    return tiles



def merge_tile_perimage(X):
    """
    This function merges the predicted tiles back to whole image of dimension
    2156*2256 and saves them in tiff format for evaluation"
    """

    # For 512,512
    arr1 = np.concatenate((X[0:4]), axis=1)
    arr2 = np.concatenate((X[4:8]), axis=1)
    arr3 = np.concatenate((X[8:12]), axis=1)
    arr4 = np.concatenate((X[12:]), axis=1)
    a = np.vstack((arr1, arr2, arr3, arr4))
    
    # For 256,256
    """
    arrs = []
    for n,i in enumerate(range(8)):
        if i < 7:
            arr = np.concatenate((X[i*8: (i + 1)*8]), axis = 1) 
        else:
            arr = np.concatenate((X[i*8:]), axis = 1)
        arrs.append(arr)
    a = np.vstack(tuple(arrs))
    """
 

    return tf.expand_dims(a,axis=0)
