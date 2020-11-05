"""
This file contains the methods for merging and tiling the images.
"""
import tensorflow as tf
import image_slicer
from image_slicer import join
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

DATA_PATH = "temp_directory"

def get_tiles_images(image,DATA_PATH):
    """
    This function creates tiles from an input image

    inputs:
    image: image filename
    path:  directory where the folder is located

    outputs:
    tiles: image_slicer tiles object containing the image tiles
    """
    tiles = image_slicer.slice(os.path.join(DATA_PATH,image), 16, save=False)
    return tiles




def get_tiles_array(mip_image,DATA_PATH):
    """
     This function to get the image tiles in numpy array format

     inputs:
     image: image filename
     path:  directory where the folder is located

     outputs:
     tiles: numpy stack containing the image tiles
     """

    tiles = get_tiles_images(mip_image,DATA_PATH)
    tile_list = []

    for tile in tiles:
        tile_array = np.array(tile.image)
        tile_list.append(tile_array)

    return tile_list

def create_tiles_in_tensorflow(input_image):
    """
     This function adapts the "get_tiles_array" method to be used during training on tensorflow

     inputs:
     image: image tensor

     outputs:
     tiles: numpy stack containing the image tiles
     """

    tf.keras.preprocessing.image.save_img(os.path.join(DATA_PATH,"temp.jpg"), x=input_image[0])
    tiles = get_tiles_array("temp.jpg", DATA_PATH)

    return tiles



"""
## Test call
data_path ='rC:\ Users\bcabg\Documents\Adipocyte Cell Imaging Challenge\CODE-LAB6\pix2pix\images_for_preview\40x images\input'
mip_images = sorted([os.path.basename(x) for x in glob.glob(r'C:\ Users\bcabg\Documents\Adipocyte Cell Imaging Challenge\CODE-LAB6\pix2pix\images_for_preview\40x images\input\*.tif')])
for i in range (len(mip_images)):
    mip_image = mip_images[i]
    tiles = get_tiles_array(mip_image,data_path)

    for i in range(len(tiles)):
        plt.imshow(tiles[i])
        plt.show()
"""