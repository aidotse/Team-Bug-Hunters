#!/usr/bin/env python
# coding: utf-8
# %%
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from utils import SSIM, PSNR, augment_in_tensorflow
import matplotlib.pyplot as plt
import params
import time
from tensorflow.keras.layers import Layer, InputSpec
import numpy as np
import imageio
import multiprocess
from joblib import Parallel, delayed
from merge_and_tile import create_tiles_in_tensorflow, merge_tile_perimage
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import imageAugmentationTiles


# %%
#physical_devices

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')

try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])
except RuntimeError as e:
    print(e)


def get_tiles(input,downampling_factor,return_array = True):
    """
    Args:
        input: np.array of np.arrays containing the full resolution images
        downampling_factor: factor of the initial tile shape to downsample the tiles
        return_array: boolean to state if a np.array or a list of arrays should be returned

    Returns:
        np.array or list of all the downsampled tiles of the input array

    """
    tiles_list = []

    for i in range(input.shape[0]):
        tiles = create_tiles_in_tensorflow(input[i, :, :, 0])
        tile_shape = [(tiles.shape[1]) // params.downsample_factor, (tiles.shape[2]) // params.downsample_factor]
        for t in range(tiles.shape[0]):
            tile = tf.image.resize(tf.expand_dims(tiles[t, :, :], axis=-1), tile_shape, name="Resize tiles")
            tiles_list.append(np.expand_dims(tile[:, :, 0],-1))

    if return_array:
        return np.stack(tiles_list, axis=0)

    return tiles_list

def ParallelPathAccess(path, folder):
    """
    Access inputs and targets in parallel

    Params:
        - path : str
            Parent folder
        - folder : str
            Individual image folder to be accessed

    Returns:
        - (x_path, y_path) : tuple
            Corresponding paths for inputs and targets

    """
    fluor_tag = "A0" + str(params.fluor_ch) + "Z01C0" + str(params.fluor_ch) + ".tif"
    mip_tag = "A04Z01C04.tif"
    fluor_file = folder + fluor_tag
    mip_file = folder + mip_tag

    x_path = os.path.join(path, folder, "MIP", mip_file)
    y_path = os.path.join(path, folder, "Fluor", fluor_file)

    return (x_path, y_path)

def getInputTargetPaths(path):
    """
    From a parent path, get all paths for all inputs and all targets

    Params:
        - path : str
            Parent folder

    Outputs:
        - x_paths : list
            Set of input paths
        - y_paths : list
            Set of fluorescence paths

    """
    folders = sorted(os.listdir(path))
    parallel_out = Parallel(multiprocess.cpu_count())(delayed(ParallelPathAccess)(path, folder) for folder in folders)

    # Parallel out is a list of parallelized tuples: [(x1,y1),(x2,y2),...,(xN,y)]
    parallel_out = np.array(parallel_out)

    x_paths = parallel_out[:, 0]
    y_paths = parallel_out[:, 1]

    return x_paths, y_paths

def load_images_from_folder(input_paths, target_paths):
    """
    Load images for training in TensorFlow

    Params:
        folder : str
            Parent folder where images are located)
        input_paths : list of str
            MIP paths
        target_paths : list of str
            Validation paths

    Returns:
        x_image_array : np.ndarray
            Loaded input images
        y_image_array : np.ndarray
            Loaded target images

    """

    x_images = [imageio.imread(input_path) for input_path in input_paths]
    y_images = [imageio.imread(target_path) for target_path in target_paths]

    x_image_array = np.stack(x_images, axis=0)
    x_image_array = np.expand_dims(x_image_array, axis=-1)

    y_image_array = np.stack(y_images, axis=0)
    y_image_array = np.expand_dims(y_image_array, axis=-1)

    return x_image_array, y_image_array


def upsample_tiles(arr):
    """

    Args:
        arr: tf.tensor or np.array containing the tiles to upsample

    Returns: array containing the upsampled arrays

    """
    upsampled_tiles = []

    tile_shape = [(arr.shape[1]) * params.downsample_factor,
                  (arr.shape[2]) * params.downsample_factor]

    for t in range(arr.shape[0]):
        tile_sample = tf.image.resize(tf.expand_dims(arr[t, :, :], axis=-1), tile_shape, name="Resize tiles")
        upsampled_tiles.append(tile_sample[:, :, 0])

    return np.stack(upsampled_tiles, axis=0)

