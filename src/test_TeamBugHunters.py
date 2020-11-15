# +
import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import getopt
import os
import sys
from sklearn.model_selection import train_test_split
import time
import numpy as np
import imageio
import multiprocess
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd
import skimage.transform
import glob 
import cv2

# Import code from other .py files (to be located in the same directory as this file)
from utils import SSIM, PSNR
import params
from merge_and_tile import create_tiles_in_tensorflow, merge_tile_perimage
from utils import imageAugmentationTiles


def main(argv):
    
    """
    File testing: read the test dataset with brightfield images to output the correspondent fluorescent predictions. The 
    code is fit to run the following folder structure:
        - Parent folder:
            - Magnification type 1 (20x_images)
                - Set of brightfield images, separated into different Z slices
            - Magnification type 2 (40x_images)
                ...
                        
            And so on
            
    The model files used in the prediction SHOULD BE LOCATED IN THE SAME FOLDER AS THIS CODE
            
    The corresponding predictions will be saved in a folder that can be inputted by the user. If no output directory is 
    inputted by the user, the folder with the predictions will be created in the same directory as this code with the
    name "predictions_TeamBugHunters". The structure of the output folder will be the same as the structure of the input folder, 
    but with the prediction files (output filenames will be "AssayPlate_Greiner_#655090_xxx_TxxxFxxxLxxA0cZ01C0c.tif",
    with "_xxx_TxxxFxxxLxx" given by each set of Z stack and "c" given by the fluorescence channel that is analyzed)
    
            
    Params:
        - path : str
            Name of input test folder
        - out : str (optional)
            Name of output folder with results
            
    Outputs:
        void
        Saved prediction files as .tiff format in the output folder

    """
    init = time.time()
    
    path = ""
    out = ""
    
    # Configuration for TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    try:
        #tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

    
    # Read given arguments
    try:
        opts, args = getopt.getopt(argv, "hi:o:r", ["in=","out="])
    except getopt.GetoptError:
        print("test_TeamBugHunters.py -i <input_path> -o <output_path>")
        sys.exit(2)
       
    for opt, arg in opts:
        if opt == '-h':
            print("test_TeamBugHunters.py -i <input_path> -o <output_path>")
            sys.exit()
        elif opt in ("-i", "--in"):
            path = arg
        elif opt in ("-o", "--out"):
            out = arg
            
    if path == "":
        print("Input path has not been provided. Please provide an input path")
        sys.exit(2)
        
    if not(os.path.exists(path)) or not(os.path.isdir(path)):
        print("Input path {} does not exist or is not a directory. Please provide a valid input path".format(path))
        sys.exit(2)
        
    if out == "":
        print("No output folder provided. Outputting results to a folder called '{}' in the current working directory".format("predictions_TeamBugHunters"))
        out = os.path.join(os.getcwd(),"predictions_TeamBugHunters")
        os.makedirs(out)
    elif not(os.path.exists(out)):
        print("Output folder '{}' does not exist. Creating it...".format(out))
        os.makedirs(out)
        
    print("Loading data from '{}'...".format(path))
    
    files, magnifications_used = dataAccess(path) # List of files found sorted by magnification and grouped under the same Z stack
    
    if len(files) == 0:
        print("No .tif files found in {}. Please provide a correct folder".format(path))
        sys.exit(2)
        
    magnification_times = [] # Store total inference time spent per magnification setting
    magnification_times_per_stack = [] # Store total inference time spent per magnification setting and per evaluated stack
    
    for magnification,magnification_files in zip(magnifications_used,files): # Accessing the set of files of each magnification (first 20x, then 40x and at last 60x)
        print("Start inference for {}x magnification images...".format(magnification))
        
        out_subfolder = os.path.join(out,str(magnification) + "x_images") # Subfolder where to output prediction images from the same magnification

        if not os.path.exists(out_subfolder):
            os.makedirs(out_subfolder)

        time_counter_inference = 0 # Accumulate inference times for all stacks per magnification here

        # Load models for the magnification in use for the three output fluorescent channels
        models = []
        channel_tags = []
        for channel in range(1,4): # Store models as [model_ch1, model_ch2, model_ch3]
            channel_magn_tag = "ch" + str(channel) + "_magn" + str(magnification)

            try:
                models.append(tf.keras.models.load_model(channel_magn_tag + ".hdf5", custom_objects={'loss':tf.keras.losses.MeanSquaredError(),'SSIM':SSIM, 'PSNR': PSNR}))
                channel_tags.append(channel_magn_tag)
            except:
                print("There was a problem loading model '{}'. Inference skipped for magnification {} and fluorescence channel {}".format(channel_magn_tag + ".hdf5", magnification, channel))


        for n,stack_files in enumerate(magnification_files): # Access sets of files belonging to the same stack
            stack = []
            skip = 0

            # Preprocessing
            for s, img_file in enumerate(stack_files): # Access each individual image of the stack
                try:
                    img = imageio.imread(img_file)
                    stack.append(img)
                except:
                    print("Brightfield image '{}' could not be read. This stack will be skipped".format(img_file))
                    skip = 1

            if skip == 1:
                continue


            image_file = os.path.basename(os.path.normpath(img_file))
            file_id = image_file.replace("A04Z07C04.tif","")
            
            # MIP processing
            stack_array = np.stack(stack) # 3D array with all Z slices
            mip = np.max(stack_array,0) # Maximum intensity projection
            
            original_size = mip.shape # Save shape for later resizing results to original MIP shape
            resized_mip = skimage.transform.resize(mip, (2048,2048), order=0, preserve_range=True) # Resize MIP to 2048,2048 size to adapt it to the network Tensor size

            # Tile obtention
            tiles = get_tiles(resized_mip) # Stack of 2D arrays
            
            # Inference
            for channel_tag, model in zip(channel_tags, models): 

                # Inference + chronometer
                try:
                    t1 = time.time()
                    output = model.predict(x=tiles, batch_size = params.batch_size) # Get prediction for a given fluorescence channel
                    t2 = time.time()

                    time_counter_inference += t2 - t1 # Accumulator for inference time

                    # Post-processing
                    merged_prediction = merge_tile_perimage(output[:, :, :, 0]) # Merge predicted tiles
                    merged_prediction = merged_prediction.numpy()

                    if merged_prediction.min() < 0:
                        merged_prediction[merged_prediction < 0] = 0 # Clip merged predictions to zero, in case of negative numbers

                    resized_prediction = skimage.transform.resize(merged_prediction[0], original_size, order = 0, preserve_range=True) # Resize prediction to original shape
                    resized_prediction = resized_prediction.astype(np.uint16)

                    # Output file processing
                    output_name = "{}A0{}Z01C0{}.tif".format(file_id,channel_tag[2],channel_tag[2])

                    cv2.imwrite(os.path.join(out_subfolder, output_name), np.expand_dims(resized_prediction,axis=-1).astype(np.uint16))

                    print("Saved fluorescence prediction for image '{}' and for model '{}' in '{}'".format(file_id, channel_tag, out_subfolder))

                except:
                    print("There was some problem when trying to evaluate image file '{}' with model '{}'".format(file_id, channel_tag))


        # Final information on computational time
        time_per_stack = time_counter_inference/(n + 1)
        magnification_times.append(time_counter_inference)
        magnification_times_per_stack.append(time_per_stack)

        print("All images from {}x magnification have been predicted".format(magnification))
        
    # Final information on computational time
    total_inference_time = 0
    total_files = 0
    for magnification, magnification_time, magnification_time_per_stack in zip(magnifications_used, magnification_times, magnification_times_per_stack):
        print("Accumulated inference time for {}x magnification images: {} sec ({} sec/image stack)".format(magnification, round(magnification_time,4), magnification_time_per_stack))
        total_inference_time += magnification_time
        total_files += magnification_time//(magnification_time_per_stack + np.finfo(float).eps)
        
    if len(magnifications_used) > 1: # Print total inference time spent for all magnifications (if there is more than one)
        print("Total inference time for all magnifications: {} sec ({} sec/image stack)".format(round(total_inference_time,4), total_inference_time/total_files))      
        
        
# Helper functions for loading the data and the models

    
def dataAccess(path):
    """
    Access image data, and load it
    
    Params:
        path : str
            Parent folder where to start looking for the images
    
    Outputs:
        files: list of str
            Set of files grouped as: [[[file1_Z01, file2_Z02,...],[file2...], fileN],...]
        magnifications_used : list of int
            Different magnifications used
        file_ids : list of str
            Read file IDs
    
    """
    # Getting all .tif files in the given input folder
    tif_files = np.array(sorted(glob.glob(os.path.join(path, "**","*.tif"), recursive = True)))
    magnification_tags = ["20x","40x","60x"]
    
    files = [] # Storage of organized filepaths
    file_ids = [] # Storage of file IDs
    magnifications_used = [] # Storage of present magnifications in test dataset 
    
    # Get tif_files under the same magnification
    for magnification_tag in magnification_tags:
        magnification = int(magnification_tag[:(-1)])
        magnification_files_list = []
        ind = np.flatnonzero(np.core.defchararray.find(tif_files,magnification_tag)!=-1) # Index of .tif files from a certain magnification
        if len(ind) == 0: # In case magnifications are tested separately
            continue
        magnification_files = tif_files[ind] # .tif files from the same magnification
        brightfield_ind = np.flatnonzero(np.core.defchararray.find(magnification_files, "C04.tif")!=-1) # Index of .tif files corresponding to brightfield images
        brightfield_files = magnification_files[brightfield_ind] # Brightfield files from the same magnification
        ind_img = np.flatnonzero(np.core.defchararray.find(brightfield_files,"A04Z01C04.tif")!=-1) # Index of individual image files (Z stack) inside each magnification
        for i in ind_img: # Go through all images found
            file = os.path.basename(os.path.normpath(brightfield_files[i]))
            fileID = file.replace("A04Z01C04.tif","") # Common identifier from all brightfield images belonging to the same Z stack
            ind_stack = np.flatnonzero(np.core.defchararray.find(brightfield_files,fileID)!=-1)
            stack_files = brightfield_files[ind_stack] # Filepaths from the same stack
            magnification_files_list.append(stack_files)
    
        files.append(magnification_files_list)
        magnifications_used.append(magnification)

    return files, magnifications_used


def get_tiles(input,return_array = True):
    """
    Args:
        input: np.array of np.arrays containing the full resolution images
        downampling_factor: factor of the initial tile shape to downsample the tiles
        return_array: boolean to state if a np.array or a list of arrays should be returned

    Returns:
        np.array or list of all the downsampled tiles of the input array

    """
    tiles_list = []

    tiles = create_tiles_in_tensorflow(input)
    tile_shape = [(tiles.shape[1]) // params.downsample_factor, (tiles.shape[2]) // params.downsample_factor]
    for t in range(tiles.shape[0]):
        tile = tf.image.resize(tf.expand_dims(tiles[t, :, :], axis=-1), tile_shape, name="Resize tiles")
        tiles_list.append(np.expand_dims(tile[:, :, 0],-1))

    if return_array:
        return np.stack(tiles_list, axis=0)

    return tiles_list


def bn_act(x, act=True):
    x = tf.keras.layers.BatchNormalization()(x)
    if act == True:
        x = tf.keras.layers.LeakyReLU()(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    #conv = ReflectionPadding2D(padding=(1,1))(conv)
    #conv = tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0]], 'REFLECT'))(conv)
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    #x_pad = ReflectionPadding2D(padding=(1,1))(x)
    #x_pad = tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0]], 'REFLECT'))(x)
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
 
    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = tf.keras.layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = tf.keras.layers.Add()([shortcut, res])
    return output

def upsample_block(x):
    u = tf.keras.layers.UpSampling2D((2, 2))(x)
    return u
    
def concat_block(x, xskip):
    c = tf.keras.layers.Concatenate()([x, xskip])
    return c
    
def attention(tensor, att_tensor, n_filters=512, kernel_size=[1, 1]):
    g1 = tf.keras.layers.Conv2D(n_filters, kernel_size)(tensor)
    x1 = tf.keras.layers.Conv2D(n_filters, kernel_size)(att_tensor)
    net = tf.keras.layers.Add()([g1, x1])
    net = tf.keras.layers.ReLU()(net)
    net = tf.keras.layers.Conv2D(1, kernel_size)(net)
    net = tf.nn.sigmoid(net)
    #net = tf.concat([att_tensor, net], axis=-1)
    net = net * att_tensor
    return net


def AttentionResUNet():
    f = [64, 128, 256, 512, 512]
    inputs = tf.keras.layers.Input((params.tile_size//params.downsample_factor, params.tile_size//params.downsample_factor, 1))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    
    ## Decoder
    u1 = upsample_block(b1)
    att1 = attention(e4, u1, f[4])
    c1 = concat_block(u1, att1)
    d1 = residual_block(c1, f[4])
    
    u2 = upsample_block(d1)
    att2 = attention(e3, u2, f[3])
    c2 = concat_block(u2, att2)
    d2 = residual_block(c2, f[3]) 
    
    u3 = upsample_block(d2)
    att3 = attention(e2, u3, f[2])
    c3 = concat_block(u3, att3)
    d3 = residual_block(c3, f[2]) 
    
    u4 = upsample_block(d3)
    att4 = attention(e1, u4, f[1])
    c4 = concat_block(u4, att4)
    d4 = residual_block(c4, f[1]) 

    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same")(d4)
    outputs = tf.keras.layers.LeakyReLU()(outputs)
    model = tf.keras.Model(inputs, outputs)
    return model

if __name__ == '__main__':
    main(sys.argv[1:]) 
# -


