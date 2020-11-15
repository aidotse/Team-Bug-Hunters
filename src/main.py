# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from utils import SSIM, PSNR, augment_in_tensorflow
import matplotlib.pyplot as plt
import params
from data_loading import *
from model import *
import pandas as pd


# +
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def main():

    # Data samples input path
    PATH = os.path.join("./preprocessed_data", str(params.magnification) + "x_images") # This varies depending on the magnification we want to train on

    # Split into train, test and validation images for the specific magnification and fluorescence channel
    input_paths, target_paths = getInputTargetPaths(PATH) # Input and target paths
    
    # Initial splitting of test set
    train_input_paths, test_input_paths, train_target_paths, test_target_paths = train_test_split(input_paths, target_paths, test_size = params.test_size, random_state = 42)
    # Splitting into training and validation for the remaining data
    train_input_paths, val_input_paths, train_target_paths, val_target_paths = train_test_split(train_input_paths,
                                                                                                train_target_paths,
                                                                                                test_size=params.val_size,
                                                                                                random_state=42)


    # Load images

    # Train
    x_train, y_train = load_images_from_folder(train_input_paths, train_target_paths)
    x_train_array = get_tiles(x_train, params.downsample_factor)
    y_train_array = get_tiles(y_train, params.downsample_factor)

    # Validation
    x_val, y_val = load_images_from_folder(val_input_paths, val_target_paths)
    x_val_array = get_tiles(x_val, params.downsample_factor)
    y_val_array = get_tiles(y_val, params.downsample_factor)

    # Test
    x_test, y_test = load_images_from_folder(test_input_paths, test_target_paths)
    x_test_array = get_tiles(x_test, params.downsample_factor)
    y_test_array = get_tiles(y_test, params.downsample_factor)
    
    print(type(x_test_array), x_test_array.shape)


    # Call the generator
    generator = AttentionResUNet()
    generator.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=params.learning_rate, beta_1=params.beta1, name='Adam'),
              loss=tf.keras.losses.MeanSquaredError(),metrics=[SSIM, PSNR])

    
    
    data_gen_args = dict(horizontal_flip=True, vertical_flip = True)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(x_test_array, augment=True, seed=seed)
    mask_datagen.fit(y_test_array, augment=True, seed=seed)

    image_generator = image_datagen.flow(x_train_array,seed=seed, batch_size = params.batch_size)
    mask_generator = mask_datagen.flow(y_train_array,seed=seed, batch_size = params.batch_size)

    train_generator = zip(image_generator, mask_generator)
    
    # Checkpoints
    if not(os.path.exists("./traning_checkpoints")):
        os.makedirs("./traning_checkpoints")
    cpt_file = os.path.join("./training_checkpoints","checkpoint_attUNet" + str(params.magnification) 
                                   + "_fluor" + str(params.fluor_ch))
    #cpt_file = os.path.join("./training_checkpoints","ch" + str(params.fluor_ch) + "_magn" + str(params.magnification) + ".hdf5")
    
    # Early stopping
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=20, verbose=1, mode='auto')
    
    # Learning rate scheduler
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cpt_file,save_weights_only=False, monitor='val_SSIM', 
                                                     mode='max',save_best_only=True, verbose=1)
    
    
    if os.path.exists(cpt_file):

        generator = tf.keras.models.load_model(cpt_file, custom_objects={'SSIM':SSIM, 'PSNR': PSNR})
        

        results = generator.fit(train_generator,verbose = 1, 
                            steps_per_epoch=x_train_array.shape[0] // params.batch_size, 
                            validation_data = (x_val_array,y_val_array), epochs=params.epochs, 
                            callbacks=[cp_callback, lr_callback, early_stopper], validation_batch_size = params.batch_size) 
    else:
        results = generator.fit(train_generator,verbose = 1, 
                            steps_per_epoch=x_train_array.shape[0] // params.batch_size, 
                            validation_data = (x_val_array,y_val_array), epochs=params.epochs, 
                            callbacks=[cp_callback,lr_callback,early_stopper], validation_batch_size = params.batch_size) 

    
    # Get prediction for the first test sample
    prediction = generator.predict(x=x_test_array,verbose=1, batch_size = params.batch_size)
    
    #Learning and loss curves
    plt.plot(results.history['SSIM'])
    plt.plot(results.history['val_SSIM'])
    plt.title('SSIM')
    plt.ylabel('SSIM')
    plt.xlabel('Epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.grid()
    plt.show()

    # summarize history for accuracy
    plt.plot(results.history['PSNR'])
    plt.plot(results.history['val_PSNR'])
    plt.title('PSNR')
    plt.ylabel('PSNR')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    plt.show()
    
    # summarize history for accuracy
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('Loss Curve')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    plt.show()
    
    # Comparison figure of the input, prediction and target test images
    for i in range (3):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(x_test_array[i,:,:], cmap="gray")
        plt.title("Input")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(prediction[i,:,:,0], cmap="gray")
        plt.title("Prediction")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(y_test_array[i,:,:], cmap="gray")
        plt.title("Target")
        plt.axis('off')
        plt.show()
        
    # Merge tiles back
    merged_prediction = merge_tile_perimage(prediction[:16, :, :, 0])
    merged_input = merge_tile_perimage(x_test_array[:16, :, :, 0])
    merged_target = merge_tile_perimage(y_test_array[:16, :, :, 0])


    if not os.path.exists("log"):
        os.makedirs("log")
    
    # Save images
    image_name = "history_ch{}_mag_{}_lr{}_downf_{}_attUNet.tiff".format(params.fluor_ch, params.magnification, params.learning_rate, params.downsample_factor)
    tf.keras.preprocessing.image.save_img(
        os.path.join("log","pred_"+image_name), np.expand_dims(merged_prediction[0],axis=-1))

    tf.keras.preprocessing.image.save_img(
        os.path.join("log", "input_" + image_name), np.expand_dims(merged_input[0],axis=-1))

    tf.keras.preprocessing.image.save_img(
        os.path.join("log", "target_" + image_name), np.expand_dims(merged_target[0],axis=-1))


    log_file_name = "history_ch{}_mag_{}_lr{}_downf_{}_attUNetRetrained.csv".format(params.fluor_ch, params.magnification, params.learning_rate, params.downsample_factor)
    pd.DataFrame(results.history).to_csv(os.path.join("log",log_file_name))
    
    # Save model as hdf5 file
    # File where to save model:
    saved_model_file = os.path.join("./training_checkpoints","ch" + str(params.fluor_ch) + "_magn" + str(params.magnification) + ".hdf5")
    tf.keras.models.save_model(generator, saved_model_file)
    
if __name__ == "__main__":
    main()
# -

