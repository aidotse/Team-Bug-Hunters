# Pix2Pix

import tensorflow as tf
import os
import time
import numpy as np
from merge_and_tile import create_tiles_in_tensorflow, merge_tile_perimage
from utils import imageAugmentation, augment_in_tensorflow, SSIM, PSNR
from matplotlib import pyplot as plt
from IPython import display
from tfrecords_helper_functions import image_example, parse_record, decode_record
import imageio
from sklearn.model_selection import train_test_split
import datetime
import multiprocess
from joblib import Parallel, delayed
from tensorflow.keras.layers import Layer, InputSpec
import random
import params
from PIL import Image
tf.enable_eager_execution

class ReflectionPadding2D(Layer):
  def __init__(self, padding=(1, 1), **kwargs):
    self.padding = tuple(padding)
    self.input_spec = [InputSpec(ndim=4)]
    super(ReflectionPadding2D, self).__init__(**kwargs)

  def get_output_shape_for(self, s):
    """ If you are using "channels_last" configuration"""
    return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

  def call(self, x, mask=None):
    w_pad, h_pad = self.padding
    return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


def ParallelPathAccess(path,folder):
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
    fluor_tag = "A0" + str(params.fluor_class) + "Z01C0" + str(params.fluor_class) + ".tif"
    mip_tag = "A04Z01C04.tif"
    fluor_file = folder + fluor_tag
    mip_file = folder + mip_tag
    
    x_path = os.path.join(path,folder,"MIP",mip_file)
    y_path = os.path.join(path,folder,"Fluor",fluor_file)
    
    return (x_path,y_path)



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
    
    x_paths = parallel_out[:,0]
    y_paths = parallel_out[:,1]
    
    return x_paths, y_paths


@tf.function
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
        
    x_image_array = np.stack(x_images, axis = 0)
    x_image_array = np.expand_dims(x_image_array, axis=-1)
    
    y_image_array = np.stack(y_images, axis = 0)
    y_image_array = np.expand_dims(y_image_array, axis=-1)
    
    return x_image_array, y_image_array


def downsample(filters, size, apply_batchnorm=True):
  """
  UNet downsampling layer for the generator, with a given number of filters, size, and optionally apply Dropout (default: no)
  Conv2D applied with reflection padding
  
  """
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      ReflectionPadding2D(padding=(1, 1)))
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='valid',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result



def upsample(filters, size, apply_dropout=False):
  """
  UNet upsampling layer for the generator, with a given number of filters, size, and optionally apply Dropout (default: no)
  Conv2DTranspose applied
  
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))


  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result



def Generator():
  """
  Generator architecture: UNet
  
  """  
    
  inputs = tf.keras.layers.Input(shape=[params.tile_size,params.tile_size,1])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  
    downsample(128, 4),  
    downsample(256, 4),  
    downsample(512, 4),  
    downsample(512, 4),  
    downsample(512, 4),  
    downsample(512, 4),  
    downsample(512, 4),  
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  
    upsample(512, 4, apply_dropout=True),  
    upsample(512, 4, apply_dropout=True),  
    upsample(512, 4),  
    upsample(256, 4), 
    upsample(128, 4),  
    upsample(64, 4),  
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(params.output_channels, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') 

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def generator_loss(disc_generated_output, gen_output, target):
  """
  Generator loss, given the discriminator output with the generated image, the generated image itself
  and the target: adversarial_loss + lambda*L1 + SSIM loss
  
  """
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  
  # SSIM loss
  ssim = tf.py_function(SSIM, [gen_output, target], tf.float32, name="ssim")
    
  ssim_loss = 1 - ssim
  total_gen_loss = gan_loss + (params.LAMBDA * l1_loss) + params.ssim_lambda*ssim_loss

  return total_gen_loss, gan_loss, l1_loss



def Discriminator():
  """
  Discriminator architecture: PatchGAN
  
  """
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[params.tile_size,params.tile_size, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[params.tile_size,params.tile_size, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output):
  """
  Discriminator loss, given the real output and the generated output
  
  """
    
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss






def generate_images(model, test_input, tar):
  """## Generate Images

    Write a function to plot some images during training.

    * We pass images from the test dataset to the generator.
    * The generator will then translate the input image into the output.
    * Last step is to plot the predictions and **voila!**

    Note: The `training=True` is intentional here since
    we want the batch statistics while running the model
    on the test dataset. If we use training=False, we will get
    the accumulated statistics learned from the training dataset
    (which we don't want)
  """
  prediction = model(test_input, training=False)
  plt.figure(figsize=(15,15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()


def train_step(input_image, target, epoch):
  """
  Processes taking during data training
  
  Params:
      - input_image : Tensorflow tensor
          MIP for input inside the training dataset
      - target : Tensorflow tensor
          Target for input inside the training dataset
      - epoch : int
          Epoch number

          
  Outputs:
      - gen_mean_loss : float
          Mean generator loss for validation set
      - disc_mean_loss : float
          Mean discriminator loss for validation set
      - ssim_mean : float
          Mean SSIM metric for validation set
      - psnr_mean : float
          Mean PSNR metric for validation set
  
  """

  # Augmentation
  input_image, target = augment_in_tensorflow(input_image.numpy(), target.numpy())

  # MIP tiling
  tiles = tf.py_function(
    create_tiles_in_tensorflow, [input_image], tf.uint16, name="create_tiles_image"
  )

  # Fluorescence tiling
  tiles_target = tf.py_function(
    create_tiles_in_tensorflow, [target], tf.uint16, name="create_tiles_target"
  )


  gen_tiles = [] # List of generated tiles
  cumulative_gen_loss = 0
  cumulative_disc_loss = 0
  cumulative_ssim = 0
  cumulative_psnr = 0
  for i in range(tiles.shape[0]):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape: # Gradient definition

      gen_output = generator(tf.expand_dims(tiles[i,:,:,:],axis=0),training=True) # Generated tile
      gen_tiles.append(np.squeeze(gen_output)) # Save generated tile

      disc_real_output = discriminator([tf.expand_dims(tiles[i,:,:,:], axis=0), tf.expand_dims(tiles_target[i,:,:,:], axis=0)],
                                       training=True) # Discriminator output with "real" images
      disc_generated_output = discriminator([tf.expand_dims(tiles[i,:,:,:], axis=0), tf.expand_dims(gen_output, axis=-1)], 
                                            training=True) # Discriminator output with "generated" images

      # Generator loss computation
      gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output,
                                                                 gen_output,
                                                                 tf.expand_dims(tf.cast(tiles_target[i,:,:,:],tf.float32), axis=0))
      # Discriminator loss computation
      disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

      # Metrics computation
      ssim = tf.py_function(SSIM, [gen_output, tf.expand_dims(tf.cast(tiles_target[i,:,:,:],tf.float32),0)], tf.float32, name = "ssim")
      psnr = tf.py_function(PSNR, [gen_output, tf.expand_dims(tf.cast(tiles_target[i,:,:,:],tf.float32),0)], tf.float32, name = "psnr")

    # Gradients 
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    # Application of gradients
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    
    # Accumulate losses and metrics for all tiles
    cumulative_gen_loss += gen_total_loss
    cumulative_disc_loss += disc_loss
    cumulative_ssim += ssim
    cumulative_psnr += psnr
    
  # Mean losses and metrics
  gen_mean_loss = cumulative_gen_loss/tiles.shape[0]
  disc_mean_loss = cumulative_disc_loss/tiles.shape[0]
  ssim_mean = cumulative_ssim/tiles.shape[0]
  psnr_mean = cumulative_psnr/tiles.shape[0]
    
  return gen_mean_loss, disc_mean_loss, ssim_mean, psnr_mean



def val_step(input_image, target, epoch, show = False, counter=0):
  """
  Processes taking during data validation
  
  Params:
      - input_image : Tensorflow tensor
          MIP for input inside the validation dataset
      - target : Tensorflow tensor
          Target for input inside the validation dataset
      - epoch : int
          Epoch number
      - show : bool
          Flag to specify if treated validation image is showed or not (default: False)
          
  Outputs:
      - gen_mean_loss : float
          Mean generator loss for validation set
      - ssim_mean : float
          Mean SSIM metric for validation set
      - psnr_mean : float
          Mean PSNR metric for validation set
  
  """

  # MIP tiling
  tiles = tf.py_function(
    create_tiles_in_tensorflow, [input_image], tf.uint16, name="create_tiles_image"
  )

  # Fluorescence tiling
  tiles_target = tf.py_function(
    create_tiles_in_tensorflow, [target], tf.uint16, name="create_tiles_target"
  )


  gen_tiles = [] # List of generated tiles

  cumulative_gen_loss = 0
  cumulative_ssim = 0
  cumulative_psnr = 0

  for i in range(tiles.shape[0]):

    gen_output = generator(tf.expand_dims(tiles[i,:,:,:],axis=0),training=False) # Generated tile
    gen_tiles.append(np.squeeze(gen_output)) # Save generated tile

    disc_generated_output = discriminator([tf.expand_dims(tiles[i,:,:,:], axis=0), tf.expand_dims(gen_output, axis=-1)], 
                                          training=False) # Discriminator output with "generated" images

    # Generator loss computation
    gen_total_loss, _, _ = generator_loss(disc_generated_output,gen_output,
                                          tf.expand_dims(tf.cast(tiles_target[i,:,:,:],tf.float32), axis=0))
    
    # Metrics computation
    ssim = tf.py_function(SSIM, [gen_output, tf.expand_dims(tf.cast(tiles_target[i,:,:,:],tf.float32),0)], tf.float32, name = "ssim")
    psnr = tf.py_function(PSNR, [gen_output, tf.expand_dims(tf.cast(tiles_target[i,:,:,:],tf.float32),0)], tf.float32, name = "psnr")
    
    # Accumulate losses and metrics for all tiles
    cumulative_gen_loss += gen_total_loss
    cumulative_ssim += ssim
    cumulative_psnr += psnr

  if show:  
    # Merging generated outputs (in case we would like to see)
    if not os.path.exists(params.prediction_path):
      os.makedirs(params.prediction_path)
    gen_merged_image = tf.py_function(
      merge_tile_perimage, [gen_tiles], tf.float32, name="create_tiles"
    )
    plt.figure()
    plt.subplot(121)
    plt.imshow(np.squeeze(gen_merged_image), cmap="gray"), plt.axis('off'), plt.title("Generated image")
    plt.subplot(122)
    plt.imshow(np.squeeze(target)), plt.axis('off'), plt.title("Target image")
    plt.savefig(os.path.join(params.prediction_path,'prediction{}.tiff'.format(counter)))
    #plt.show()
    counter+=1

  # Mean losses and metrics
  gen_mean_loss = cumulative_gen_loss/tiles.shape[0]
  ssim_mean = cumulative_ssim/tiles.shape[0]
  psnr_mean = cumulative_psnr/tiles.shape[0]

  return gen_mean_loss, ssim_mean, psnr_mean


#@tf.function()
def fit(train_ds, val_ds, train_length, val_length):
  """
  Set up training and validation of a given model
  
  Params:
      - train_ds : Tensorflow dataset
          Training dataset
      - val_ds : Tensorflow dataset
          Validation dataset
      - train_length : int
          Number of input-target image pairs in the training dataset
      - val_length : int
          Number of input-target image pairs in the validation dataset
          
  Outputs:
      void
  
  """
    
  for epoch in range(params.epochs):
    start = time.time()
    print("Epoch: {}".format(epoch))

    #display.clear_output(wait=True)
      
    # Decode tfrecords to work with them : TRAINING
    cumulative_gen_loss = 0
    cumulative_disc_loss = 0
    cumulative_ssim = 0
    cumulative_psnr = 0
    for record in train_ds:
      parsed_record = parse_record(record)
      decoded_record = decode_record(parsed_record)
      input_train, target_train = decoded_record

      gen_loss, disc_loss, ssim, psnr = train_step(input_train, target_train, epoch)
      cumulative_gen_loss += gen_loss
      cumulative_disc_loss += disc_loss
      cumulative_ssim += ssim
      cumulative_psnr += psnr
      print('.', end='')

    train_gen_mean_loss = cumulative_gen_loss/train_length
    train_disc_mean_loss = cumulative_disc_loss/train_length
    train_ssim_mean = cumulative_ssim/train_length
    train_psnr_mean = cumulative_psnr/train_length

    #val_length = len(list(val_ds))
    rand_ind = random.randint(0,val_length) # Random index used to determine if validation results are printed or not

    # Decode tfrecords to work with them : VALIDATION
    cumulative_gen_loss = 0
    cumulative_ssim = 0
    cumulative_psnr = 0
    
    cont_val = 0
    for record in val_ds:
      parsed_record = parse_record(record)
      decoded_record = decode_record(parsed_record)
      input_val, target_val = decoded_record
      if rand_ind == cont_val:
        gen_loss, ssim, psnr = val_step(input_val, target_val, epoch,  True, COUNTER) # Show results for validation
      else:
        gen_loss, ssim, psnr = val_step(input_val, target_val, epoch, generator, counter=COUNTER) # Do not show anything
        cont_val += 1

      cumulative_gen_loss += gen_loss
      cumulative_ssim += ssim
      cumulative_psnr += psnr

    val_gen_mean_loss = cumulative_gen_loss/val_length
    val_ssim_mean = cumulative_ssim/val_length
    val_psnr_mean = cumulative_psnr/val_length
      
    print("Epoch: {} Train Generator loss: {} Train Discriminator Loss: {} Train SSIM: {} Train PSNR:{} Val Generator loss: {} Val SSIM: {} Val PSNR:{}"
    .format(epoch, train_gen_mean_loss, train_disc_mean_loss, train_ssim_mean, train_psnr_mean, val_gen_mean_loss, val_ssim_mean, val_psnr_mean))

    # Saving losses and metrics values
    with open(os.path.join(params.loss_path,"train_generator_loss_" + str(params.magnification) + "x_channel" + str(params.fluor_class)) + ".txt","a") as g_t, open(os.path.join(params.loss_path,"train_discriminator_loss_" + str(params.magnification) + "x_channel" + str(params.fluor_class)) + ".txt","a") as d_t:
      g_t.write("{}\n".format(train_gen_mean_loss))
      d_t.write("{}\n".format(train_disc_mean_loss))

    with open(os.path.join(params.loss_path,"val_generator_loss_" + str(params.magnification) + "x_channel" + str(params.fluor_class)) + ".txt","a") as g_v:
      g_v.write("{}\n".format(val_gen_mean_loss))

    with open(os.path.join(params.metrics_path,"train_SSIM_" + str(params.magnification) + "x_channel" + str(params.fluor_class)) + ".txt","a") as ssim_t,open(os.path.join(params.metrics_path,"train_PSNR_" + str(params.magnification) + "x_channel" + str(params.fluor_class)) + ".txt","a") as psnr_t:
      ssim_t.write("{}\n".format(train_ssim_mean))
      psnr_t.write("{}\n".format(train_psnr_mean))

    with open(os.path.join(params.metrics_path,"val_SSIM_" + str(params.magnification) + "x_channel" + str(params.fluor_class)) + ".txt","a") as ssim_v,open(os.path.join(params.metrics_path,"val_PSNR_" + str(params.magnification) + "x_channel" + str(params.fluor_class)) + ".txt","a") as psnr_v:
      ssim_v.write("{}\n".format(val_ssim_mean))
      psnr_v.write("{}\n".format(val_psnr_mean))

    # saving (checkpoint) the model every 1/5 epochs
    if (epoch + 1) % params.save_epochs == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))


# Main script
COUNTER = 0
PATH = os.path.join("preprocessed_data", str(params.magnification) + "x_images") # This varies depending on the magnification we want to train on

if not os.path.exists(params.loss_path):
  os.makedirs(params.loss_path)

if not os.path.exists(params.metrics_path):
  os.makedirs(params.metrics_path)

# Create log files
with open(os.path.join(params.loss_path,"train_generator_loss_" + str(params.magnification) + "x_channel" + str(params.fluor_class)) + ".txt","w") as g_t:
  pass
with open(os.path.join(params.loss_path,"val_generator_loss_" + str(params.magnification) + "x_channel" + str(params.fluor_class)) + ".txt","w") as g_v:
  pass
with open(os.path.join(params.loss_path,"train_discriminator_loss_" + str(params.magnification) + "x_channel" + str(params.fluor_class)) + ".txt","w") as d_t:
  pass
with open(os.path.join(params.loss_path,"val_generator_loss_" + str(params.magnification) + "x_channel" + str(params.fluor_class)) + ".txt","w") as g_v:
  pass
with open(os.path.join(params.metrics_path,"train_SSIM_" + str(params.magnification) + "x_channel" + str(params.fluor_class)) + ".txt","w") as ssim_t:
  pass
with open(os.path.join(params.metrics_path,"train_PSNR_" + str(params.magnification) + "x_channel" + str(params.fluor_class)) + ".txt","w") as psnr_t:
  pass
with open(os.path.join(params.metrics_path,"val_SSIM_" + str(params.magnification) + "x_channel" + str(params.fluor_class)) + ".txt","w") as ssim_t:
  pass
with open(os.path.join(params.metrics_path,"val_PSNR_" + str(params.magnification) + "x_channel" + str(params.fluor_class)) + ".txt","w") as psnr_t:
  pass


# Split into train and validation images inside each magnification
input_paths, target_paths = getInputTargetPaths(PATH) # Input and target paths
train_input_paths, val_input_paths, train_target_paths, val_target_paths = train_test_split(input_paths, target_paths, test_size = params.test_size, random_state = 42)


# Load images as array stack
x_train, y_train = load_images_from_folder(train_input_paths, train_target_paths)
x_val, y_val = load_images_from_folder(val_input_paths, val_target_paths)


# +
# Train: prepare TF Records file
train_record_file = 'train.tfrecords'
train_samples = x_train.shape[0]
dimension1 = x_train.shape[1]
dimension2 = x_train.shape[2]

with tf.io.TFRecordWriter(train_record_file) as writer:
   for i in range(train_samples):
      image = x_train[i]
      label = y_train[i]
      tf_example = image_example(image, label, dimension1,dimension2)
      writer.write(tf_example.SerializeToString())


# Validation: prepare TF Records file
val_record_file = 'val.tfrecords'
val_samples = x_val.shape[0]
dimension1 = x_val.shape[1]
dimension2 = x_val.shape[2]

with tf.io.TFRecordWriter(val_record_file) as writer:
   for i in range(val_samples):
      image = x_val[i]
      label = y_val[i]
      tf_example = image_example(image, label, dimension1,dimension2)
      writer.write(tf_example.SerializeToString())
# -


# Create the dataset object from tfrecord file(s)
val_dataset = tf.data.TFRecordDataset(val_record_file, buffer_size=params.val_buffer_size)
train_dataset = tf.data.TFRecordDataset(train_record_file, buffer_size=params.train_buffer_size)


# Call the generator & the discriminator
generator = Generator()
generator.summary()
discriminator = Discriminator()
discriminator.summary()

generator_optimizer = tf.keras.optimizers.Adam(params.learning_rate, beta_1=params.beta1)
discriminator_optimizer = tf.keras.optimizers.Adam(params.learning_rate, beta_1=params.beta1)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_" + str(params.magnification) + "x_channel" + str(params.fluor_class))
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# log_dir="logs/"

#summary_writer = tf.summary.create_file_writer(
 # log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


"""Now run the training and validation loop:"""

fit(train_dataset, val_dataset, train_samples, val_samples)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

"""## Generate using test dataset"""

# Run the trained model on a few examples from the test dataset
#def predict_images(number_predictions):
#  for inp, tar in val_dataset.take(number_predictions):
#    generate_images(generator, inp, tar)

