import os

# Augmentation parameters
flip_prob = 0.5 # Flip probability
rot_angle = 45 # Max. rotation angle
rot_prob = 0.5 # Rotation probability
brightness_limit = 0.05 # Max. variation in brightness
contrast_limit = 0.05 # Max.variation in contrast
bright_cont_prob = 0.5 # Probability for alteration of brightness + contrast
alpha = 0.5 # Param for elastic deformation (unused)
sigma = 2 # Param for elastic deformation (unused)
points = 4 # Param for elastic deformation (unused)
elastic_prob = 1 # Elastic deformation probability (unused)
prob_all = 1 # Probability for augmentation events

# Tiling parameters
tile_size = 512 # Size of tiled images
downsample_factor = 1 # Downsampling factor (if 1, no actual downsampling applied)

# Training parameters
fluor_ch = 2  # 1: Nuclei, 2: Lipids, 3:Cytoplasm (CHANGE THIS FOR CHOOSING THE FLUORESCENCE CHANNEL FOR TRAINING OR VALIDATING!!)
output_channels = 1 # Output channels for the network
epochs = 100 # Epochs
val_size = 0.20 # Validation set size (0 to 1)
test_size = 0.1 # Test set size (0 to 1)
train_buffer_size = 100 # Train set buffer size (unused)
val_buffer_size = int(round(train_buffer_size*test_size)) # Validation set buffer size (relative to train set)
ssim_lambda = 1 # Whether to include SSIM in the loss or not
loss_path = os.path.join(os.getcwd(),"losses") # Folder for saving loss values (unused)
prediction_path= os.path.join(os.getcwd(),"predictions") # Folder for saving predictions
metrics_path = os.path.join(os.getcwd(),"metrics") # Folder for saving metric values
save_epochs = 20 # Save epochs every x number (unused)
learning_rate = 1e-4 # Learning rate
beta1 = 0.5 # Adam optimizer beta 1
magnification = 20 # 20: 20x Magnification, 40: 40x Magnification, 60: 60x Magnification (CHANGE THIS FOR CHOOSING THE MAGNIFICATION!!)
batch_size = 1 # Batch size
kernel_size = 4 # Convolutional kernel size (except in attention gates)
filters = [64, 128, 256, 512, 512] # Filter sizes


