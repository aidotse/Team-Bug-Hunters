import os

# Augmentation parameters
flip_prob = 0.5
rot_angle = 45
rot_prob = 0.5
brightness_limit = 0.05
contrast_limit = 0.05
bright_cont_prob = 0.5
alpha = 0.5
sigma = 2
points = 4
elastic_prob = 1
prob_all = 1

# Tiling parameters
tile_size = 512

# Training parameters
fluor_class = 1
output_channels = 1
epochs = 150
LAMBDA = 50
test_size = 0.25
train_buffer_size = 100
val_buffer_size = int(round(train_buffer_size*test_size))
ssim_lambda = 1
loss_path = os.path.join(os.getcwd(),"losses")
prediction_path= os.path.join(os.getcwd(),"predictions")
metrics_path = os.path.join(os.getcwd(),"metrics")
save_epochs = 20
learning_rate = 2e-4
beta1 = 0.5
magnification = 20
batch_size = 2
