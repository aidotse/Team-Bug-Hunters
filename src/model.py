#!/usr/bin/env python
# coding: utf-8
# %%
import os
import tensorflow as tf
import params
from tensorflow.keras.layers import Layer, InputSpec


# %%
def bn_act(x, act=True): # Batch-normalization + Activation functions
    x = tf.keras.layers.BatchNormalization()(x)
    if act == True:
        x = tf.keras.layers.LeakyReLU()(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1): # Convolution
    conv = bn_act(x)
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1): # First set of convolutions and residual shortcuts
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = tf.keras.layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1): # Residual connection block
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = tf.keras.layers.Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip): # Upsample block + concatenation (unused) (split for upsample and concat blochs for Attention U-Net)
    u = tf.keras.layers.UpSampling2D((2, 2))(x)
    c = tf.keras.layers.Concatenate()([u, xskip])
    return c

def upsample_block(x): # Upsample block
    u = tf.keras.layers.UpSampling2D((2, 2))(x)
    return u
    
def concat_block(x, xskip): # Concatenation block
    c = tf.keras.layers.Concatenate()([x, xskip])
    return c
    
def attention(tensor, att_tensor, n_filters=512, kernel_size=[1, 1]): # Attention gate
    g1 = tf.keras.layers.Conv2D(n_filters, kernel_size)(tensor)
    x1 = tf.keras.layers.Conv2D(n_filters, kernel_size)(att_tensor)
    net = tf.keras.layers.Add()([g1, x1])
    net = tf.keras.layers.ReLU()(net)
    net = tf.keras.layers.Conv2D(1, kernel_size)(net)
    net = tf.nn.sigmoid(net)
    net = net * att_tensor
    return net

def MSE_SSIM_loss(pred, ref, name = "mse_ssim"): # Custom loss: MSE + 10**5*SSIM (unused)
    ssim = 1 - tf.image.ssim(ref, pred, (2**16-1), filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    mse = tf.keras.losses.MSE(ref, pred)
    
    return mse + 10**5*ssim

def ResUNet(): # Residual U-Net (unused)
    f = [64, 128, 256, 512, 512]
    inputs = tf.keras.layers.Input((params.tile_size//params.downsample_factor, params.tile_size//params.downsample_factor, 1))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, params.filters[1], strides=2)
    e3 = residual_block(e2, params.filters[2], strides=2)
    e4 = residual_block(e3, params.filters[3], strides=2)
    e5 = residual_block(e4, params.filters[4], strides=2)
    
    ## Bridge
    b0 = conv_block(e5, params.filters[4], strides=1)
    b1 = conv_block(b0, params.filters[4], strides=1)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, params.filters[4])
    
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, params.filters[3])
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, params.filters[2])
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, params.filters[1])
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same")(d4)
    outputs = tf.keras.layers.LeakyReLU()(outputs)
    model = tf.keras.Model(inputs, outputs)
    return model


def AttentionResUNet(): # Attention-Residual U-Net
    inputs = tf.keras.layers.Input((params.tile_size//params.downsample_factor, params.tile_size//params.downsample_factor, 1))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, params.filters[0])
    e2 = residual_block(e1, params.filters[1], strides=2)
    e3 = residual_block(e2, params.filters[2], strides=2)
    e4 = residual_block(e3, params.filters[3], strides=2)
    e5 = residual_block(e4, params.filters[4], strides=2)
    
    ## Bridge
    b0 = conv_block(e5, params.filters[4], strides=1)
    b1 = conv_block(b0, params.filters[4], strides=1)
    
    ## Decoder
    u1 = upsample_block(b1)
    att1 = attention(e4, u1, params.filters[4])
    c1 = concat_block(u1, att1)
    d1 = residual_block(c1, params.filters[4])
    
    u2 = upsample_block(d1)
    att2 = attention(e3, u2, params.filters[3])
    c2 = concat_block(u2, att2)
    d2 = residual_block(c2, params.filters[3]) 
    
    u3 = upsample_block(d2)
    att3 = attention(e2, u3, params.filters[2])
    c3 = concat_block(u3, att3)
    d3 = residual_block(c3, params.filters[2]) 
    
    u4 = upsample_block(d3)
    att4 = attention(e1, u4, params.filters[1])
    c4 = concat_block(u4, att4)
    d4 = residual_block(c4, params.filters[1]) 

    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same")(d4)
    outputs = tf.keras.layers.LeakyReLU()(outputs)
    model = tf.keras.Model(inputs, outputs)
    return model

