import numpy as np
import os
import sys
import imageio
import albumentations as A
import time
import params
import tensorflow as tf


@tf.function
def SSIM(pred, ref):
    """
    Compute SSIM between predicted and reference tensors in Tensorflow
    
    Params:
        - pred : TensorFlow tensor
            Predicted tensor
        - ref : TensorFlow tensor
            Reference tensor
            
    Outputs:
        - ssim : float
            SSIM value
    
    """

    return tf.image.ssim(pred, ref, (2**16-1), filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)


def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator


@tf.function
def PSNR(pred, ref):
    """
    Compute PSNR between predicted and reference tensors in Tensorflow
    
    Params:
        - pred : TensorFlow tensor
            Predicted tensor
        - ref : TensorFlow tensor
            Reference tensor
            
    Outputs:
        - psnr : float
            PSNR value
            
    """
    
    psnr = tf.image.psnr(pred, ref, 2**16 - 1, name=None)
    
    return psnr

