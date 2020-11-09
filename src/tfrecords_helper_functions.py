import tensorflow as tf
import numpy as np

# Convert values to compatible tf.Example types.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Create the features dictionary.
def image_example(image, label, dimension1, dimension2):
    feature = {
        'dimension1': _int64_feature(dimension1),
        'dimension2': _int64_feature(dimension2),
        'label': _bytes_feature(label.tobytes()),
        'image_raw': _bytes_feature(image.tobytes()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

# Decoding function
def parse_record(record):
    name_to_features = {
        'dimension1': tf.io.FixedLenFeature([], tf.int64),
        'dimension2': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(record, name_to_features)

def decode_record(record):
    #image = tf.io.decode_raw(
     #   record['image_raw'], out_type=np.uint16, little_endian=True, fixed_length=None, name=None
    #)
    #label = tf.io.decode_raw(
     #   record['label'], out_type=np.uint16, little_endian=True, fixed_length=None, name=None
    #)
    
    image = tf.io.decode_raw(
        record['image_raw'], out_type=np.uint16, little_endian=True, name=None)
    label = tf.io.decode_raw(
        record['label'], out_type=np.uint16, little_endian=True, name=None)

    dimension1 = record['dimension1']
    dimension2 = record['dimension2']
    image = tf.reshape(image, (dimension1, dimension2,1))
    label = tf.reshape(label, (dimension1, dimension2,1))
    return (image, label)


