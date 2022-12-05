#Local files
from params import *

#Google Cloud
from google.cloud import storage

#Tensor Flow
from tensorflow.keras import Sequential, Model #Encoding layer , Autoencoding layer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense #Encoding layer
from tensorflow.keras.layers import Reshape, Conv2DTranspose #Decoding layer
from tensorflow.keras.layers import Input #Autoencoding layer
from tensorflow.keras import preprocessing #Nearest neighbors
from tensorflow.image import resize
import tensorflow as tf

def build_encoder(latent_dimension):
    
    '''
    Summary: encodes an image down to latent space of size equal to latent_dimension input
    Input: image of specified height, length and channel amount
    Return: returns an encoder model, of output_shape equals to latent_dimension
    
    '''
    encoder = Sequential()
    
    encoder.add(Conv2D(8, (2,2), input_shape=(IMAGE_HEIGHT, IMAGE_LENGTH, CHANNELS), activation='relu'))
    encoder.add(MaxPooling2D(2))

    encoder.add(Conv2D(16, (2, 2), activation='relu'))
    encoder.add(MaxPooling2D(2))

    encoder.add(Conv2D(32, (2, 2), activation='relu'))
    encoder.add(MaxPooling2D(2))     

    encoder.add(Flatten())
    encoder.add(Dense(latent_dimension, activation='tanh'))
    
    return encoder


def build_decoder(latent_dimension):
    
    '''
    Summary: decodes an image into original dimensions
    Input: image in latent space dimensions
    Return: returns a decoder model, of output_shape equals to original image height, shape and channels
    
    '''
    
    decoder = Sequential()
    decoder.add(Dense((IMAGE_HEIGHT/4)*(IMAGE_LENGTH/4)*8, activation='tanh', input_shape=(latent_dimension,)))
    decoder.add(Reshape((25, 25, 8)))  # no batch axis here
    decoder.add(Conv2DTranspose(8, (2, 2), strides=2, padding='same', activation='relu'))

    decoder.add(Conv2DTranspose(CHANNELS, (2, 2), strides=2, padding='same', activation='relu'))
    return decoder

def build_autoencoder(encoder, decoder):
    
    '''
    Summary: zips together encoder and decoder
    Input: image of specified height, length and channel amount
    Return: returns an autoencoder model, of output_shape equals to original image height, shape and channels
    
    '''
    inp = Input((IMAGE_HEIGHT, IMAGE_LENGTH, CHANNELS))
    encoded = encoder(inp)
    decoded = decoder(encoded)
    autoencoder = Model(inp, decoded)
    return autoencoder

def compile_autoencoder(autoencoder):
    
    '''
    Summary: compiles autoencoder
    Input: autoencoder
    Return: Compiled autoencoder
    
    '''
    
    autoencoder.compile(loss='mse',
                  optimizer='adam')
    

def load_image(path):
    
    '''
    
    Summary: loads image, decodes and resizes into shape model accepts
    Input: image path
    Return: model ready image
    
    '''
    
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMAGE_HEIGHT,IMAGE_LENGTH])/255
    return image

def load_data(bucket_name):
    
    '''
    
    Summary: gets images from Cloud bucket
    Input: bucket_name
    Return: dataset of model ready images
    
    '''
    
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    dirname = 'New_Image/'
    image_paths = list(bucket.list_blobs(prefix=dirname))
    image_paths = [path.public_url for path in image_paths]
    image_paths = [path for path in image_paths if not path.endswith(".csv")]
    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.map(load_image)
    ds = ds.map(lambda x: (x,x))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.cache()
    return ds
   