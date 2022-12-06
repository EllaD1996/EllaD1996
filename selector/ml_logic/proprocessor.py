from tensorflow import image, reshape
import os

from dotenv import load_dotenv, find_dotenv

env_path = find_dotenv()
load_dotenv(env_path)

def user_image_reshape(img):
    print(img.shape)
    IMAGE_HEIGHT = 100
    IMAGE_LENGTH = 100
    CHANNELS = 3
    img = image.resize(img, [IMAGE_HEIGHT, IMAGE_LENGTH])/255
    img = reshape(img, (-1,IMAGE_HEIGHT,IMAGE_LENGTH,CHANNELS))
    return img
