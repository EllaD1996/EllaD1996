from tensorflow import image, reshape
import os

def user_image_reshape(img):
    IMAGE_HEIGHT = os.environ.get("IMAGE_HEIGHT")
    IMAGE_LENGTH = os.environ.get("IMAGE_LENGTH")
    CHANNELS = os.environ.get("CHANNELS")
    img = image.resize(img, [IMAGE_HEIGHT, IMAGE_LENGTH])/255
    img = reshape(img, (-1,IMAGE_HEIGHT,IMAGE_LENGTH,CHANNELS))
    return img
