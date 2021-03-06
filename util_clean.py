# Utility, modified on https://github.com/elleryqueenhomels/arbitrary_style_transfer.git. 
# Add a delete funciton to clean out the WikiArt images that are too large to train or have zeor pixel. 

import numpy as np
import os

from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from scipy.misc import imread, imsave, imresize

def list_images(directory):
    images = []
    for file in listdir(directory):
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jepg'):
            images.append(join(directory, file))
    return images

def list_images_del(directory):
    images = []
    for file in listdir(directory):
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))   
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
    delete_wrong_images(directory)
    return images

def delete_wrong_images(directory):
    # Add a delete funciton to clean out the WikiArt images that are too large to train or have zeor pixel. 
    for file in listdir(directory):
        path = join(directory, file)
        print(f"checking validation of {path}...")
        try:
            image = imread(path, mode='RGB')
        except:
            print(f"{path} is too big. Remove it")
            os.remove(path)
            continue
        if len(image.shape) == 0:
            print(f"{path} has irregular shape: {image.shape}. Please remove this image and run it again.")
            os.remove(path)
     
def get_train_images(paths, resize_len=512, crop_height=256, crop_width=256):
    images = []
    for path in paths:
        try:
            image = imread(path, mode='RGB')
        except:
            print(f"{path} is too big to read. Please remove this image and run it again")
            continue
        if len(image.shape) == 0:
            print(f"{path} has irregular shape: {image.shape}. Please remove this image and run ti again.")
            continue
        height, width, _ = image.shape

        if height < width:
            new_height = resize_len
            new_width  = int(width * new_height / height)
        else:
            new_width  = resize_len
            new_height = int(height * new_width / width)

        image = imresize(image, [new_height, new_width], interp='nearest')

        # crop the image
        start_h = np.random.choice(new_height - crop_height + 1)
        start_w = np.random.choice(new_width - crop_width + 1)
        image = image[start_h:(start_h + crop_height), start_w:(start_w + crop_width), :]

        images.append(image)

    images = np.stack(images, axis=0)

    return images


def get_images(paths, height=None, width=None):
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in paths:
        image = imread(path, mode='RGB')

        if height is not None and width is not None:
            image = imresize(image, [height, width], interp='nearest')

        images.append(image)

    images = np.stack(images, axis=0)

    return images


def save_images(paths, datas, save_path, prefix=None, suffix=None):
    if isinstance(paths, str):
        paths = [paths]

    assert(len(paths) == len(datas))

    if not exists(save_path):
        mkdir(save_path)

    if prefix is None:
        prefix = ''
    if suffix is None:
        suffix = ''

    for i, path in enumerate(paths):
        data = datas[i]

        name, ext = splitext(path)
        name = name.split(sep)[-1]
        
        path = join(save_path, prefix + name + suffix + ext)

        imsave(path, data)

