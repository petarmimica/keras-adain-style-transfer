# -*- coding: utf-8 -*-

import numpy as np
import cv2
from typing import Tuple

# Credit: https://gist.github.com/IdeaKing/11cf5e146d23c5bb219ba3508cca89ec
def resize_with_pad(image: np.array,
                    new_shape: Tuple[int, int],
                    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

def preprocess(image, img_size=(224,224)):
    """
    # Args
        image : rgb-ordered array
    """
    image = np.expand_dims(cv2.resize(image.astype(np.float32), img_size), axis=0)
    return image

def preprocess_with_pad(image, img_size=(224,224)):
    """
    # Args
        image : rgb-ordered array
    """
    image = np.expand_dims(resize_with_pad(image.astype(np.float32), img_size), axis=0)
    return image


def get_params(t7_file):
    import torchfile
    t7 = torchfile.load(t7_file, force_8bytes_long=True)
    weights = []
    biases = []
    for idx, module in enumerate(t7.modules):
        weight = module.weight
        bias = module.bias
        if idx == 0:
            print(bias)
        elif weight is not None:
            weight = weight.transpose([2,3,1,0])
            weights.append(weight)
            biases.append(bias)
    return weights, biases


def set_params(model, weights, biases):
    i = 0
    for layer in model.layers:
        # assign params
        if len(layer.get_weights()) > 0:
            layer.set_weights([weights[i], biases[i]])
            i += 1

def plot(imgs):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title("content image")
    plt.imshow(imgs[0])
    plt.subplot(1, 3, 2)
    plt.axis('off')    
    plt.title("style image")
    plt.imshow(imgs[1])
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title("stylized image")
    plt.imshow(imgs[2])
    plt.show()


def print_t7_graph(t7_file):
    import torchfile
    t7 = torchfile.load(t7_file, force_8bytes_long=True)
    for idx, module in enumerate(t7.modules):
        print("{}, {}".format(idx, module._typename))
        
        weight = module.weight
        bias = module.bias
        if weight is not None:
            weight = weight.transpose([2,3,1,0])
            print("    ", weight.shape, bias.shape)


if __name__ == '__main__':
    from adain import PROJECT_ROOT
    import os
    # print_t7_graph(os.path.join(PROJECT_ROOT, "pretrained", "vgg_normalised.t7"))
    print_t7_graph(os.path.join(PROJECT_ROOT, "pretrained", "decoder.t7"))

