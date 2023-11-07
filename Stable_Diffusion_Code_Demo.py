# -*- coding: utf-8 -*-
"""Stable Diffusion Code Demo
# High-performance image generation using Stable Diffusion in KerasCV
"""

!pip install tensorflow keras_cv --upgrade --quiet

import time
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt

"""First, construct a model:"""

model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

"""Next, give it a prompt:"""

images = model.text_to_image("photograph of an astronaut riding a horse", batch_size=3)


def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")


plot_images(images)

"""Test the model with a medical text prompt"""

test = model.text_to_image("chest x-ray with a pleural effusion", batch_size=3)

def plot_images(test):
    plt.figure(figsize=(20, 20))
    for i in range(len(test)):
        ax = plt.subplot(1, len(test), i + 1)
        plt.imshow(test[i])
        plt.axis("off")


plot_images(test)

"""Pretty incredible!"""
