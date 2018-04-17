# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 21:28:13 2018

https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

@author: zeynab
"""

from scipy.misc import imsave
import numpy as np
import time
from keras import applications

from vis.utils import utils
from keras import activations

model = applications.VGG16(include_top=False,
                           weights='imagenet')

layer_dict = dict([(layer.name, layer) for layer in model.layers])
#print(layer_dict)

from vis.visualization import visualize_activation

from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (18, 6)

# 20 is the imagenet category for 'ouzel'
img = visualize_activation(model, -1, filter_indices=20)
plt.imshow(img)

layer_idx = utils.find_layer_idx(model, 'block5_conv1')
model.layers[-1].activation = activations.linear
model = utils.apply_modifications(model)