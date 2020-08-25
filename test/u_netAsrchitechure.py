'''
@python-version:3.x
@license: Apache Licence
@author: Joker
@file: u_netAsrchitechure.py
@time: 2020/08/25 09:48:46
@blog: https://github.com/Heyra-Joker
@description: --
🤡
code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
🤡
'''

import os
import gc
import copy
import sys

import numpy as np
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback

import matplotlib
import matplotlib.pyplot as plt

# os.chdir('..')
sys.path.append(os.getcwd())
print(sys.path)
from libs.util import MaskGenerator
from libs.pconv_model import PConvUnet

# Settings
BATCH_SIZE = 4

# Imagenet Rescaling
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


mask_generator = MaskGenerator(512, 512, 3, rand_seed=42)
# Load image
img = np.array(Image.open('/Users/yanxiya/hw/cv/PConv-Keras-master/data/train/1.jpg').resize((512, 512))) / 255


class DataGenerator(ImageDataGenerator):
    def flow(self, x, *args, **kwargs):
        while True:
            # Get augmentend image samples
            ori = next(super().flow(x, *args, **kwargs))

            # Get masks for each image sample
            mask = np.stack([mask_generator.sample() for _ in range(ori.shape[0])], axis=0)

            # Apply masks to all image sample
            masked = copy.deepcopy(ori)
            masked[mask==0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori

# Create datagen
datagen = DataGenerator(  
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Create generator from numpy array
batch = np.stack([img for _ in range(BATCH_SIZE)], axis=0)
generator = datagen.flow(x=batch, batch_size=BATCH_SIZE)

model = PConvUnet(vgg_weights='./data/logs/pytorch_to_keras_vgg16.h5')

model.fit_generator(
    generator,
    steps_per_epoch=2,
    epochs=1,
    callbacks=[
        TensorBoard(
            log_dir='./data/logs/single_image_test',
            write_graph=False
        ),
        ModelCheckpoint(
            './data/logs/single_image_test/weights.{epoch:02d}-{loss:.2f}.h5',
            monitor='loss', 
            save_best_only=True, 
            save_weights_only=True
        ),
    ],
)

