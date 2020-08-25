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
from libs.util import MaskGenerator
from libs.pconv_model import PConvUnet

# Settings
BATCH_SIZE = 4

# Imagenet Rescaling
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def plot_sample_data(masked, mask, ori, middle_title='Raw Mask', name=""):
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].imshow(masked[:,:,:])
    axes[0].set_title('Masked Input')
    axes[1].imshow(mask[:,:,:])
    axes[1].set_title(middle_title)
    axes[2].imshow(ori[:,:,:])
    axes[2].set_title('Target Output')
    plt.savefig(f"./{name}.jpg")
    

mask_generator = MaskGenerator(512, 512, 3, rand_seed=42)
# Load image
img = np.array(Image.open('./data/train/1.jpg').resize((512, 512))) / 255

# Load mask
mask = mask_generator.sample()

# Image + mask
masked_img = copy.deepcopy(img)
masked_img[mask==0] = 1

plot_sample_data(masked_img, mask * 255, img)


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
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # horizontal_flip=True
)

# Create generator from numpy array
batch = np.stack([img for _ in range(BATCH_SIZE)], axis=0)

generator = datagen.flow(x=batch, batch_size=BATCH_SIZE)

# for [masked, mask], ori in generator:
#     images = np.hstack([masked[0], mask[0], ori[0]])
#     plt.imshow(images)
#     plt.show()
#     break

model = PConvUnet(vgg_weights='./data/logs/pytorch_to_keras_vgg16.h5')

model.fit_generator(
    generator,
    steps_per_epoch=2,
    epochs=10,
    callbacks=[
        TensorBoard(
            log_dir='./data/logs/single_image_test_checkpoints',
            write_graph=False
        ),
        ModelCheckpoint(
            './data/logs/single_image_test/weights.{epoch:02d}-{loss:.2f}.h5',
            monitor='loss', 
            save_best_only=True, 
            save_weights_only=True
        ),
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: plot_sample_data(
                masked_img, 
                model.predict(
                    [
                        np.expand_dims(masked_img,0), 
                        np.expand_dims(mask,0)
                    ]
                )[0]
                , 
                img,
                middle_title='Prediction',
                name=epoch
            )
        )
    ],
)