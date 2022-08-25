#!/usr/bin/env python3

import os
import sys
import argparse
import json
import logging
import glob
import cv2 as cv2
import tensorflow as tf
import datetime
from tensorflow import keras
from tensorflow.keras import layers
"""
Name:       pt5_Xception
Author:     robertdcurrier@gmail.com
Created:    2022-02-31
Notes:      Used examples from F. Chollet's repo to get started.
            Now with small version of Xception network. Much better
            performance. Requires TF 2.7.0.
"""

def genDataSet():
    """
    Creates test and train datasets
    """
    logging.info('genDataSet()...')
    config = get_config()
    img_x = config["keras"]["img_size_x"]
    img_y = config["keras"]["img_size_y"]
    image_size = (img_x, img_y)
    batch_size = 32
    platform = config['system']['platform']
    training_dir = config['keras']['platform'][platform]['training_dir']

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        training_dir,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        training_dir,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    return(train_ds, val_ds)


def make_model(input_shape, num_classes):
    """ Let's make a model, baby! Using small version of Xception"""
    logging.info('make_model()...')
    inputs = keras.Input(shape=input_shape)
    # Let's add some diversity
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ]
    )
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


def trainModel():
    """
    What it says
    """
    logging.info('trainModel()...')
    config = get_config()
    num_classes = config['keras']['num_classes']
    if num_classes == 2:
        loss_func = 'binary_crossentropy'
    else:
        loss_func = 'sparse_categorical_crossentropy'
    img_x = config["keras"]["img_size_x"]
    img_y = config["keras"]["img_size_y"]
    image_size = (img_x, img_y)
    epochs = config["keras"]["epochs"]
    train_ds, val_ds = genDataSet()
    model = make_model(input_shape=image_size + (3,),num_classes=num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=loss_func,
        metrics=["accuracy"]
    )
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)
    model.fit(
        train_ds, epochs=epochs, validation_data=val_ds,
        callbacks=[tensorboard_callback])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)
    platform = config['system']['platform']
    model_file = config['keras']['platform'][platform]['model_file']
    logging.info('trainModel(): Saving %s as new model file' % model_file)
    model.save(model_file,save_format='h5')

    return model


def testModel(model):
    """ See how good we are
    Author: robertdcurrier@gmail.com
    Created:    2022-02-18
    Modified:   2022-04-21

    """
    logging.info('testModel()...')
    # Get some sample images
    config = get_config()
    labels = config['keras']['labels']
    img_x = config["keras"]["img_size_x"]
    img_y = config["keras"]["img_size_y"]
    num_classes = config["keras"]["num_classes"]
    platform = config['system']['platform']
    testing_dir = config['keras']['platform'][platform]['testing_dir']
    the_glob = '%s/*.png' % testing_dir
    tfiles = sorted(glob.glob(the_glob))
    key_list = list(labels.keys())
    val_list = list(labels.values())


    for test_image in tfiles:
        image_size = (img_x, img_y)
        img = keras.preprocessing.image.load_img(
            test_image, target_size=image_size
            )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)
        labeled_scores = []
        scores = (predictions[0])
        index = 0
        print('testModel(): %s classified as:' % test_image)
        for score in scores:
            print("%s: %0.2f%%" % (key_list[index], scores[index]*100))
            index+=1
        print('-----------')


def load_model():
    """Load TensorFlow model and cell weights from lib.

    Author: robertdcurrier@gmail.com
    Created:    2019-07-18
    Modified:   2022-01-31
    """
    logging.info('load_model()...')
    config = get_config()
    model_file = config['keras']['model_file']
    model = keras.models.load_model(model_file)
    return model


def get_config():
    """From config.json.

    Author: robertdcurrier@gmail.com
    Created:    2019-07-18
    Modified:   2019-07-18
    """
    logging.info('get_config()...')
    c_file = open('configs/pt5_Xception.cfg').read()
    config = json.loads(c_file)
    return config


def get_cli_args():
    """What it say.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2022-01-31

    Notes: Slimmed down for pt5_Xception
    """
    logging.debug('get_cli_args()')
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("-to", "--testonly", help="test only",
                        action="store_true")
    args = vars(arg_p.parse_args())
    return args


def train_and_test():
    """
    Main loop
    """
    logging.info('train_and_test()...')
    args = get_cli_args()
    if args["testonly"]:
        model = load_model()
        testModel(model)
    else:
        model = trainModel()
        testModel(model)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('pt5_Xception training and testing')
    train_and_test()
