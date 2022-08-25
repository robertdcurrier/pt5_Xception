#!/usr/bin/env python3
# coding: utf-8
"""
Name:       pt5_image_classifier_standalone
Author:     robertdcurrier@gmail.com
Created:    2022-02-28
Modified:   2022-04-18
Notes:      Decided to make a stand-alone single image classifier. This will
allow me to keep all the 'web centric' code elsewhere, and stick to
classification only in this app.

2022-04-14: TO DO -- implement the new mask code and import most functions
from pt5_utils.py. We will follow the new convention and include only
necessary code here. Most of the functions are now defined in pt5_utils.py:
classify_frame, process_video, process_image, etc. NOTE:  process_images needs
to be moved to pt5_utils.

2022-04-18: Began work on converting to using pt5_utils and mask_me
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Helper libraries
import json
import time
import sys
import os
import logging
import ffmpeg
import argparse
import statistics
import itertools
import glob
import multiprocessing as mp
import cv2 as cv2
from natsort import natsorted
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

# Local utilities
from pt5_utils import (string_to_tuple, get_config, load_scale,
write_frame, mask_me, classify_frame, caption_frame,
calc_cellcount,load_model, check_focus, clean_tmp, process_image_sf)

# Keep TF from yapping incessantly
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array
# globals
thumbs = 0

def get_cli_args():
    """What it say.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2022-02-29

    Notes: Restored edges and contours
    """
    logging.info('get_cli_args()')
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("-o", "--output", help="output file",
                       nargs="?")
    arg_p.add_argument("-i", "--input", help="input file",
                       required='true')
    arg_p.add_argument("-c", "--contours", help="Show contours for testing",
                       action="store_true")
    arg_p.add_argument("-e", "--edges", help="Show edges for testing",
                       action="store_true")

    args = vars(arg_p.parse_args())
    return args


def caption_frame(frame, config, taxa, cell_count, cpL):
    """Put caption on frames. Need to add date of most recent
    video processing date/time and date/time of capture
    """
    logging.info('caption_frame(%s, %s)' % (taxa, cpL))
    the_date = time.strftime('%c')
    # Title
    version_text = "Version: "
    the_text = version_text + config['captions']['title'] + config['captions']['version']
    x_pos = config['captions']['caption_x']
    y_pos = config['captions']['caption_y']
    cap_font_size = config['captions']['cap_font_size']
    cap_font_color = string_to_tuple(config['captions']['cap_font_color'])
    cv2.putText(frame, the_text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size,
                cap_font_color)
    y_pos = y_pos + 20
    # Date/Time
    the_text = "Processed: %s" % the_date
    cv2.putText(frame, the_text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size, cap_font_color)

    # Model
    y_pos = y_pos + 20
    the_text = "Taxa: %s" % taxa
    cv2.putText(frame, the_text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size, cap_font_color)

    # Cell count
    y_pos = y_pos + 20
    the_text = "Cells: %s" % cell_count
    cv2.putText(frame, the_text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size, cap_font_color)

    # CPL
    y_pos = y_pos + 20
    the_text = "Estimated c/L: %d" % (cpL)
    cv2.putText(frame, the_text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size, cap_font_color)
    return frame

def init_app():
    """
    Kick it
    """
    args = get_cli_args()
    file_name = args["input"]
    (taxa, frame, contours) = process_image_sf(file_name)
    logging.info('Found %d %s ROIs' % (len(contours), taxa))
    (frame, matches) = classify_frame(taxa, frame, contours)
    logging.info("%s has %d matches" % (taxa, matches))
    write_frame(taxa, frame)

if __name__ == '__main__':
    tensorflow.keras.backend.set_image_data_format("channels_last")
    logging.basicConfig(level=logging.INFO)
    logging.info('pt5_masked_classifier_standalone...')
    init_app()
