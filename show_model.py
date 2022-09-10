#!/usr/bin/env python3
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
import numpy as np
import pymongo as pymongo
from pymongo import MongoClient
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageOps
from keras.preprocessing import image
# Local utilities
from pt5_utils import (string_to_tuple, get_config, load_scale,
process_video_sf, write_frame, gen_coral,  classify_frame,
gen_bboxes, caption_frame, calc_cellcount,load_model, check_focus, clean_tmp)

model = load_model()
print(model.summary())
