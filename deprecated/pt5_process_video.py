#!/usr/bin/env python3
# coding: utf-8
"""
Name:       pt5_process_video
Author:     robertdcurrier@gmail.com
Created:    2022-01-31
Notes:      Starting from scratch as we are now using Xception net and
            three classes: alexandrium, brevis and detritus. Testing and
            training are going well, so we need to integrate the
            classification code in test_and_train into this app to process
            HABscope videos.

            Note: We are going to test NOT using the ROI filtering but rather
            sending ALL ROIs to the classifier.  Not sure what's going to
            happen but it would greatly simplify the code.

2022-02-10: Added check_focus. We are using ROI filtering once again. Moved
edges and contours to hablab.  Need to start working on adding the
web components. We should use a -w option to signify web deployment. We will
have to bring in all the user auth code and pymongo code from pt4.  I was
hoping to not have everything in one app, but it gets too complicated to keep
two versions in sync. If I make a change in process_video I don't want to
have to remember to make the same change somewhere else.  So we have moved all
the hablab functions OUT of this code, so its only purpose is to process the
video and produce an image with the results. If running with -w true then all
the website stuff will need to happen. DB inserts, etc.

2022-03-09: Had a bit of bother with using the ROI vs writing to file and
loading the image. Turns out PIL used RGBA vs BGRA so we were getting corrupt
image data as far as the classifier was concerned. Writing the ROI to file
showed it as a blue image,and this evidently whacked out the classifier. We now
use cv2 to do the conversion and so far all is working great.  We also
eliminated classify_frame() from pt5_image_classifier_standalone and
pt5_image_classifier_web and just do an import from here. Now we will always
by in synch and not risk having three slightly different versions.
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
import numpy as np
import pymongo as pymongo
from pymongo import MongoClient
from natsort import natsorted
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageOps
from keras.preprocessing import image


# Keep TF from yapping incessantly
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array
# globals
thumbs = 0


def string_to_tuple(str):
    """
    Converts string to tuple for use in OpenCV
    drawing functions. Sort of an ass-backswards methond, but...

    Author: robertdcurrier@gmail.com
    Created:    2018-11-07
    Modified:   2018-11-07
    """
    logging.info('string_to_tuple(%s)' % str)
    color = []
    tup = map(int, str.split(','))
    for val in tup:
        color.append(val)
        color = tuple(color)
    return color

def get_config():
    """From config.json.

    Author: robertdcurrier@gmail.com
    Created:    2019-07-18
    Modified:   2019-07-18
    """
    c_file = open('configs/pt5_Xception.cfg').read()
    config = json.loads(c_file)
    return config


def load_scale(taxa):
    """What it say.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2021-06-08
    """
    logging.info('load_scale(): Using %s' % taxa)
    config = get_config()
    scale_file = config['taxa'][taxa]['scale_file']

    try:
        scale_file = open(scale_file)
    except IOError:
        logging.info("load_scale(): Failed to open %s" % scale_file)
        sys.exit()

    scale = json.loads(scale_file.read())
    scale_file.close()
    logging.info("Loaded %s scale successfully" % taxa)
    return scale


def get_cli_args():
    """What it say.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2021-05-16

    Notes: We want to retore -g, edges and contours
    """
    logging.info('get_cli_args()')
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("-o", "--output", help="output file",
                       nargs="?")
    arg_p.add_argument("-r", "--raw", help="write raw frame",
                       action='store_true')
    arg_p.add_argument("-i", "--input", help="input file",
                       required='true')
    arg_p.add_argument("-m", "--mask", help="show mask",
                       action='store_true')
    args = vars(arg_p.parse_args())
    return args


def process_video():
    """Process the sucker.

    Read video input file, gets cons list,
    draws target indicator and updates settings. Skips first n frames
    and last n frames to avoid shaking. N defined in config file

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2022-02-09
    Notes:

    Now working on pt5_process_video using Xception network.  We need
    taxa to check for min/max cell size and ROI spacing but we no longer
    pass as an arg. The input file will contain taxa by default so we
    extract from there, thus allowing us to not specify taxa for the
    actual classifcation task, but still use all the pre-classifcation
    screening methods to winnow down the list.
    """
    global thumbs
    # local variables
    frame_count = 0
    target_frame = None
    target_cons = None
    max_num_cons = 0

    args = get_cli_args()
    input_file = args["input"]
    taxa = validate_taxa(input_file)
    config = get_config()
    file_name = input_file
    logging.info('process_video(%s)' % taxa)

    # Where we store our good cons
    good_cons = []
    video_file = cv2.VideoCapture(file_name)
    size = (int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    max_frames = (int(video_file.get(cv2.CAP_PROP_FRAME_COUNT)))
    skip_frames = config['cv2']['skip_frames']
    skip_snip_frames = config['cv2']['skip_snip_frames']
    fps = config['cv2']['fps']
    # Taxa-specific descriptors
    edges_min = config['taxa'][taxa]['edges_min']
    edges_max = config['taxa'][taxa]['edges_max']
    roi_spacing = config['taxa'][taxa]['roi_spacing']
    cell_width_min = config['taxa'][taxa]['cell_width_min']
    cell_width_max = config['taxa'][taxa]['cell_width_max']
    cell_height_min = config['taxa'][taxa]['cell_height_min']
    cell_height_max = config['taxa'][taxa]['cell_height_max']
    line_thick = config['taxa'][taxa]['poi']['line_thick']
    rect_color = eval(config['taxa'][taxa]['poi']['rect_color'])
    x_offset = config['taxa'][taxa]['x_offset']
    y_offset = config['taxa'][taxa]['y_offset']
    poi_w = config['taxa'][taxa]['poi']['width']
    poi_h = config['taxa'][taxa]['poi']['height']

    # Loop over frames skipping the first few...
    frame_count += 1
    frames_read = 0
    thumbs = 0

    logging.info('process_video(): Read %d frames...' % max_frames)
    while frame_count <= skip_frames:
        _, frame = video_file.read()
        frame_count += 1

    while frame_count < (max_frames - skip_frames):
        _, frame = video_file.read()
        # Equalize histogram and get contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3,3), cv2.BORDER_WRAP)
        edges = cv2.Canny(blurred, edges_min, edges_max)

        contours, _ = (cv2.findContours(edges, cv2.RETR_EXTERNAL,
                       cv2.CHAIN_APPROX_NONE))
        old_x = 0
        old_y = 0
        for con in contours:
            rect = cv2.boundingRect(con)
            x,y,w,h = rect
            # Screen for likely candidatesc
            if ((w > cell_width_min) and (w < cell_width_max) and
                (h > cell_height_min) and (h < cell_height_max)):
                if abs(x-old_x) > roi_spacing or abs(y-old_y) > roi_spacing:
                    old_x = x
                    old_y = y
                    good_cons.append(con)
        ncons = len(good_cons)
        logging.debug(('process_video(%s): Frame %d: %d contours' %
                     (taxa, frame_count, ncons)))
        logging.debug(("Frame: %d Cons: %d MaxCons: %d" % (frame_count,
                      len(contours), max_num_cons)))
        if len(contours) > max_num_cons:
            target_cons = contours
            target_frame = frame
            edges_frame = edges
            max_num_cons = len(contours)
        frame_count += 1
    if max_num_cons == 0:
        logging.warning('NO CONTOURS FOUND. ABORTING...')
        return
    else:
        logging.info('process_video(): Found %d cons' % max_num_cons)

    if args["raw"]:
        logging.info('process_video(): Writing raw frame')
        fname = 'results/%s_raw.png' % taxa
        cv2.imwrite(fname, target_frame)

    old_x = 0
    old_y = 0
    good_cons = []

    # Filter out bad cons
    for con in target_cons:
        rect = cv2.boundingRect(con)
        x,y,w,h = rect
        # minimal checking
        if (w > cell_width_min):
            if abs(x-old_x) > roi_spacing and abs(y-old_y) > roi_spacing:
                # compute the center of the cons
                M = cv2.moments(con)
                if M["m10"] > 0 and M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    good_cons.append(con)
                old_x = x
                old_y = y

    ncons = len(good_cons)
    logging.debug('process_video(%s): Found %d contours' %(taxa, ncons))
    if ncons > 0:
        if args["mask"]:
            mask_me(taxa, target_frame)
            #show_all_cons(taxa, target_cons, target_frame, edges)
            #logging.info('show_all_cons(): Exiting...')
            sys.exit()
        (frame, matches) = classify_frame(taxa, target_frame, good_cons)
        logging.info('classify_frame(): Calculating cpL for %s with %d cells' %
                     (taxa, matches))
        write_frame(taxa, frame)
    else:
        logging.warning('classify_frame(): No cons found!')


def write_frame(taxa, frame):
    """
    Name:       write_frame
    Author:     robertdcurrier@gmail.com
    Created:    2022-03-15
    Modified:   2022-03-15
    Notes:      Writes to results folder for stand-alone version
    """
    outfile = 'results/%s_results.png' % taxa
    logging.info('write_frame(): Writing %s' % outfile)
    cv2.imwrite(outfile, frame)

def mask_me(taxa, target_frame):
    """
    """
    # The kernel to be used for dilation purpose
    kernel = np.ones((5, 5), np.uint8)

    # converting the image to HSV format
    hsv = cv2.cvtColor(target_frame, cv2.COLOR_BGR2HSV)

    # defining the lower and upper values of HSV,
    # this will detect yellow colour
    Lower_hsv = np.array([50, 10, 50])
    Upper_hsv = np.array([150, 255, 255])

    # creating the mask by eroding,morphing,
    # dilating process
    Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
    Mask = cv2.erode(Mask, kernel, iterations=1)
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
    Mask = cv2.dilate(Mask, kernel, iterations=1)

    # Inverting the mask by
    # performing bitwise-not operation
    Mask = cv2.bitwise_not(Mask)
    outfile = 'results/%s_masked.png' % taxa
    logging.info('mask_me(): Writing %s' % outfile)
    cv2.imwrite(outfile, Mask)
    return Mask

def show_all_cons(taxa, cons, target_frame, edges):
    """
    Name:       show_all_cons
    Author:     robertdcurrier@gmail.com
    Created:    2022-03-10
    Modified:   2022-03-10
    Notes:      Shows all cons so we can use for debugging
    """
    config = get_config()
    # Taxa-specific descriptors
    edges_min = config['taxa'][taxa]['edges_min']
    edges_max = config['taxa'][taxa]['edges_max']
    roi_spacing = config['taxa'][taxa]['roi_spacing']
    cell_width_min = config['taxa'][taxa]['cell_width_min']
    cell_width_max = config['taxa'][taxa]['cell_width_max']
    cell_height_min = config['taxa'][taxa]['cell_height_min']
    cell_height_max = config['taxa'][taxa]['cell_height_max']
    line_thick = config['taxa'][taxa]['poi']['line_thick']
    rect_color = eval(config['taxa'][taxa]['poi']['rect_color'])
    x_offset = config['taxa'][taxa]['x_offset']
    y_offset = config['taxa'][taxa]['y_offset']
    poi_w = config['taxa'][taxa]['poi']['width']
    poi_h = config['taxa'][taxa]['poi']['height']
    y_label_spacer = config['taxa'][taxa]['poi']['y_label_spacer']
    old_cX = 0
    old_cY = 0
    old_x = 0
    old_y = 0
    good_cons = []

    cv2.drawContours(target_frame, cons, -1, (0, 255, 0), 2)
    for con in cons:
        rect = cv2.boundingRect(con)
        x,y,w,h = rect
        if abs(x-old_x) > roi_spacing and abs(y-old_y) > roi_spacing:
            # compute the center of the cons
            M = cv2.moments(con)
            if M["m10"] > 0 and M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                good_cons.append(con)
            old_x = x
            old_y = y

    for con in good_cons:
        rect = cv2.boundingRect(con)
        x,y,w,h = rect
        M = cv2.moments(con)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        vals = "%d,%d:%d,%d" % (x,y,w,h)
        (cv2.rectangle(target_frame,(cX-poi_w,cY-poi_h),
        (cX+poi_w,cY+poi_h),(0,0,0),line_thick))

    ncons = len(good_cons)
    logging.debug('show_all_cons(%s): Found %d contours' %(taxa, ncons))
    if ncons > 0:
        # Write frame out for testing with pt5_classify_image
        fname = 'results/%s_cons.png' % taxa
        cv2.imwrite(fname, target_frame)
        fname = 'results/%s_edges.png' % taxa
        cv2.imwrite(fname, edges)
    else:
        logging.warning('show_all_cons(): No cons found!')


def classify_frame(taxa, frame, good_cons):
    """Does what it says.

    Author: robertdcurrier@gmail.com
    Created:    2022-02-02
    Modified:   2022-03-09
    Notes: New for pt5_Xception. Major tweaks to get working with
    new cell_snipper code.
    2022-02-14: Got working with img_array vs writing ROI to files.
    2022-03-09: Fixed problem that resulted from PIL using RGBA vs BGRA
    """
    logging.info('classify_frame(%s)' % taxa)
    config = get_config()
    confidence_index = config['keras']['confidence_index']
    line_thick = config['taxa'][taxa]['poi']['line_thick']
    rect_color = eval((config['taxa'][taxa]['poi']['rect_color']))
    fail_color = eval((config['taxa'][taxa]['poi']['fail_color']))
    x_offset = config['taxa'][taxa]['x_offset']
    y_offset = config['taxa'][taxa]['y_offset']
    y_label_spacer = config['taxa'][taxa]['poi']['y_label_spacer']
    num_classes = config["keras"]["num_classes"]
    poi_w = config['taxa'][taxa]['poi']['width']
    poi_h = config['taxa'][taxa]['poi']['height']
    font_size = config['taxa'][taxa]['poi']['font_size']
    model = load_model()
    img_x = int(config["keras"]["img_size_x"])
    img_y = int(config["keras"]["img_size_y"])
    image_size = (img_x, img_y)
    labels = config['keras']["labels"]
    key_list = list(labels.keys())
    val_list = list(labels.values())
    logging.debug('classify_frame(): Labels: %s' % labels)
    matches = 0

    max_num_cons = len(good_cons)
    logging.info("classify_frame(): Classifying %d good cons" % max_num_cons)
    for con in good_cons:
        rect = cv2.boundingRect(con)
        x,y,w,h = rect
        image_size = (img_x, img_y)
        if x > x_offset and y > y_offset:
            roi = frame[y-h:y+h, x-w:x+w]
            # PIL reads colors differently so we need to invert order
            if len(roi) == 0:
                logging.info('classify_frame(): Skipping ROI of 0')
                continue
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            # Now we make the ROI a numpy array
            img_array = Image.fromarray(roi)
            img_array = img_array.resize((img_x, img_y))
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis
            predictions = model.predict(img_array)
            scores = (predictions[0])

            index = 0
            logging.debug('classify_frame(): ROI results:')
            for s in scores:
                score = scores[index]*100
                score_str = str("%0.2f%%" % score)
                logging.debug(("%s: %0.2f%%" % (key_list[index], score)))
                index+=1

            index = labels[taxa]
            # Check taxa of interest score
            taxa_score = scores[index]*100
            if  taxa_score > confidence_index:
                logging.debug("classify_frame(): Match for %s" % taxa)
                matches += 1
                score_str = str("%0.2f%%" % taxa_score)
                rect = cv2.boundingRect(con)
                M = cv2.moments(con)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                (cv2.rectangle(frame,(x-w-x_offset,y-h-y_offset),
                (x+w+x_offset,y+h+y_offset),rect_color,line_thick))
                cv2.putText(frame, score_str, (cX-poi_w,
                            cY+poi_h+y_label_spacer),
                            0,font_size,rect_color)
            else:
                logging.debug("classify_frame(): No Match for %s" % taxa)
                score_str = str("%0.2f%%" % taxa_score)
                (cv2.rectangle(frame,(x-w-x_offset,y-h-y_offset),
                (x+w+x_offset,y+h+y_offset),fail_color,line_thick))
                cv2.putText(frame, score_str, (x-poi_w,
                            y+h+y_label_spacer),
                            0,font_size,fail_color)

    # Change to returning frame vs writing here. Web version will have
    # different naming conventions, so we need to be able to deal with this
    logging.info('classify_frame(): Found %d %s cells' % (matches, taxa))
    return (frame, matches)


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


def calc_cellcount(cells, taxa):
    """Calculate eCPL based on interpolated scale.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2021-07-23
    """
    logging.info('calc_cellcount(): Using %s' % taxa)
    config = get_config()
    max_cells_cutoff = config["taxa"][taxa]["max_cells_cutoff"]
    scale = load_scale(taxa)
    if cells == 0:
        cpL = 0
    if cells >= max_cells_cutoff:
        logging.warning('calc_celcount(): Exceeded max_cells_cutoff!')
        cells = max_cells_cutoff-1
        msg = "cells: %d max_cells_cutoff: %d" % (cells, max_cells_cutoff )
        logging.info(msg)
    cpL = scale['scale'][cells]
    return cpL

def validate_taxa(input_file):
    """
    Check for alexandrium, karenia or detritus in file name. Assign to same.
    """
    logging.info('validate_taxa()')
    config = get_config()
    labels = config['keras']['labels']
    key_list = list(labels.keys())
    for key in key_list:
        if key in input_file:
            taxa = key
            logging.info('validate_taxa(): Validated %s' % taxa)
            return taxa

    # No taxa so warn and return
    logging.warning('validate_taxa(): Invalid taxa! Not processing...')


def load_model():
    """Load TensorFlow model and cell weights from lib.

    Author: robertdcurrier@gmail.com
    Created:    2019-07-18
    Modified:   2022-02-02
    """
    logging.info('load_model()...')
    config = get_config()
    model_file = config['keras']['model_file']
    try:
        model = keras.models.load_model(model_file)
        logging.info('load_model(): Loaded %s' % model_file)
        return model
    except:
        logging.warning('load_model(): FAILED to load %s!' % model_file)
        sys.exit()


def check_focus(roi):
    """
    Uses Laplacian to select cells in reasonable focus
    """
    global thumbs
    config = get_config()
    flatten = (sum(roi.flatten()))
    if flatten > 0:
        focus = cv2.Laplacian(roi, cv2.CV_64F).var()
        logging.info('check_focus(%s): Cell has focus %0.2f' %
                     (taxa, focus))
        if focus > config['taxa'][taxa]['focus']:
            thumbs+=1
            return True
        else:
            return False


def clean_tmp():
    """
    Empties all the folders in tmp before running
    """
    logging.info('clean_tmp(): Emptying tmp')
    os.system('rm -rf tmp')
    os.mkdir('tmp')


if __name__ == '__main__':
    tensorflow.keras.backend.set_image_data_format("channels_last")
    logging.basicConfig(level=logging.INFO)
    logging.info('Initializing pt5_Xception...')
    clean_tmp()
    process_video()
