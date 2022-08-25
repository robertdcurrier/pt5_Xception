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

def process_image_sf(input_file):
    """
    Author: robertdcurrier@gmail.com
    Created:    2022-04-18
    Modified:   2022-07-05
    """
    taxa = validate_taxa(input_file)
    config = get_config()
    logging.info('process_image_sf(%s)' % taxa)

    # Open still image
    try:
        frame = cv2.imread(input_file)
    except:
        logging.warning('process_image(): Failed to open %s' % input_file)
        return

    (contours, edges) = gen_cons(taxa, frame)
    logging.debug("process_image_sf(): Cons: %d" % len(contours))
    return(taxa, frame, contours)


def gen_cons(taxa, frame, args):
    """
    Name:	gen_cons
    Author: 	robertdcurrier@gmail.com
    Created:    2022-07-04
    Modified:   2022-07-07
    Notes:      Back to contours and edges. Mask works great in the
                lab with clear water but barfs in the wild. Another negative
                for mask is inability to deal with lighting variations.
    """
    config = get_config()
    # Taxa settings
    edges_min = config['taxa'][taxa]['edges_min']
    edges_max = config['taxa'][taxa]['edges_max']
    logging.debug('gen_cons(%s)' % taxa)
    # Edges
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), cv2.BORDER_WRAP)
    edges = cv2.Canny(blurred, edges_min, edges_max)
    contours, _ = (cv2.findContours(edges, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE))
    return(contours)



def gen_coral(args, target_frame, target_cons):
    """
    Name:       gen_coral
    Author:     robertdcurrier@gmail.com
    Created:    2022-07-11
    Modified:   2022-07-11
    Notes:      Iterates over video looking for frame with max cons. Feeds
    this frame into CORAL. If -d flag write out all intermediate images for
    debugging purposes. Returns circle cons for generating bounding boxes.
    """
    input_file = args["input"]
    taxa = validate_taxa(input_file)
    config = get_config()
    line_thick = config['taxa'][taxa]['poi']['line_thick']
    radius_boost = config['taxa'][taxa]['radius_boost']
    con_color = config['taxa'][taxa]['con_color']
    # raw
    if config['system']['debug']:
        fname = 'results/%s_raw.png' % taxa
        cv2.imwrite(fname, target_frame)
    # cons
    con_frame = target_frame.copy()
    cv2.drawContours(con_frame, target_cons, -1, (0,0,0), 1)
    if config['system']['debug']:
        fname = 'results/%s_cons.png' % taxa
        logging.info('gen_coral(%s): Writing %s' % (taxa,fname))
        cv2.imwrite(fname, con_frame)
        # circles
    circle_frame = target_frame.copy()
    for con in target_cons:
        (x,y), radius = cv2.minEnclosingCircle(con)
        center = (int(x), int(y))
        radius = int(radius+radius_boost)
        cv2.circle(circle_frame, center, radius, (0,0,0), -1)
    if config['system']['debug']:
        fname = 'results/%s_circles.png' % taxa
        logging.info('gen_coral(%s): Writing %s' % (taxa,fname))
        cv2.imwrite(fname, circle_frame)
    # edges of circles -- should we threshold instead?
    edges_min = config['taxa'][taxa]['edges_min']
    edges_max = config['taxa'][taxa]['edges_max']
    thresh_min = config['taxa'][taxa]['thresh_min']
    thresh_max = config['taxa'][taxa]['thresh_max']
    gray  = cv2.cvtColor(circle_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), cv2.BORDER_WRAP)
    threshold = cv2.threshold(blurred, thresh_min, thresh_max,
                              cv2.THRESH_BINARY)[1]
    edges = cv2.Canny(threshold, edges_min, edges_max)

    if config['system']['debug']:
        fname = 'results/%s_circle_edges.png' % taxa
        logging.info('gen_coral(%s): Writing %s' % (taxa,fname))
        cv2.imwrite(fname, edges)
        fname = 'results/%s_threshold.png' % taxa
        logging.info('gen_coral(%s): Writing %s' % (taxa,fname))
        cv2.imwrite(fname, threshold)

    circ_cons, _ = (cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE))
    coral_frame = target_frame.copy()
    cv2.drawContours(coral_frame, circ_cons, -1, (0,255,0), 3)
    if config['system']['debug']:
        fname = 'results/%s_coral.png' % taxa
        logging.info('gen_coral(%s): Writing %s' % (taxa,fname))
        cv2.imwrite(fname, coral_frame)
    return(target_frame, circ_cons)


def gen_bboxes(args, circ_cons):
    """
    Name:       gen_bboxes
    Author:     robertdcurrier@gmail.com
    Created:    2022-07-11
    Modified:   2022-07-11
    Notes:      Iterates over video looking for frame with max cons. Feeds
    this frame into CORAL. If -d flag write out all intermediate images for
    debugging purposes.
    """
    input_file = args["input"]
    taxa = validate_taxa(input_file)
    config = get_config()
    bboxes = []
    good_cons = []
    ncons = len(circ_cons)
    logging.debug('gen_bboxes(%s): %d circ_cons' % (taxa, ncons))
    for con in circ_cons:
        area = cv2.contourArea(con)
        logging.debug('gen_bboxes(%s): Con has area of %d' % (taxa, area))
        if (area > config['taxa'][taxa]['min_con_area'] and area <
            config['taxa'][taxa]['max_con_area']):
            logging.debug('gen_bboxes(%s): Appending con of %d' % (taxa, area))
            good_cons.append(con)
    ncons = len(good_cons)
    logging.debug('gen_bboxes(%s): Using %d good_cons' % (taxa, ncons))

    for con in good_cons:
        rect = cv2.boundingRect(con)
        x1 = rect[0]
        y1 = rect[1]
        x2 = x1+rect[2]
        y2 = y1+rect[3]
        bboxes.append([x1,y1,x2,y2])
    logging.info('gen_bboxes(%s): Found %d ROIs' % (taxa, len(bboxes)))
    return (taxa, bboxes)


def process_video_sf(args):
    """
    Name:       process_video_sf
    Author:     robertdcurrier@gmail.com
    Created:    2022-04-11
    Modified:   2022-07-11
    Notes:      Iterates over video looking for frame with max cons. Feeds
    this frame into CORAL. If -d flag write out all intermediate images for
    debugging purposes.
    """
    input_file = args["input"]
    taxa = validate_taxa(input_file)
    config = get_config()
    file_name = input_file
    logging.info('process_video_sf(%s)' % taxa)

    # Where we store our frames
    frames = []
    target_cons = []
    max_num_cons = 0
    video_file = cv2.VideoCapture(file_name)
    size = (int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    max_frames = (int(video_file.get(cv2.CAP_PROP_FRAME_COUNT)))
    # Loop over frames
    frames_read = 0
    max_cons = 0

    while frames_read < max_frames:
        _, frame = video_file.read()
        (contours) = gen_cons(taxa, frame, args)
        # Keep count of max cons
        logging.debug(("Frame: %d Cons: %d MaxCons: %d" % (frames_read,
                     len(contours), max_num_cons)))
        frames_read += 1
        if len(contours) == 0:
            logging.info('process_video_sf(%s): No contours.' % taxa)
        if len(contours) > max_cons:
            target_cons = contours
            target_frame = frame
            max_cons = len(contours)
    if max_cons == 0:
        target_frame = frame
    logging.info('process_video_sf(): Read %d frames...' % frames_read)
    logging.info('process_video_sf(): %d cons on frame %d' % (max_cons,
                                                              frames_read))
    return(target_frame, target_cons)


def process_video_all(args):
    """Process the sucker and return ALL frames/cons
    Author: robertdcurrier@gmail.com
    Created:    2022-04-14
    Modified:   2022-04-19
    Notes:

    2022-04-14: Renamed as process_video_all, as we need all frames for
    snipper, but process_video only returns frame with max_cons.  Easier to
    add another function than try to add flags and modify this working code.

    2022-04-19: Added -w flag so we can write the mask for testing

    2022-07-12: No flags, just gimme the cons.
    """

    input_file = args["input"]
    taxa = validate_taxa(input_file)
    config = get_config()
    file_name = input_file
    logging.info('process_video_all(%s)' % taxa)

    # Where we store our frames
    frames = []
    cons = []
    video_file = cv2.VideoCapture(file_name)
    size = (int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    max_frames = (int(video_file.get(cv2.CAP_PROP_FRAME_COUNT)))
    # Loop over frames skipping the first few..
    frames_read = 0

    while frames_read < max_frames:
        _, frame = video_file.read()
        contours = gen_cons(taxa, frame, args)
        frames_read += 1
        frames.append(frame)
        cons.append(contours)
    logging.info('process_video_all(): Read %d frames...' % frames_read)
    return (frames, cons)


def write_frame(taxa, frame):
    """
    Name:       write_frame
    Author:     robertdcurrier@gmail.com
    Created:    2022-03-15
    Modified:   2022-03-15
    Notes:      Writes to results folder for stand-alone version
    """
    outfile = './results/%s_results.png' % taxa
    logging.info('write_frame(): Writing %s' % outfile)
    cv2.imwrite(outfile, frame)


def classify_frame(args, taxa, frame, bboxes):
    """Does what it says.

    Author: robertdcurrier@gmail.com
    Created:    2022-02-02
    Modified:   2022-03-09
    Notes: New for pt5_Xception. Major tweaks to get working with
    new cell_snipper code.
    2022-02-14: Got working with img_array vs writing ROI to files.
    2022-03-09: Fixed problem that resulted from PIL using RGBA vs BGRA
    2022-04-15: Now using Moments to determine ROI area
    """
    logging.info('classify_frame(%s)' % taxa)
    config = get_config()
    noclass= config['system']['noclass']
    confidence_index = config['keras']['confidence_index']
    line_thick = config['taxa'][taxa]['poi']['line_thick']
    rect_color = eval((config['taxa'][taxa]['poi']['rect_color']))
    fail_color = eval((config['taxa'][taxa]['poi']['fail_color']))
    all_color =  eval((config['taxa'][taxa]['poi']['all_color']))
    no_focus_color =  eval((config['taxa'][taxa]['poi']['no_focus_color']))
    all_thick = config['taxa'][taxa]['poi']['all_thick']
    y_label_spacer = config['taxa'][taxa]['poi']['y_label_spacer']
    font_size = config['taxa'][taxa]['poi']['font_size']
    model = load_model()
    labels = config['keras']["labels"]
    key_list = list(labels.keys())
    val_list = list(labels.values())
    img_x = config['keras']['img_size_x']
    img_y = config['keras']['img_size_y']
    logging.debug('classify_frame(): Labels: %s' % labels)
    matches = 0
    con_index = 0
    moments = []
    out_frame = frame.copy()

    max_num_cons = len(bboxes)
    logging.info("classify_frame(): Classifying %d ROIs" % max_num_cons)
    for bbox in bboxes:
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        roi = frame[y1:y2,x1:x2]
        logging.debug('classify_frame(): ROI size is %d' % len(roi))
        if len(roi) == 0:
            logging.warning('classify_frame(): Skipping 0 byte ROI')
            continue
        # PIL reads colors differently so we need to invert order
        try:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        except:
            logging.debug('cvtColor failure for %d roi ' % len(roi))
            continue

        # Check focus
        focus = bool(check_focus(taxa, roi))
        if not focus:
            color = no_focus_color
        else:
            color = all_color
        # If we are in noclass mode just mark the ROIs
        if config['system']['noclass']:
            cv2.rectangle(out_frame,(x1,y1),(x2,y2), color, all_thick)
            continue
        else:
            # Now we make the ROI a numpy array
            img_array = Image.fromarray(roi)
            img_array = img_array.resize((img_x, img_y))
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis
            predictions = model.predict(img_array, verbose=0)
            scores = (predictions[0])

            index = 0
            logging.debug('classify_frame(%s): results' % taxa)
            score = scores[index]*100
            score_str = str("%0.2f%%" % score)
            logging.debug(("%s: %0.2f%%" % (key_list[index], score)))
            index = labels[taxa]
            # Check taxa of interest score
            taxa_score = scores[index]*100

            if  taxa_score > confidence_index:
                logging.debug("classify_frame(%s): Match" % taxa)
                score_str = str("%0.2f%% %s" % (taxa_score, taxa))
                cv2.rectangle(out_frame,(x1,y1),(x2,y2), rect_color, line_thick)
                cv2.putText(out_frame, score_str, (x1, y1+y_label_spacer),0,
                            font_size,rect_color)
                matches+=1
            else:
                logging.debug("classify_frame(%s): No Match" % taxa)
                fail_str = str("%0.2f%% %s" % (taxa_score, taxa))
                cv2.rectangle(out_frame,(x1,y1),(x2,y2), fail_color, line_thick)
                cv2.putText(out_frame, fail_str, (x1, y1+y_label_spacer),0,
                            font_size,fail_color)
    if config['system']['noclass']:
        return(out_frame, max_num_cons)
    else:
        return (out_frame, matches)


def caption_frame(frame, taxa, cell_count):
    """Put  on frames. Need to add date of most recent
    video processing date/time and date/time of capture
    """
    config = get_config()

    logging.info('_frame(%s, %s)' % (taxa, cell_count))
    the_date = time.strftime('%c')
    # Title
    title = config['captions']['title']
    x_pos = config['captions']['caption_x']
    y_pos = config['captions']['caption_y']
    cap_font_size = config['captions']['cap_font_size']
    cap_font_color = (255,255,255)
    cv2.putText(frame, title, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size,
                cap_font_color)
    y_pos = y_pos + 20
    # Version
    the_text = "Version: %s" % config['captions']['version']
    cv2.putText(frame, the_text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size, cap_font_color)
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

    if config["system"]["noclass"]:
        # DEBUG WARNING
        y_pos = y_pos + 40
        the_text = "DEBUG MODE"
        cap_font_size = 3
        cap_font_color = (0,0,255)
        cv2.putText(frame, the_text, (x_pos, y_pos),
                    cv2.FONT_HERSHEY_PLAIN, cap_font_size, cap_font_color)
    return frame


def calc_cellcount(cells, taxa):
    """Calculate eCPL based on interpolated scale.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2022-04-28
    """
    logging.info('calc_cellcount(): Using %s' % taxa)
    config = get_config()
    scale = config["taxa"][taxa]["scale"]
    max_cells_cutoff = config["taxa"][taxa]["max_cells_cutoff"]
    if cells == 0:
        cpL = 0
    if cells >= max_cells_cutoff:
        logging.warning('calc_celcount(): Exceeded max_cells_cutoff!')
        cells = max_cells_cutoff-1
        msg = "cells: %d max_cells_cutoff: %d" % (cells, max_cells_cutoff )
        logging.info(msg)
    cpL = scale[cells]
    return cpL

def validate_taxa(input_file):
    """
    Check for alexandrium, karenia or detritus in file name. Assign to same.
    """
    config = get_config()
    labels = config['keras']['labels']
    key_list = list(labels.keys())
    for key in key_list:
        if key in input_file:
            taxa = key
            logging.debug('validate_taxa(): Validated %s' % taxa)
            return taxa

    # No taxa so warn and return
    logging.warning('validate_taxa(): Invalid taxa! Not processing...')
    sys.exit()


def load_model():
    """Load TensorFlow model and cell weights from lib.

    Author: robertdcurrier@gmail.com
    Created:    2019-07-18
    Modified:   2022-02-20
    """
    config = get_config()
    platform = config['system']['platform']
    model_file = config['keras']['platform'][platform]['model_file']
    try:
        model = keras.models.load_model(model_file)
        logging.info('load_model(): Loaded %s' % model_file)
        return model
    except:
        logging.warning('load_model(): FAILED to load %s!' % model_file)
        sys.exit()


def check_focus(taxa, roi):
    """
    Uses Laplacian to select cells in reasonable focus
    """
    global thumbs
    config = get_config()
    flatten = (sum(roi.flatten()))
    if flatten > 0:
        focus = cv2.Laplacian(roi, cv2.CV_64F).var()
        logging.debug('check_focus(%s): Cell has focus %0.2f' %
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

def portal_write_frame(full_file_name, frame):
    """
    """
    logging.info('portal_write_frame()')
    (vol_root, file_name) = os.path.split(full_file_name)
    outfile = "%s/%s" % (vol_root, file_name.replace('raw', 'pro'))
    outfile = outfile.replace('mp4','png')
    logging.info('portal_write_frame(): Writing %s' % outfile)
    try:
        cv2.imwrite(outfile, frame)
    except:
        logging.warning('portal_write_frame(): Failed to write %s' % outfile)


def load_volunteer(serial_number):
    """ Get volunteer metadata from users table.

    Author:     robertdcurrier@gmail.com
    Created:    2019-07-18
    Modified:   2022-03-22
    Notes:      Updated to use HSV2 serial numbers
    """
    logging.info('load_volunteer(%s)' % serial_number)
    client = connect_mongo()
    db = client.habscope2
    try:
        vol_data = db.users.find({"serial" : serial_number})
        return vol_data[0]
    except:
        logging.warning("load_volunteer(): %s not found" % serial_number)
        return False


def connect_mongo():
    """D'oh'.

    Author: robertdcurrier@gmail.com
    Created:    2019-07-18
    Modified:   2022-03-15
    """
    logging.debug('connect_mongo(): creating connection')
    client = MongoClient('mongo:27017')
    return client


def fetch_site(lat, lon):
    """ Get sites from siteCoordinates.

    Author: robertdcurrier@gmail.com
    Created:    2019-07-18
    Modified:   2022-05-05
    """
    client = connect_mongo()
    db = client.habscope2
    results = db.siteCoordinates.find({"coords" : [lat, lon]})
    for result in results:
        return(result['site'])



def insert_record(doc):
    """Insert one record into imageLogs collection.

    Author: robertdcurrier@gmail.com
    Created:    2019-07-18
    Modified:   2022-03-22
    """
    logging.info("insert_record(%s)" % doc)
    client = connect_mongo()
    db = client.habscope2
    result = db.imageLogs.insert_one(doc)
    return result


def build_db_doc(file_name, cells):
    """ Construct user document from file_name metadata.

    Author: robertdcurrier@gmail.com
    Created:    2019-07-18
    Modified:   2022-03-22
    """
    logging.info('build_db_doc(%s, %d)' % (file_name, cells))
    (serial_number, taxa, recorded_ts, lat, lon, _) = file_name.split('_')
    # Convert lon/lat to floats from string
    lon = float(lon)
    lat = float(lat)
    config = get_config()
    volunteer = load_volunteer(serial_number)
    # get all metadata
    doc = {}
    #extract what we want
    timestamp = {'_id' : int(recorded_ts)}
    doc.update(timestamp)
    user_name = {"user_name" : volunteer["user_name"]}
    doc.update(user_name)
    user_email = {"user_email" : volunteer["user_email"]}
    doc.update(user_name)
    user_org = {"user_org" : volunteer["user_org"]}
    doc.update(user_org)
    # TO DO: Get taxa from file instead of DB
    taxa = {"taxa" : taxa}
    doc.update(taxa)
    site = fetch_site(lat, lon)
    site = {"site" : site}
    doc.update(site)
    # Change names to reflect processing status
    videofile = '/data/habscope2/videos/%s/%s' % (serial_number, file_name)
    file_name = file_name.replace('raw', 'pro')
    # The classifier outputs a PNG so we need to change the extension
    imgfile = "/data/habscope2/videos/%s/%s" % (serial_number,
                                                file_name.replace('mp4', 'png'))
    image_name = {"image_name" : imgfile}
    doc.update(image_name)
    video_name = {"video_name" : videofile}
    doc.update(video_name)
    # We add recorded time in this version as HS1 only used processing time
    recorded_ts = {"recorded_ts" : int(recorded_ts)}
    doc.update(recorded_ts)
    processed_ts = int(time.time())
    doc.update({"processed_ts" : processed_ts})
    # GPS coordinates from metadata in file name
    user_gps = {"user_gps" : [lat, lon]}
    doc.update(user_gps)
    analyst = {"analyst" : "Pending"}
    doc.update(analyst)
    status = {"status" : "Pending"}
    doc.update(status)
    cells_manual = {"cells_manual" : 0}
    doc.update(cells_manual)
    cells_habscope = {"cells_habscope" : cells}
    doc.update(cells_habscope)
    # Only do cpL if taxa is Karenia
    if taxa['taxa'] == 'karenia':
        cpL = calc_cellcount(cells, taxa['taxa'])
        cpl_habscope = { "cpl_habscope" : cpL}
        cpl_manual = { "cpl_manual" : 0}
    else:
        cpl_habscope = { "cpl_habscope" : 'N/A'}
        cpl_manual = { "cpl_manual" : 'N/A'}
    doc.update(cpl_manual)
    doc.update(cpl_habscope)

    return doc

if __name__ == '__main__':
    results = fetch_site(28.9312, -89.4429)
    print(results)
