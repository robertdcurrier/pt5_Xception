#!/usr/bin/env python3
# coding: utf-8
"""
Name:       pt5_image_classifier_standalone
Author:     robertdcurrier@gmail.com
Created:    2022-03-02
Notes:     This is the web portal version with all the MongoDB and volunteer
functions needed for site locations, db inserts, etc.

2022-04-14: TO DO -- implement the new mask code and import most functions
from pt5_utils.py. We will follow the new convention and include only
necessary code here: watchdog, MongoDB, etc.  Most of the functions are now
defined in pt5_utils.py: classify_frame, process_video, process_image, etc.
NOTE:  process_images needs to be moved to pt5_utils.

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
import shutil
from datetime import datetime
import pymongo as pymongo
from pymongo import MongoClient
import multiprocessing as mp
import cv2 as cv2
from natsort import natsorted
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from pt5_process_video import classify_frame
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

# Keep TF from yapping incessantly
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array



class hsImageHandler(PatternMatchingEventHandler):
    def on_created(self,event):
        logging.info(event)
        # Need to check file length in loop so we can tell when upload finished
        size_past = -1
        index = 0
        while True:
            size_now = os.path.getsize(event.src_path)
            logging.info('pt5_Xception: size_past: %d size_now: %d index: %d' %
                         (size_past, size_now, index))
            # Check size and add a five second wait for the laggies
            if size_now - size_past == 0:
                # Wait five seconds to be sure
                logging.info('pt5_Xception(): Sleeping for five seconds...')
                time.sleep(5)
                size_now = os.path.getsize(event.src_path)
                if size_now - size_past == 0:
                    logging.info("File upload complete")
                    # We are looking for cells and cpL in results
                    cells = process_image(event)
                    (_, file_name) = os.path.split(event.src_path)
                    doc = build_db_doc(file_name, cells)
                    insert_record(doc)
                    break
            else:
                index+=1
                size_past = size_now
                logging.info('pt5_Xception(): Upload in progress...')
                time.sleep(1)

def pt5_watchdog():
    """Turn on watchdog for file processing."""
    config = get_config()
    file_path = config['system']['watch_dir']
    watch_type = config['system']['watch_type']
    logging.info('pt5_watchdog(): Monitoring %s for %s' % (file_path, watch_type))
    observer = Observer()
    event_handler = hsImageHandler(patterns=[watch_type])
    observer.schedule(event_handler, file_path, recursive=True)
    observer.start()
    observer.join()


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


def process_image(event):
    """Process the sucker.

    Read PNG file, gets cons list,
    draws target indicator and updates settings. Skips first n frames
    and last n frames to avoid shaking. N defined in config file

    Author:     robertdcurrier@gmail.com
    Created:    2018-02-28
    Modified:   2022-03-25
    Notes:      Modded the process_video code to deal with a single frame.
    """
    # local variables
    target_frame = None
    target_cons = None
    max_num_cons = 0
    input_file = event.src_path

    if not 'png' in input_file:
        logging.warning('process_image(): Invalid file type. PNG only!')
        return
    config = get_config()
    file_name = event.src_path
    taxa = validate_taxa(file_name)
    logging.info('process_image(%s)' % taxa)

    # Where we store our good cons
    good_cons = []

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

    # Open still image
    try:
        frame = cv2.imread(input_file)
    except:
        logging.warning('process_image(): Failed to open %s' % input_file)
        return
    # Equalize histogram and get contours
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1,1), cv2.BORDER_WRAP)
    edges = cv2.Canny(blurred, edges_min, edges_max)
    contours, _ = (cv2.findContours(edges, cv2.RETR_EXTERNAL,
                      cv2.CHAIN_APPROX_NONE))

    # Now we use the much better algo of only keeping the max frame
    # vs stuffing all of them and testing postprocess. D'oh
    logging.info("process_image(): Cons: %d" % len(contours))
    if len(contours) == 0:
        logging.warning('process_image(): NO CONTOURS FOUND. ABORTING...')
        return

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
    # Write 'em out'
    ncons = len(good_cons)
    logging.info('process_video(%s): Found %d good contours' %(taxa, ncons))
    # if no cons just write frame so we can retrieve and show to volunteer
    if ncons > 0:
        # We need to get cells back from classify_frame so we can do cpL calc
        (frame, cells) = classify_frame(taxa, frame, good_cons)
        portal_write_frame(file_name, frame)
    else:
        logging.warning('process_image(): No cons found.')
        portal_write_frame(file_name, frame)
    return cells

def portal_write_frame(full_file_name, frame):
    """
    """
    logging.info('portal_write_frame()')
    (vol_root, file_name) = os.path.split(full_file_name)
    outfile = "%s/%s" % (vol_root, file_name.replace('raw', 'pro'))
    logging.info('write_frame(): Writing %s' % outfile)
    try:
        cv2.imwrite(outfile, frame)
    except:
        logging.warning('portal_write_frame(): Failed to write %s' % outfile)


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
    Modified:   2022-03-15
    """
    client = connect_mongo()
    db = client.habscope2
    results = (db.siteCoordinates.find({}))
    for result in results:
        return result


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

    volunteer = load_volunteer(serial_number)
    # get all metadata
    doc = {}
    #extract what we want
    timestamp = {'_id' : recorded_ts}
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
    site = {"site" : "N/A"}
    doc.update(site)
    outfile = "/data/habscope2/images/%s/%s" % (serial_number,
                                                file_name.replace('raw', 'pro'))
    file_name = {"file_name" : outfile}
    doc.update(file_name)
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
    cells = { "cells" : cells}
    doc.update(cells)
    cpl_habscope = { "cpl_habscope" : 0}
    doc.update(cpl_habscope)
    cpl_manual = { "cpl_manual" : 0}
    doc.update(cpl_manual)
    return doc


if __name__ == '__main__':
    tensorflow.keras.backend.set_image_data_format("channels_last")
    logging.basicConfig(level=logging.INFO)
    logging.info('pt5_image_classifier_web initializing...')

    pt5_watchdog()
