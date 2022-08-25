#!/usr/bin/env python3
# coding: utf-8
"""This is the NVIDIA GPU version.

Modified: 2020-10-30
Beginning integration of upload watcher functionality. Got working. Now using
input, output and taxa args. Will need to add multiprocesing support and do
a map over the args for each taxa.

Modified: 2020-11-03
YEAH!  Compiled OpenCV into the phytotracker4 container. No longer a need
for ffmpeg. No more tmp file name bullshit. Much better way for sure. Also
not pushing source into the phytotracker4 image. We'll use the volume mount
that we use in HABscope. We need to set up a phytotracker4 production  repo
so we don't clobber out dev tree and so that we can make changes in the hsv3
path and still push to the production repo.  Dev should be for big changes,
not minor tweaks.

Modified: 2020-11-04
Incoporated MongoDB functionality into this code as we only need a couple
of methods, not all of the website move/add/delete/update etc.

Modified: 2020-11-05
Got geodata and db writing working. Removed hardwired volunteer name. Added
try: except for volunteer lookups. Think all loopholes closed...

Modified: 2020-11-06
Continued building out videoLogs table. Added defaults for analyst, status,
cpl_count and cpl_manual to watcher code. Removed hsv3_phytotracker4.py and
went with simply coping DEV to phytotracker4.py.  We play with DEV and then
push to phytotracker4, which hsv3 will then copy with update_me.sh

Modified: 2020-11-09
Globals were hangover from HABscope
Removed global MAX_CELLS and started using args of cpl_count, max_cells and
cell_count.  This eliminated the stomping on the globals we were seeing.  Not
sure how elegant this is, but hey, if it works...

Modified: 2020-11-10
Continued to remove globals. Args is now the default data dict for all params
Added passing of args to load_scale, load_model and calc_cellcount.

Fixed cell count not updating on caption_frame.

Modified: 2021-04-06
Testing with HABscope_Pi

Modified: 2021-04-07
Dropped back to 2fps

Modified: 2021-05-17
Restoring -g, -e, -c and -z options. Debugging tools needed to work with new
HABscope_Pi videos. Found that compression totally scrogs the video and
background subtraction no longer works. We will need to fix this or
motion tracking is toast.

Modified: 2021-06-08
Removed all non-essential code. This code will now become phytotracker4,
and phytotracker4_nomo will become phytotracker_dev.  We no longer do background
subtraction, and we use edges to find the contours. We push frames and
contours into buffers and then interate over contours to find the max_index.
We use the max_index to get the frame with the most contours. This frame
then gets sent to target_cells along with the contours for that index. A
.png file is written out after targeting, POI and captions have been applied.
We have been able to reduce run time from 4 minutes to 4 seconds on a very
dense video, and from 20 seconds to 1.8 seconds on a low density video.

Modified: 2021-06-09
Split phytotracker3 and phytotracker4 into two repos. We will use PT3 for lab
work and testing. PT4 will go with HSV3.

Modified: 2021-07-07
Switched to new version of production phytotracker4. No longer keeping all
frames, we do a compare of contours and if contours > max we keep that frame.
If not, we keep going. So we at all times only have one frame in memory.
MUCH MUCH BETTER.

NOTE: For the standalone version we have to have:

    # get settings
    args = get_cli_args()
    input = args["input"]
    output = args["output"]
    taxa = args["taxa"]
    config = get_config()

    in process_video as we don't use passed parameters. We have
    to  change process_video(in, out, vol, taxa) to process_video().

    We also have to remove all db_write instances as there is no db.

SO EACH TIME WE TWEAK PHYTOTRACKER4 process_video we need to copy over
and then apply the above changes to allow for stand-alone mode.


Modified: 2021-07-29
Need to add video mode so we can grab cells from all frames not
just the max_con frame.

Modified: 2021-07-30

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
import multiprocessing as mp
import cv2 as cv2
import pymongo as pymongo
from pymongo import MongoClient


# Keep TF from yapping incessantly
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
# globals
THUMB_COUNT = 0

def string_to_tuple(str):
    """
    Converts string to tuple for use in OpenCV
    drawing functions. Sort of an ass-backswards methond, but...

    Author: robertdcurrier@gmail.com
    Created:    2018-11-07
    Modified:   2018-11-07
    """
    color = []
    tup = map(int, str.split(','))
    for val in tup:
        color.append(val)
        color = tuple(color)
    return color


def draw_poi(frame, x, y, config, taxa, preds):
    """
    New draw_poi for phytotracker3. We lose the dashed lines
    and simply add a cross in the center. All config items
    now come from config.json. 0 means taxa match, 1 no match.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-07
    Modified:   2021-07-22
    Notes: Restored Red Targets for debugging
    """
    if preds == 0:
        # karenia
        rect_color=(0,0,255)
    if preds == 1:
        # alexandrium
        rect_color=(255,0,0)

    # Need to arg so we can show red target indicator for debugging
    w = config['taxa'][taxa]['poi']['width']
    h = config['taxa'][taxa]['poi']['height']
    corner_thick = config['taxa'][taxa]['poi']['corner_thick']
    line_thick = config['taxa'][taxa]['poi']['line_thick']
    dash_length = config['taxa'][taxa]['poi']['dash_length']
    # FAT CORNERS
    # top left
    cv2.line(frame, (x,y),(x,y+dash_length),rect_color,corner_thick)
    cv2.line(frame, (x,y),(x+dash_length,y),rect_color,corner_thick)
    # top right
    cv2.line(frame, (x+w,y),(x+w,y+dash_length),rect_color,corner_thick)
    cv2.line(frame, (x+(w-dash_length),y),(x+w,y),rect_color,corner_thick)
    # bottom left
    cv2.line(frame, (x,y+h),(x,y+(h-dash_length)),rect_color,corner_thick)
    cv2.line(frame, (x,y+h),(x+dash_length,y+h),rect_color,corner_thick)
    # bottom right
    cv2.line(frame, (x+w,y+h),(x+w,y+(h-dash_length)),rect_color,corner_thick)
    cv2.line(frame, (x+(w-dash_length),y+h),(x+w,y+h),rect_color,corner_thick)


def get_config():
    """From config.json.

    Author: robertdcurrier@gmail.com
    Created:    2019-07-18
    Modified:   2019-07-18
    """
    c_file = open('configs/phyto5.cfg').read()
    config = json.loads(c_file)
    return config


def string_to_tuple(the_str):
    """
    Converts string to tuple for use in OpenCV
    drawing functions. Sort of an ass-backswards method, but...

    Author: robertdcurrier@gmail.com
    Created:    2018-11-07
    Modified:   2018-11-07
    """
    color = []
    tup = map(int, the_str.split(','))
    for val in tup:
        color.append(val)
    color = tuple(color)
    return color


def load_model(taxa):
    """Load TensorFlow model and cell weights from lib.

    Author: robertdcurrier@gmail.com
    Created:    2019-07-18
    Modified:   2021-06-08
    """
    config = get_config()
    model_file = config['taxa'][taxa]['model_file']

    try:
        json_file = open(model_file)
        logging.info("load_model(): Loaded %s model_file successfully" % taxa)
    except IOError:
        logging.info("load_model(): Failed to open %s model_file" % model_file)
        sys.exit()
    model_json = json_file.read()
    json_file.close()
    model = tensorflow.keras.models.model_from_json(model_json)
    weights_file = config['taxa'][taxa]['weights_file']
    try:
        model.load_weights(weights_file)
        logging.info("load_model(): Loaded %s weights_file successfully" % taxa)
    except IOError:
        logging.info("load_model(): Failed to open %s" % weights_file)
        sys.exit()
    return model


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
    logging.debug('get_cli_args()')
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("-o", "--output", help="output file",
                       nargs="?")
    arg_p.add_argument("-c", "--contours", help="show contours",
                       action="store_true")
    arg_p.add_argument("-n", "--nopreds", help="Don't classify",
                       action="store_true")
    arg_p.add_argument("-g", "--gui", help="Show image preview",
                       action="store_true")
    arg_p.add_argument("-l", "--learn", help="Cut cells for training",
                       action="store_true")
    arg_p.add_argument("-i", "--input", help="input file",
                       required='true')
    arg_p.add_argument("-t", "--taxa", help="taxa",
                       required='true')
    args = vars(arg_p.parse_args())
    return args


def validate_taxa(args):
    """Confirm taxa is in config file."""
    config = get_config()
    taxa = args["taxa"]
    if config['system']['debug']:
        logging.info("validate_taxa(): Validating %s" % taxa)
    if taxa in config['taxa'].keys():
        logging.info("validate_taxa(): Found %s taxa settings" % taxa)
    else:
        logging.info("validate_taxa(): %s not found!" % taxa)
        sys.exit()

def process_video():
    """Process the sucker.

    Read video input file, gets cons list,
    draws target indicator and updates settings. Skips first n frames
    and last n frames to avoid shaking. N defined in config file

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2021-07-07
    Changed this for pt4_standalone. We don't pass parameters,
    we use args only. We fetch the args here. -i, -o and -t only.
    """
    # local variables
    frame_count = 0
    target_frame = None
    target_cons = None
    max_num_cons = 0

#===============================================================
    # get args for stand-alone-version -- this won't be in
    # production version so when we update we need to insert
    args = get_cli_args()
    input_file = args["input"]
    output_file = args["output"]
    taxa = args["taxa"]
    validate_taxa(args)
    config = get_config()
    file_name = input_file
    model = load_model(taxa)
    scale = load_scale(taxa)
    logging.info('process_video()')
#===============================================================
    video_file = cv2.VideoCapture(file_name)
    size = (int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    max_frames = (int(video_file.get(cv2.CAP_PROP_FRAME_COUNT)))
    skip_frames = config['taxa'][taxa]['skip_frames']
    fps = config['taxa'][taxa]['fps']
    thresh_min = config['taxa'][taxa]['thresh_min']
    thresh_max = config['taxa'][taxa]['thresh_max']
    MOG_thresh = config['taxa'][taxa]['MOG_thresh']
    MOG_history = config['taxa'][taxa]['MOG_history']
    learning_rate = config['taxa'][taxa]['learning_rate']
    con_area_min = config['taxa'][taxa]['con_area_min']
    con_area_max = config['taxa'][taxa]['con_area_max']
    max_cons = config['taxa'][taxa]['max_cons']
    logging.debug("process_video(): max_frames %d" % max_frames)
    logging.debug("process_video(): loading %s" % file_name)
    logging.debug("process_video(): Skipping %d frames" % skip_frames)
    logging.debug("process_video(): thresh_min %d" % thresh_min)
    logging.debug("process_video(): thresh_max %d" % thresh_max)
    logging.debug("process_video(): MOG_history %d" % MOG_history)
    logging.debug("process_video(): MOG_threshold %d" % MOG_thresh)
    logging.debug("process_video(): con_area_min %d" % con_area_min)
    logging.debug("process_video(): con_area_max %d" % con_area_max)
    logging.debug("process_video(): learning_rate %0.6f" % learning_rate)
    # Loop over frames skipping the first few...
    frame_count += 1
    while frame_count <= skip_frames:
        _, frame = video_file.read()
        frame_count += 1

    while frame_count < (max_frames - skip_frames):
        _, frame = video_file.read()
        # Equalize histogram and get contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3,3), cv2.BORDER_WRAP)
        edges = cv2.Canny(blurred, config["taxa"][taxa]["edges_min"],
                          config["taxa"][taxa]["edges_max"],3)
        contours, _ = (cv2.findContours(edges, cv2.RETR_EXTERNAL,
                       cv2.CHAIN_APPROX_NONE))

        # Now we use the much better algo of only keeping the max frame
        # vs stuffing all of them and testing postprocess. D'oh
        if len(contours) > max_num_cons:
            target_cons = contours
            max_num_cons = len(contours)
            target_frame = frame
        #
        frame_count += 1
    if max_num_cons == 0:
        logging.warning('NO CONTOURS FOUND. ABORTING...')
        return

    caption_frame, cpL, cells = target_cells(target_frame, target_cons, taxa)
    logging.info('Writing %s' % output_file)
    con_color = config['taxa'][taxa]['con_color']
    con_color = string_to_tuple(con_color)
    if args["contours"]:
        logging.info('Showing contours')
        cv2.drawContours(caption_frame, target_cons, -1, con_color, 1)
    if args["output"]:
        cv2.imwrite(output_file, caption_frame)
    if args["gui"]:
        cv2.imshow('Preview', caption_frame)
        cv2.waitKey(0)
    # For scale generator


def classify_cell(model, cell):
    """What it say.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2021-09-23

    We changed this up a bit... we are now passing buffers,
    not writing to file for cell images. We reshape to
    40, 40 to match the model. If mode.predict returns none for
    some reason we return a 2 so we can ignore or flag. Taxa
    returns 0 and Not Taxa returns a 1.
    Notes: Now using a much cleaner method of prepping cell for classification
    """
    cell = cv2.resize(cell, (40,40))
    logging.debug('classify_cell(%s)' % model)
    img = img_to_array(cell)
    img = img.reshape(1, 40, 40, 3)
    preds = model.predict(img)
    return preds


def target_cells(frame, contours, taxa):
    """Draw 'Person of Interest' target around detected cells.

    if TensorFlow model returns True. If False then draw
    a red rectangle if '-a' option set.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2021-07-23
    """
    logging.info('target_cells(%s)' % taxa)
    model = load_model(taxa)
    config = get_config()
    args = get_cli_args()

    x_offset = config['taxa'][taxa]['x_offset']
    y_offset = config['taxa'][taxa]['y_offset']
    roi_offset = config['taxa'][taxa]['roi_offset']
    cell_count = 0
    con_num = 0
    config = get_config()
    con_color = config['taxa'][taxa]['con_color']
    con_color = string_to_tuple(con_color)
    cx = 0
    cy = 0
    old_cx = 0
    old_cy = 0
    moments=[]
    for con in contours:
        con = cv2.convexHull(con)
        M = cv2.moments(con)
        if M['m00'] > 0 and M['m10'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            area = cv2.contourArea(con)
            if area > config['taxa'][taxa]['con_area_min']:
                moments.append([cx,cy])

    # Sort and remove dupes
    moments.sort()
    list(moments for moments,_ in itertools.groupby(moments))
    for moment in moments:
        cx = moment[0]
        cy = moment[1]
        if cx > 0 and cy > 0:
            if abs(cx-old_cx) > 10 or abs(cy-old_cy) > 10:
                old_cx = cx
                old_cy = cy
                roi = frame[cy-y_offset:cy+y_offset,cx-x_offset:cx+x_offset]
                if args["learn"]:
                    epoch = time.time_ns()
                    img_name = "tmp/%d_%s.png" % (epoch, args["taxa"])
                    logging.info("target_cells(): Writing image %s" % (img_name))
                    cv2.imwrite(img_name, roi)
                else:
                    shape = roi.shape
                    if shape[0] > 0 and shape[1] > 0:
                        preds = classify_cell(model, roi)
                        preds=preds[0]
                        print(preds)
                        print(
                                "This image is %.2f percent Brevis and %.2f percent Alexandrium."
                                % (100 * (1 - preds), 100 * preds)
                        )
                        """
                        if preds[0][0] == 0:
                            # karenia
                            cv2.rectangle(frame,(cx-x_offset, cy-y_offset),
                                            (cx+x_offset, cy+y_offset),(0,0,255),2)
                            cell_count += 1
                        if preds[0][0] == 1:
                            # alexandrium
                            cv2.rectangle(frame,(cx-x_offset, cy-y_offset),
                                            (cx+x_offset, cy+y_offset),(255,0,0),2)
                        """

    logging.info('target_cells(): %d %s cells' % (cell_count, taxa))
    cpL = calc_cellcount(cell_count, taxa)
    logging.info('target_cells(): %d %s c/L' % (cpL, taxa))
    frame = caption_frame(frame, config, taxa, cell_count, cpL)
    return frame, cpL, cell_count



def caption_frame(frame, config, taxa, cell_count, cpL):
    """Put caption on frames. Need to add date of most recent
    video processing date/time and date/time of capture
    """
    logging.debug('caption_frame(%s, %s)' % (taxa, cpL))
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


if __name__ == '__main__':
    tensorflow.keras.backend.set_image_data_format("channels_last")
    logging.basicConfig(level=logging.INFO)
    logging.info('Phytotracker5 multi-class version...')
    process_video()
