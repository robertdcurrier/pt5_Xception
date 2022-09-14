#!/usr/bin/env python3
# coding: utf-8
"""
Name:       pt5_coral_snipper
Author:     robertdcurrier@gmail.com
Created:    2022-04-14
Modified:   2022-09-13
Notes:      Now using CORAL to determine ROIs. We will make this operate
like a self hosting compiler. We will include an -a flag to get all ROIS.
This will allow us to take an unknown taxa and manual build an annotated
library large enough for a first training. Once we get results > 50% we
can then run without the -a flag, allowing the AI to only select cells that
pop positive.  This will eliminate most of the junk and greatly speed up
the building of a massive training set.
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
import numpy as np
import multiprocessing as mp
import cv2 as cv2
# Local utilities
from pt5_utils import (string_to_tuple, get_config, load_scale,
get_all_frames, write_frame, gen_coral,  classify_frame, validate_taxa,
gen_bboxes, caption_frame, calc_cellcount, load_model, check_focus,
clean_tmp, gen_cons)

thumbs = 0


def get_cli_args():
    """What it say.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2022-02-10

    Notes: Added edges and contour options as we work on hablab
    """
    logging.debug('get_cli_args()')
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("-i", "--input", help="input file",
                       required='true')
    arg_p.add_argument("-c", "--classifier", help="Use classifier",
                       action='store_true')
    args = vars(arg_p.parse_args())
    return args


def cell_snipper(frame, bboxes):
    """
    Name:       cell_snipper
    Author:     robertdcurrier@gmail.com
    Created:    2022-02-01
    Modified:   2022-07-12
    Notes: Chops out cells from frame for use in training

    2022-07-12: Migrated to CORAL. Letting her do all the hard work and just
    pass us bboxes for snipping. Eliminates a fuck ton of code here.
    """
    args = get_cli_args()
    config = get_config()
    taxa = validate_taxa(args["input"])
    logging.info('cell_snipper(%s)' % taxa)
    # Set our count to 0 thumbs
    thumbs = 0

    thumb_count = 0
    for bbox in bboxes:
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        width = x2-x1
        height = y2-y1
        if width > 10 and height > 10:
            roi = frame[y1:y2,x1:x2]
            epoch_time = int(time.time_ns())
            fname = 'tmp/%d_%s.png' %(epoch_time, taxa)
            logging.info('cell_snipper(%s): Writing %s' % (taxa, fname))
            cv2.imwrite(fname, roi)
            thumb_count+=1
    return thumb_count


def get_rois(frame):
    """
    Name:       get_rois
    Author:     robertdcurrier@gmail.com
    Created:    2022-09-12
    Modified:   2022-09-12
    Notes:      Uses new CORAL to id areas of interest for snipping
    """
    args = get_cli_args()
    taxa = validate_taxa(args["input"])
    (contours) = gen_cons(taxa, frame, args)
    circ_cons = gen_coral(frame, contours, args)
    bboxes = gen_bboxes(circ_cons, args)
    # Keep count of max cons
    logging.info("get_rois(): %d bboxes" % len(bboxes))
    if len(bboxes) == 0:
        logging.warning('process_video(%s): No contours.' % taxa)
    return bboxes


def the_snipper():
    """
    Name:       the_snipper
    Author:     robertdcurrier@gmail.com
    Created:    2022-09-12
    Modified:   2022-09-14
    Notes:      Main entry point.
    """
    clean_tmp()
    frame_count = 0
    thumbs = 0
    args = get_cli_args()
    taxa = validate_taxa(args['input'])
    config = get_config()
    model = load_model()
    max_thumbs = config["taxa"][taxa]["max_thumbs"]
    (frames) = get_all_frames(args)
    frame_num = 0
    for frame in frames:
        frame_num+=1
        bboxes = get_rois(frame)
        thumbs+=len(bboxes)
        if thumbs < max_thumbs:
            cell_snipper(frame, bboxes)
        else:
            logging.warning('the_snipper(): Max thumb count exceeded')
            sys.exit()



if __name__ == '__main__':
    """
    Main entry point
    """
    logging.basicConfig(level=logging.INFO)
    logging.info('coral_snipper initializing...')
    the_snipper()
