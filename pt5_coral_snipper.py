#!/usr/bin/env python3
# coding: utf-8
"""
Name:       pt5_coral_snipper
Author:     robertdcurrier@gmail.com
Created:    2022-04-14
Modified:   2022-07-12
Notes:      Now using CORAL to determine ROIs
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
from natsort import natsorted
# Local utilities
from pt5_utils import (string_to_tuple, get_config, load_scale,
process_video_all, write_frame, gen_coral,  classify_frame, validate_taxa,
gen_bboxes, caption_frame, calc_cellcount,load_model, check_focus, clean_tmp)
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
    arg_p.add_argument("-m", "--mask", help="write masked frames",
                       action='store_true')
    arg_p.add_argument("-f", "--frames", help="write raw frames",
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
        roi = frame[y1:y2,x1:x2]
        epoch_time = int(time.time_ns())
        fname = 'tmp/%d_%s.png' %(epoch_time, taxa)
        logging.info('cell_snipper(%s): Writing %s' % (taxa, fname))
        cv2.imwrite(fname, roi)
        thumb_count+=1
    return thumb_count


if __name__ == '__main__':
    """
    Main entry point
    """
    logging.basicConfig(level=logging.INFO)
    logging.info('coral_snipper initializing...')
    clean_tmp()
    frame_count = 0
    thumbs = 0
    args = get_cli_args()
    taxa = validate_taxa(args['input'])
    config = get_config()
    max_thumbs = config["taxa"][taxa]["max_thumbs"]
    (frames, cons) = process_video_all(args)
    for frame in frames:
        (target_frame, circ_cons) = gen_coral(args, frame, cons[frame_count])
        (taxa, bboxes) = gen_bboxes(args, circ_cons)
        thumb_count = cell_snipper(frame, bboxes)
        frame_count+=1
        thumbs = thumbs + thumb_count
        if thumbs > max_thumbs:
            logging.info('coral_snipper(%s): Thumbcount max of %d hit.' %
                         (taxa, thumbs))
            sys.exit()
    logging.info('coral_snipper(%s): Wrote %d thumbs' % (taxa, thumbs))
