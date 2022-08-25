#!/usr/bin/env python3
"""
Name: pt4_scale_generator
Author: robertdcurrier@gmail.com
Date:   2021-08-05
Modified: 2021-09-07
Notes:
"""
from __future__ import division
import glob
import os
import time
import sys
import re
import argparse
import json
import logging
import shlex
import sqlite3 as sqlite
import numpy as np
import subprocess
from  pt4_standalone import load_model
#pylint: disable=E1101

def get_config():
    """From config.json.

    Author: robertdcurrier@gmail.com
    Created:    2019-07-18
    Modified:   2019-07-18
    """
    c_file = open('configs/phyto4.cfg').read()
    config = json.loads(c_file)
    return config


def write_scale(scale):
    """
    Name: write_scale
    Author: robertdcurrier@gmail.com
    Created: 2021-Aug-06
    Modified:2021-09-13
    Notes: Need to change this from old method of writing .py file
    to writing .json file as pt4 requires taxa_scale.json
    """
    args = get_args()
    taxa = args['taxa']
    mode = args['mode']
    logging.info( "write_scale(): Writing tmp/%s_%s_scale.json" % (taxa, mode))
    outfile = 'tmp/%s_%s_scale.json' % (taxa, mode)
    the_date = time.strftime('%c')
    title = "Phytotracker4"
    scale_dict = {}
    scale_dict['title'] = title
    scale_dict['taxa'] = taxa
    scale_dict['mode'] = mode
    scale_dict['created'] = the_date
    scale_dict['scale'] = scale

    json_file = open(outfile, 'w')
    try:
        json_file.write(json.dumps(scale_dict))
        json_file.close()
        return
    except:
        logging.warning('write_scale(): FAILED to write %s.' % the_file)
        json_file.close()
        sys.exit()


def list_files(the_dir):
    """
    Name:
    Author:
    Date:
    Notes:
    """
    the_glob = '%s/*.mp4' % the_dir
    logging.debug('list_files(): Using glob %s' % the_glob)
    the_files = sorted(glob.glob(the_glob))
    logging.debug('list_files(): Found files: %s' % the_files)
    return the_files


def create_db():
    """
    Name:
    Author:
    Date:
    Notes:
    """
    conn = sqlite.connect('tmp/scale.db')
    conn.row_factory = sqlite.Row
    cursor = conn.cursor()
    #create from scratch
    logging.info( "create_db(): Creating tables...")
    the_query = "create table IF NOT EXISTS range('measured_cells' int, 'max_cells' int)"
    cursor.execute(the_query)
    the_query = """create table IF NOT EXISTS scale('measured' int, 'min' int, 'max' int,
     'mult' int, 'avg' int)"""
    cursor.execute(the_query)
    logging.info( "create_db(): Cleaning out range and scale tables...")
    the_query = """delete from range"""
    cursor.execute(the_query)
    the_query = """delete from scale"""
    cursor.execute(the_query)
    return cursor, conn


def connect_db():
    """
    Name:
    Author:
    Date:
    Notes:
    """
    conn = sqlite.connect('tmp/scale.db')
    conn.row_factory = sqlite.Row
    cursor = conn.cursor()
    return cursor


def calc_multiplier(cell_avg, measured_cells):
    """
    Name:
    Author:
    Date:
    Notes:
    """
    cell_mult = (measured_cells*1000)/cell_avg
    return cell_mult


def get_stats(cursor):
    """
    Name:
    Author:
    Date:
    Notes:
    """
    the_query = """ select measured_cells as c_measured, min(max_cells) as
    c_min, max(max_cells) as c_max, avg(max_cells) as c_avg from range
    group by measured_cells;"""
    results = cursor.execute(the_query)
    for the_record in results.fetchall():
        cell_mult = (calc_multiplier(the_record['c_avg'],
                     the_record['c_measured']))
        the_query = ("INSERT INTO scale values('%d', '%d', '%d','%d', '%d')" %
        (the_record['c_measured'], the_record['c_min'], the_record['c_max'],
	 cell_mult, the_record['c_avg']))
        cursor.execute(the_query)


def db_to_scale(cursor):
    """
    Name: gen_scale
    Author: robertdcurrier@gmail.com
    Date: 2021-08-05
    Notes: Generates dict scale for phytotracker 4
    """
    cursor = connect_db()
    scale = []
    the_query = """SELECT measured, min, max, mult, avg from scale
     ORDER BY measured ASC"""
    results = cursor.execute(the_query).fetchall()
    for result in results:
        the_range = ({"measured": result["measured"], "min": result["min"],
                     "max": result["max"], "mult": result["mult"],
                     "avg": result["avg"]})
        scale.append(the_range)
    return scale


def get_args():
    """
    Name: get_args
    Author: robertdcurrier@gmail.com
    Created:   2021-09-14
    Modified:  2021-09-14
    Notes:
    """
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("-d", "--directory", help="Path to files",
                       action="store_true")
    arg_p.add_argument("-g", "--generate", help="No process, just make scale",
                       action="store_true")
    arg_p.add_argument("-f", "--file", help="Use this file name",
                        action="store_true")
    arg_p.add_argument("-t", "--taxa", help="Taxa",
                        required="True")
    arg_p.add_argument("-m", "--mode", help="Live or Fixed",
                        required="True")
    args = vars(arg_p.parse_args())
    return args


def interp_scale(results):
    """
    Name: generate_scale
    Author: robertdcurrier@gmail.com
    Date: 2021-08-05
    Notes: Interpolates scale
    """
    the_date = time.strftime('%c')
    cell_range = range(1,300)
    index = 0
    scale=[]
    measured = []
    cell_count = []
    for interval in results:
        measured.append(results[index]['measured'])
        cell_count.append(results[index]['avg'])
        index+=1

    for cells in cell_range:
        interp_cells = int(np.interp(cells,cell_count,measured))
        scale.append(interp_cells)

    return scale


def generate_scale():
    """
    Name: generate_scale
    Author: robertdcurrier@gmail.com
    Created: 2021-08-05
    Modified: 2021-09-14
    Notes: Generates scale for phytotracker 4
    Added cli args on 2021-09-14
    """
    args = get_args()
    if args['generate']:
        regen_scale()
        sys.exit()

    taxa = args['taxa']
    logging.info('generate_scale(%s)' % taxa)
    config = get_config()

    model = load_model(taxa)
    if not args['directory']:
        data_dir = config['taxa'][taxa]['calibration_dir']
    else:
        data_dir = args['directory']

    cursor, conn = create_db()

    file_count = 0
    the_files = list_files(data_dir)
    patt_cpl = re.compile('^[0-9]+')
    if len(the_files) > 0:
        for the_file in the_files:
            logging.debug('generate_scale(): Processing %s' % the_file)
            f_name = os.path.split(the_file)[1]
            serial, taxa, cpl, epoch, type = f_name.split('_')
            file_count += 1
            command = "./pt4_standalone.py -i %s -t %s" % (the_file, taxa)
            process = subprocess.Popen(shlex.split(command),
                                       stdout=subprocess.PIPE)
            output = process.stdout.readline()
            cells = int(output.decode())
            logging.info( "%s has %d cells..." % (f_name, cells))
            if cells > 0:
                the_query = ("insert into range values('%d', '%d')" %
                            (int(cpl), cells))
                cursor.execute(the_query)
            else:
                logging.info('generate_scale(): Zero cells, skipping')
    # TO DO: clean out high and low values for each cell count
    prune_outliers(cursor)
    get_stats(cursor)
    conn.commit()
    scale = db_to_scale(cursor)
    scale = interp_scale(scale)
    write_scale(scale)


def regen_scale():
    """
    Name: regen_scale()
    Author: robertdcurrier@gmail.com
    Created: 2021-09-15
    Modified: 2021-09-15
    Notes: Just regenerate scale and write to file. Don't reprocess
    all the videos. This allows us to delete skewed entries and recreate
    the scale without having to process all the good videos.
    """
    logging.info("regen_scale(): Regenerating scale w/no processing")
    cursor = connect_db()
    get_stats(cursor)
    scale = db_to_scale(cursor)
    scale = interp_scale(scale)
    write_scale(scale)
    sys.exit()


def prune_outliers(cursor):
    """
    Name: prune_outliers
    Author: robertdcurrier@gmail.com
    Date: 2021-09-07
    Notes: Removes whack cell counts for each cpL
    """
    logging.info('prune_outliers()')




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('Phytotracker4 scale generator...')
    generate_scale()
