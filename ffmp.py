#!/usr/bin/env python3
import ffmpy
import os
import logging

infile = "testing/hsv0001_kareniaBrevis_1662667091_028.9312_-089.4429_Currier-Lab_raw.mp4"
outfile = "testing/fftest_pro.mp4"


def ffmpeg_it(infile, outfile):
    """
    """
    file_size = os.path.getsize(infile)
    logging.info('ffmpeg_it(): Infile %s Outfile %s',infile, outfile)
    if file_size == 0:
        logging.warning("ffmpeg_it(): Truncated input file %s of %d bytes" %
                (infile, file_size))
        return False
    else:
        logging.debug("ffmpeg_it(): %s is okay at %d bytes" % (infile, file_size))
    ff = ffmpy.FFmpeg(
            inputs={infile: None},
            outputs={outfile: '-strict -2 -loglevel 3'},
            global_options={'-y'}
        )
    ff.run()
    return True

ffmpeg_it(infile, outfile)
