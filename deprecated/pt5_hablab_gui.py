#!/usr/bin/env python3
import sys
import cv2
import logging
import pygame
import pygame.camera
import PySimpleGUI as sg
import numpy as np
"""
HABLAB GUI 2022-02-11 robertdcurrier@gmail.com
We are developing this app to make it easier to add new taxa. We can load
a video and change all the settings while watching the results.  Eventually
we will be able to pickle the results and save to a JSON file for easy
incorporation into pt5_Xception.cfg

TO DO: We read until end of frame and die. We need to put in a test so that
we loop instead of blowing up. Need to implement cell size and the display
of contours.
"""

# Keep settings in-house for pyinstaller ease of use
SCALE_PERCENT = 50
WIDTH = 1280
HEIGHT = 720
FPS = 10
DELAY = 50
LOADED = False

def make_layout():
    """
    Creates PySimpleGUI layout
    """
    # Define the window layout
    layout = [
        [sg.Image(filename="", key="-IMAGE-",size=(WIDTH, HEIGHT),
         background_color='black')],

        [sg.FileBrowse(key="-IN-")],
        [sg.Radio("None", "Radio", True, size=(10, 1))],
        [
            sg.Radio("Cell Size", "Radio", size=(10, 1), key="-SIZE-"),
            sg.Slider(
                (0, 255),
                128,
                1,
                orientation="h",
                size=(40, 15),
                key="-SIZE SLIDER-",
            ),
        ],
        [
            sg.Radio("Edges", "Radio", size=(10, 1), key="-CANNY-"),
            sg.Slider(
                (0, 255),
                128,
                1,
                orientation="h",
                size=(20, 15),
                key="-CANNY SLIDER A-",
            ),
            sg.Slider(
                (0, 255),
                128,
                1,
                orientation="h",
                size=(20, 15),
                key="-CANNY SLIDER B-",
            ),
        ],
        [sg.Button("Contours", size=(10, 1))],
        [sg.Button("Exit", size=(10, 1))],
    ]
    return layout


def event_loop():
    """
    The main event
    """
    LOADED = False

    sg.theme("Dark Blue 15")

    layout = make_layout()

    # Create the window and show it without the plot
    window = sg.Window("HABLAB: HABscope Optical Workbench", layout,
                        margins=(0,0),finalize=True)

    HAVE_FRAME = False
    while True:
        event, values = window.read(timeout=DELAY)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if (values["-IN-"]):
            try:
                frame = cv2.imread(values["-IN-"])
                # Resize so we can fit in the window
                frame = resize_frame(frame)
                HAVE_FRAME = True
            except:
                logging.warning('event_loop(): Failed to open %s' % values["-IN-"])
                sys.exit()
        if values["-SIZE-"]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
            frame = cv2.threshold(
                frame, values["-THRESH SLIDER-"], 255, cv2.THRESH_BINARY
            )[1]
        elif values["-CANNY-"]:
            frame = cv2.Canny(
            frame, values["-CANNY SLIDER A-"], values["-CANNY SLIDER B-"]
        )
        if HAVE_FRAME:
            imgbytes = cv2.imencode(".png", frame)[1].tobytes()
            window["-IMAGE-"].update(data=imgbytes)

    window.close()


def resize_frame(frame):
    """
    2022-01-10
    Resizes image coming from file so we don't blow out the
    window. Need to keep things down to about 960x720 or less for
    easy display
    """
    width = int(frame.shape[1] * SCALE_PERCENT / 100)
    height = int(frame.shape[0] * SCALE_PERCENT / 100)
    dim = (width, height)
    frame = cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)
    return(frame)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    event_loop()
