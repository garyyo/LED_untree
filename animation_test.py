#!/home/pi/.virtualenvs/rpi_led_string/bin/python
import math
import operator
import os
# from perlin_noise import PerlinNoise
import numpy as np
import pandas as pd
import time
import rpi_ws281x
from rpi_ws281x import PixelStrip
from rpi_ws281x import Color
import argparse
import atexit
import functools

import pattern_containers

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# the real one
NEW_YEAR_TIMESTAMP = 1672549200
# NEW_YEAR_TIMESTAMP = 1672528500
# for debug, uncomment this line, you get some seconds of test time
NEW_YEAR_TIMESTAMP = time.time() + 25

# NEW_YEAR_TIMESTAMP = 1672520760
TIME_OFFSET = time.time()

# LED strip configuration:
STRIP_CONFIG = {
    "num": 400,  # Number of LED pixels.
    "pin": 18,  # GPIO pin connected to the pixels (18 uses PWM!).
    # "pin": 10, # GPIO pin connected to the pixels (10 uses SPI /dev/spidev0.0).
    "freq_hz": 800000,  # LED signal frequency in hertz (usually 800khz)
    "dma": 10,  # DMA channel to use for generating signal (try 10)
    "invert": False,  # Set to 0 for darkest and 255 for brightest
    "brightness": 255,  # True to invert the signal (when using NPN transistor level shift)
    "channel": 0,  # set to '1' for GPIOs 13, 19, 41, 45 or 53
    "strip_type": rpi_ws281x.ws.WS2811_STRIP_RGB,  # the color config of the strip, normal is
}


def rgb_255(rgb):
    return (int(rgb[0]*255)<<16) + (int(rgb[1]*255)<<8) + (int(rgb[2]*255))


def color_off(strip):
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, int(Color(0, 0, 0)))
    strip.show()
    print("\ngoodbye!")


def normalize_coordinates(df):
    x = df.loc[:, "x"].to_numpy()
    x = x - np.min(x)
    x = x / np.max(x)
    y = df.loc[:, "y"].to_numpy()
    y = y - np.min(y)
    y = y / np.max(y)
    z = df.loc[:, "z"].to_numpy()
    z = z - np.min(z)
    z = z / np.max(z)
    return np.array([x, y, z]).T


def initialize(args, strip):
    func = None
    # func = globals()[args.function]

    # clear out on exit
    if args.clear:
        exit_handler_bound = functools.partial(color_off, strip)
        atexit.register(exit_handler_bound)
        print("Lights will clear on exit")

    # which animation should be used
    # animation_file = args.video
    # if animation_file is not None:
    #     # try to read the fps from the filename
    #     # todo: this should be stored within the file itself, not in the name
    #     # anton: but I don't want to refactor that much right now
    #     try:
    #         fps = float(os.path.basename(animation_file).split("(")[-1].split(")")[0])
    #     except (NameError, TypeError, ValueError):
    #         fps = 30
    #
    #     # load the file
    #     tree_animation = np.load(animation_file, allow_pickle=True)
    #     func = functools.partial(play_animation, tree_animation=tree_animation, fps=fps)

    # sound
    # sound_file = args.audio
    # if sound_file is not None:
    #     # we wait to import till here because pygame is annoying.
    #     import pygame
    #     pygame.mixer.init()
    #     pygame.mixer.music.load(sound_file)
    #     pygame.mixer.music.play()
    #     pass

    # load coordinates, normalize
    df = pd.read_csv(args.coords)
    coords = normalize_coordinates(df)
    # breakpoint()

    effect_container = pattern_containers.fire_sweep_sequence(coords)
    # effect_container = pattern_containers.FireAnimation(coords)

    return effect_container, coords


def animate(strip, effect_container, fps_counter):
    start_time = TIME_OFFSET
    n = 100

    while True:
        if fps_counter:
            t0 = time.time()

        for j in range(n):
            colors = effect_container(
                delta_time=None,
                program_time=time.time() - start_time,
                real_time=time.time()
            )
            for i, color in enumerate(colors):
                strip.setPixelColor(i, rgb_255(color))
            strip.show()
        if fps_counter:
            print(n / (time.time() - t0))


def main():
    # Make arguments
    parser = argparse.ArgumentParser(description="Play some cool effects on your ws281x lights!")
    parser.add_argument("--coords", required=False, default="final_coords.csv")
    parser.add_argument("-f", "--function", default="fire", required=False)
    parser.add_argument("-v", "--video", required=False)
    parser.add_argument("-a", "--audio", required=False)
    parser.add_argument("--fps", default=False, required=False, action="store_true")
    parser.add_argument("-c", "--clear", required=False, action="store_true")

    # read arguments
    args = parser.parse_args()

    # Create PixelStrip object with appropriate configuration.
    strip = PixelStrip(**STRIP_CONFIG)
    strip.begin()
    effect_container, coords = initialize(args, strip)

    # play the actual animation
    try:
        animate(strip, effect_container, args.fps)
    except KeyboardInterrupt:
        exit()


if __name__ == '__main__':
    main()
