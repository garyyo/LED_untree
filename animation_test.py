#!/home/pi/.virtualenvs/rpi_led_string/bin/python
import csv
import os
import numpy as np
import pandas as pd
import time
import argparse
import atexit
import functools
import inspect
import rpi_ws281x
from rpi_ws281x import PixelStrip
from rpi_ws281x import Color

import pattern_containers

to_csv_kwargs = {"index": False, "quoting": csv.QUOTE_NONNUMERIC, "encoding": "utf-8"}

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

# get a list of available classes for command line stuff
command_line_classes = {name: obj for name, obj in inspect.getmembers(pattern_containers) if inspect.isclass(obj)}
command_line_class_options = "\n\t".join(command_line_classes.keys())


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
    animation_container = command_line_classes.get(args.function, pattern_containers.new_year_sequence)
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

    effect_container = animation_container(coords)
    # effect_container = pattern_containers.FireAnimation(coords)

    return effect_container, coords


def animate(strip, effect_container, fps_counter, brightness_multiplier):
    real_time_start = time.time()
    start_time = time.time()
    last_time = time.time()
    t0 = time.time()
    program_time = 0
    n = 100
    debug = False

    while True:
        for j in range(n):
            if debug:
                delta_time = 1/50
                program_time += delta_time
                real_time = real_time_start + program_time
            else:
                current_time = time.time()
                delta_time = current_time - last_time
                last_time = current_time
                program_time = time.time() - start_time
                real_time = time.time()

            colors = effect_container(delta_time=delta_time, program_time=program_time, real_time=real_time)

            for i, color in enumerate(colors):
                # todo: figure out how to implement the brightness multiplier
                strip.setPixelColor(i, rgb_255(color))
                pass
            strip.show()
        if fps_counter:
            print(n / (time.time() - t0))
            t0 = time.time()


def gen_csv():
    cwd = os.getcwd()
    pd.DataFrame([
        {
            "name": name,
            "desc": obj.desc,
            "command": f"{cwd}/animation_test.py -c -f {name}"
        }
        for name, obj in inspect.getmembers(pattern_containers)
        if inspect.isclass(obj)
        and issubclass(obj, pattern_containers.AnimationContainer)
        and not issubclass(obj, pattern_containers.AnimationSequencer)
    ]).to_csv("commands.csv", **to_csv_kwargs)

    print("commands generated and stored in \"commands.csv\"")


def get_args():
    # Make arguments
    parser = argparse.ArgumentParser(description="Play some cool effects on your ws281x lights!")
    parser.add_argument("--coords", required=False, default="final_coords.csv", help="Specify a set of coordinate. Defaults to final_coords.csv in current working directory.")
    parser.add_argument("-f", "--function", default=None, required=False, help="Choose an animation to play, check the source code for details", choices=command_line_classes.keys(), metavar='ANIMATIONCLASS')
    parser.add_argument("--function_options", help="Print out available functions for '-f'", action="store_true")
    parser.add_argument("--gen_csv", help="Generate the CSV file for the server", action="store_true")
    parser.add_argument("-v", "--video", required=False)
    parser.add_argument("-a", "--audio", required=False)
    parser.add_argument("--fps", default=False, required=False, action="store_true")
    parser.add_argument("-c", "--clear", required=False, action="store_true")
    parser.add_argument("-b", "--brightness", required=False, default="1", help="value from (0, 1] for a brightness multiplier, to make it less bright.")

    # read arguments
    args = parser.parse_args()

    if args.function_options:
        print(f"Choose from one of these options: \n\t{command_line_class_options}")
        exit()

    if args.gen_csv:
        gen_csv()
        exit()

    return args


def main():
    args = get_args()

    # Create PixelStrip object with appropriate configuration.
    strip = PixelStrip(**STRIP_CONFIG)
    strip.begin()
    effect_container, coords = initialize(args, strip)

    # play the actual animation
    try:
        animate(strip, effect_container, args.fps, brightness_multiplier=args.brightness)
    except KeyboardInterrupt:
        exit()


if __name__ == '__main__':
    main()
