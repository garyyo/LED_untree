#!/home/pi/.virtualenvs/rpi_led_string/bin/python
import math
import operator
import os
from perlin_noise import PerlinNoise
import numpy as np
import pandas as pd
import time
import matplotlib
import rpi_ws281x
from rpi_ws281x import PixelStrip
from rpi_ws281x import Color
import argparse
import atexit
import functools

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


# region helpers
def hsv_to_rgb(hsv_colors, length):
    hsv_colors = hsv_colors.reshape((length, 1, 3))
    colors = matplotlib.colors.hsv_to_rgb(hsv_colors)
    colors = colors[:, 0, :]
    return colors


def rgb_255(rgb):
    return (int(rgb[0]*255)<<16) + (int(rgb[1]*255)<<8) + (int(rgb[2]*255))


# returns the midpoint between two rgb values
def rgb_mix(rgb1, rgb2):
    return np.mean([rgb1, rgb2], 0)


def rgb_interpolate(rgb1, rgb2, t=0.5):
    return np.sum([rgb1*t, rgb2*(1-t)], 0)


# returns the midpoint between two hsl values
def hsv_mix(hsv1, hsv2):
    h1 = hsv1[0]
    h2 = hsv2[0]
    if np.abs(h1 - h2) > .5:
        if h1 > h2:
            h2 += 1
        else:
            h1 += 1
    h = np.mean([h1, h2]) % 1
    s, v = np.mean([hsv1[1:], hsv2[1:]], 0)
    return [h, s, v]


def hsv_interpolate(hsv1, hsv2, t=0.5):
    h1 = hsv1[0]
    h2 = hsv2[0]
    if np.abs(h1 - h2) > .5:
        if h1 > h2:
            h2 += 1
        else:
            h1 += 1
    h = np.sum([h1*t, h2*(1-t)]) % 1
    s, v = np.sum([hsv1[1:] * t, hsv2[1:] * (1 - t)], 0)
    return [h, s, v]


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
    return x, y, z


def color_off(strip):
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, int(Color(0, 0, 0)))
    strip.show()
    print("\ngoodbye!")


def initialize(parser, strip):
    # read arguments
    args = parser.parse_args()
    func = globals()[args.function]

    # clear out on exit
    if args.clear:
        exit_handler_bound = functools.partial(color_off, strip)
        atexit.register(exit_handler_bound)
        print("Lights will clear on exit")

    # which animation should be used
    animation_file = args.video
    if animation_file is not None:
        # try to read the fps from the filename
        # todo: this should be stored within the file itself, not in the name
        # anton: but I don't want to refactor that much right now
        try:
            fps = float(os.path.basename(animation_file).split("(")[-1].split(")")[0])
        except (NameError, TypeError, ValueError):
            fps = 30

        # load the file
        tree_animation = np.load(animation_file, allow_pickle=True)
        func = functools.partial(play_animation, tree_animation=tree_animation, fps=fps)

    # sound
    sound_file = args.audio
    if sound_file is not None:
        # we wait to import till here because pygame is annoying.
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
        pass

    # load coordinates, normalize
    df = pd.read_csv(args.coords)
    x, y, z = normalize_coordinates(df)

    return func, x, y, z


# endregion


# region effects


def sweep(x, y, z, progress):
    on_time = 3
    time_scale = 3
    # randomish (non-repeating)
    np.random.seed(int(np.floor(progress / time_scale)))
    hue = np.random.random()

    if progress % (time_scale * 3) < (1 * time_scale):
        hsv_colors = np.tile([hue, 0.7, 0], [len(x), 1])
        hsv_colors[:, 2] = np.clip(np.mod(x - progress, on_time) - (on_time - 1), 0, 1)
        # hsv_colors[:, 2] = np.clip(x + p / 3, 0, 1)
    elif progress % (time_scale * 3) < (2 * time_scale):
        hsv_colors = np.tile([hue, 0.7, 0], [len(x), 1])
        hsv_colors[:, 2] = np.clip(np.mod(y - progress, on_time) - (on_time - 1), 0, 1)
    else:
        hsv_colors = np.tile([hue, 0.7, 0], [len(x), 1])
        hsv_colors[:, 2] = np.clip(np.mod(z - progress, on_time) - (on_time - 1), 0, 1)

    colors = hsv_to_rgb(hsv_colors, len(x))
    return colors


def spin(x, y, z, progress):
    hsv_colors = np.tile([1, 0.5, 0], [len(x), 1])
    hsv_colors[:, 2] = np.mod(np.arctan2(x - 0.5, z - 0.5) / (2 * np.pi) + 0.5 - progress / 3, 1)
    colors = hsv_to_rgb(hsv_colors, len(x))
    return colors


def contra_spin(x, y, z, progress):
    hsv_colors = np.tile([1, 0.9, 0], [len(x), 1])

    wave_1 = np.clip(
        np.mod(
            np.arctan2(x - 0.5, z - 0.5) / (2 * np.pi) + 0.5 - progress / 3 + y,
            1
        ),
        a_min=0, a_max=1
    )
    wave_2 = np.clip(
        np.mod(
            (np.arctan2(x - 0.5, z - 0.5) / (2 * np.pi) + 0.5 - (progress / np.pi) / 3),
            1
        ) * 3 - 2,
        a_min=0, a_max=1
    )
    hsv_colors[:, 2] = np.clip(wave_1 * 0.5 + wave_2 * 0, 0, 1)
    hsv_colors[:, 0] = np.clip(wave_1 * 0.5 + wave_2 * 0, 0, 1)
    colors = hsv_to_rgb(hsv_colors, len(x))
    return colors


def counter_contra_spin(x, y, z, progress):
    hsv_colors = np.tile([1, 0.9, 0], [len(x), 1])

    wave_1 = np.clip(
        np.mod(
            (-np.arctan2(x - 0.5, z - 0.5) / (2 * np.pi) + 0.5 - (progress) / 3 + y),
            1
        ),
        a_min=0, a_max=1
    )
    wave_2 = np.clip(
        np.mod(
            (-np.arctan2(x - 0.5, z - 0.5) / (2 * np.pi) + 0.5 - (progress / np.pi) / 3),
            1
        ) * 3 - 2,
        a_min=0, a_max=1
    )
    hsv_colors[:, 2] = np.clip(wave_1 * 0.5 + wave_2 * 0, 0, 1)
    hsv_colors[:, 0] = np.clip(wave_1 * 0.5 + wave_2 * 0, 0, 1)
    colors = hsv_to_rgb(hsv_colors, len(x))
    return colors


def fire2(x, y, z, progress):
    hsv_colors = np.tile([0.01, 1, 0], [len(x), 1])
    angle = np.arctan2(x - 0.5, z - 0.5) / (2 * np.pi) + 0.5
    cycle_1 = (np.sin(progress) + 1) / 3
    cycle_2 = (np.sin(5.777 * progress) + 1) / 2
    cycle_3 = (np.sin(4 * np.pi * angle + 7.777 * progress) + 1) / 2
    cycle_4 = (np.sin(6.444 * progress) + 1) / 3
    brightness_top = y ** (2 * cycle_1 + 0.5 * cycle_2 + 0.5 * cycle_3)
    hue_bottom = 0.03 * y ** (10 * cycle_4)
    hsv_colors[:, 2] = brightness_top
    hsv_colors[:, 0] += hue_bottom

    # hsv_colors[:, 0] = 0.01 + 0.04 * brightness
    # hsv_colors[:, 1] = 0.9 - (0.1 * brightness)
    colors = hsv_to_rgb(hsv_colors, len(x))
    return colors


def fire(x, y, z, progress):
    hsv_colors = np.tile([0.05, 0.9, 0], [len(x), 1])
    angle = np.arctan2(x - 0.5, z - 0.5) / (2 * np.pi) + 0.5
    
    cycle_1 = (np.sin(progress) + 1) / 2
    cycle_2 = (np.sin(5.777 * progress) + 1) / 2
    cycle_3 = (np.sin(3 * progress + 4 * angle) + 1) / 2
    
    brightness = y ** (2.5 * cycle_1 + 0.5 * cycle_2 + 0.5 * cycle_3)
    
    hsv_colors[:, 2] = brightness
    hsv_colors[:, 0] = 0.01 + 0.05 * brightness
    # hsv_colors[:, 1] = 0.9 - (0.1 * brightness)
    colors = hsv_to_rgb(hsv_colors, len(x))
    breakpoint()
    return colors


def rainbow(x, y, z, progress):
    hue = (progress/10) % 1

    hsv_colors = np.tile([hue, 1, .5], [len(x), 1])
    return hsv_to_rgb(hsv_colors, len(x))


def off(x, y, z, progress):
    if progress > 1:
        exit()
    hsv_colors = np.tile([0, 0, 0], [len(x), 1])
    colors = hsv_to_rgb(hsv_colors, len(x))
    return colors


# plays from file
def play_animation(x, y, z, progress, tree_animation=None, fps=24):
    total_time = np.floor(len(tree_animation) / fps)
    frame = int(np.floor((progress % total_time) * fps))

    colors = tree_animation[frame] / 255
    return colors


# endregion


# region christmas
def new_years_default(x, y, z, progress):
    # todo: fill in with something else
    return fire(x, y, z, progress)


def new_years_approach(x, y, z, progress):
    # the fire fades
    countdown = NEW_YEAR_TIMESTAMP - (progress + TIME_OFFSET) - 12
    fade_progress = np.clip(countdown / 10, 0, 1)

    hsv_colors = np.tile([0.05, 0.9, 0], [len(x), 1])
    angle = np.arctan2(x - 0.5, z - 0.5) / (2 * np.pi) + 0.5
    cycle_1 = (np.sin(progress) + 1) / 2
    cycle_2 = (np.sin(5.777 * progress) + 1) / 2
    cycle_3 = (np.sin(3 * progress + 4 * angle) + 1) / 2
    brightness = y ** (2.5 * cycle_1 + 0.5 * cycle_2 + 0.5 * cycle_3)
    hsv_colors[:, 2] = brightness * fade_progress
    hsv_colors[:, 0] = 0.01 + 0.05 * brightness
    # hsv_colors[:, 1] = 0.9 - (0.1 * brightness)
    colors = hsv_to_rgb(hsv_colors, len(x))

    return colors


def dropdown_param(y, countdown, num_lights, speed, reverse_direction, reverse_bright, ring_width=1 / 10, init_hue=1.0):
    # hsv_colors = np.tile([init_hue, 1, .5], (num_lights, 1))

    # apply a ring of color starting at -1 moving to 0 from t=10->0
    ring_pos = countdown / (10 * speed)
    if reverse_direction:
        ring_pos = 1 - ring_pos

    hsv_colors = []
    for i, light_y in enumerate(y):
        v = np.clip(np.abs(light_y - ring_pos) / ring_width, 0, 1)
        if reverse_bright:
            v = 1 - v
        hsv_colors.append([init_hue, 1, v / 2])

    return hsv_to_rgb(np.array(hsv_colors), num_lights)


def new_years_countdown(x, y, z, progress):
    # calculate New Year relative time
    countdown = NEW_YEAR_TIMESTAMP - (progress + TIME_OFFSET)

    ring1 = hsv_to_rgb(dropdown_param(y, countdown, len(x), 1, True, True, init_hue=.9), len(x))
    ring2 = hsv_to_rgb(dropdown_param(y, countdown, len(x), .5, True, True, init_hue=.4), len(x))
    ring3 = hsv_to_rgb(dropdown_param(y, countdown, len(x), .25, True, True, init_hue=.7), len(x))
    ring4 = hsv_to_rgb(dropdown_param(y, countdown, len(x), .125, True, True, init_hue=.8), len(x))

    # hsv_colors = np.array([hsv_mix(hsl1, hsl2) for hsl1, hsl2 in zip(ring1, ring2)])
    # rgb_colors = hsv_to_rgb(hsv_colors, len(x))

    rgb_colors1 = rgb_mix(ring1, ring2)
    rgb_colors2 = rgb_mix(ring3, ring4)
    rgb_colors = rgb_mix(rgb_colors1, rgb_colors2)
    return rgb_colors


def new_years_playback(x, y, z, progress, tree_animation=None, fps=24):
    # we need to align the time to correct playtime place, this is a hacky dumb method
    total_time = np.floor(len(tree_animation) / fps)
    countdown = NEW_YEAR_TIMESTAMP - (progress + TIME_OFFSET) - total_time
    frame = int(np.floor((-countdown % total_time) * fps))

    colors = tree_animation[frame] / 255
    return colors


def new_years_time(x, y, z, progress):
    p = progress % 2
    p = int(p * 5)

    np.random.seed(p)

    rgb = np.random.rand(400, 3)
    rgb = (rgb > .5).astype("uint8")

    return rgb


def new_years_post(x, y, z, progress):
    return rainbow(x, y, z, progress)


# list of sequences. This is sorted by time to help figure out which one to run.
# {
#   "name": name of sequence (optional),
#   "func": function to run,
#   "time": time in seconds before NEW_YEAR_TIMESTAMP when this sequence should start
# }
countdown_args = {
    "tree_animation": np.load("new_years_countdown_(30.0).npy", allow_pickle=True),
    "fps": 30
}
new_years_sequence = sorted([
    {"Name": "countdown", "func": new_years_default, "time": -math.inf},
    {"Name": "countdown", "func": new_years_approach, "time": -22},
    {"Name": "countdown", "func": functools.partial(new_years_playback, **countdown_args), "time": -10},
    # {"Name": "countdown", "func": new_years_countdown, "time": -10},  # ten seconds before
    {"Name": "countdown", "func": new_years_time, "time": 0},  # ten minutes after before
    {"Name": "countdown", "func": new_years_post, "time": 2},  # ten minutes after before
], key=lambda x: x["time"])


def new_years_sequencer(x, y, z, progress):
    time_since = (progress + TIME_OFFSET) - NEW_YEAR_TIMESTAMP

    # get it through globals so we can grab at runtime
    func = globals().get("fire", None)

    # keep updating func till we find one that fails, the one before the failed one is the one to be played.
    for sequence in new_years_sequence:
        if sequence["time"] > time_since:
            break
        func = sequence["func"]
    return func(x, y, z, progress)


noise = PerlinNoise(octaves=1, seed=1)
def snowflakes(x, y, z, progress):
    v = np.array([noise([5*y[i]-progress*0.3+5*np.arctan2(x[i],z[i])]) for i in range(len(x))])
    v = v*2 - 0.5
    v = np.clip(v, 0, 1)
    sparkle = y<(0.02+0.01*np.sin(progress+0.1))
    np.random.seed(int(progress*2))
    v[np.where(y<0.02)] = np.clip(noise(progress)+0.5, 0, 1)
    v[np.where(y>0.997)] = 0.2
    rgb = np.array([v,v,v])
    return rgb.T


def rave(x, y, z, progress):
    bpm = 165
    num_steps = np.pi * 3
    
    rate = 60 / bpm
    color_progress = progress // rate
    
    hue = (color_progress / num_steps) % 1
    hsv_colors = np.tile([hue, 1, .5], [len(x), 1])
    return hsv_to_rgb(hsv_colors, len(x))
# endregion


def animate(strip, func, x, y, z, fps_counter): 
    start_time = TIME_OFFSET
    n = 100
    
    while True:
        if fps_counter:
            t0 = time.time()

        for j in range(n):
            progress = (time.time() - start_time) % 65536
            colors = func(x, y, z, progress)
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
    parser.add_argument("--fps", default=False, required=False)
    parser.add_argument("-c", "--clear", required=False, action="store_true")

    # Create PixelStrip object with appropriate configuration.
    strip = PixelStrip(**STRIP_CONFIG)
    strip.begin()

    effect_function, x, y, z = initialize(parser, strip)

    # play the actual animation
    try:
        animate(strip, effect_function, x, y, z, True)
    except KeyboardInterrupt:
        exit()


if __name__ == '__main__':
    main()
