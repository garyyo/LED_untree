from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib


class AnimationContainer:
    def __init__(self, coords: np.ndarray, loop_time=10, num_cycles=1):
        self.coords = coords
        self.num_lights = len(coords)
        self.loop_time = loop_time
        self.num_cycles = num_cycles
        pass

    @staticmethod
    def hsv_to_rgb_old(hsv_colors, length):
        hsv_colors = hsv_colors.reshape((length, 1, 3))
        rgb_colors = matplotlib.colors.hsv_to_rgb(hsv_colors)
        rgb_colors = rgb_colors[:, 0, :]
        return rgb_colors

    def get_xyz(self):
        x, y, z = self.coords.T
        return x, y, z

    def progress(self, program_time):
        return (program_time % self.loop_time) / self.loop_time

    def cycle_progress(self, program_time):
        full_progress = self.progress(program_time) * self.num_cycles
        cycle = int(full_progress)
        return cycle, full_progress - cycle

    def __call__(self, *args, **kwargs):
        # maybe change this to extract or only accept the 3 times?
        return self.animate(*args, **kwargs)

    # Turns off and exits
    # the standard things passed into the animate method are:
    #   delta_time: time since last frame in seconds (needed for anything that just tracks state)
    #   program_time: time since program start in seconds (needed for animations that ignore state and completely recalc from time)
    #   real_time: unix timestamp in seconds (needed for new year countdown)
    def animate(self, delta_time, program_time, real_time, **kwargs):
        # since the default one is for turning off all the lights and exiting, we have an exit condition.
        # this may need to be turned into an exception if we are to implement multithreading
        if program_time > 1:
            exit()

        # not sure why we make this 1d at first
        hsv_colors = np.tile([0, 0, 0], [self.num_lights, 1])

        # since it is 1d and in hsv, the hsv_to_rgb currently converts from hsv 1d to rgb:(num_lights, 3)
        colors = self.hsv_to_rgb_old(hsv_colors, self.num_lights)
        return colors

    pass


class SweepAnimation(AnimationContainer):
    desc: str = "Turns on the lights in a sweeping motion, using a random color, and in a random direction"

    def __init__(self, coords: np.ndarray, loop_time=6, num_cycles=2):
        super().__init__(coords, loop_time, num_cycles)
        self.direction = np.array([0, 0, 1.0])
        self.state = 0
        self.hue = 0

    def signed_distance_from_plane(self, plane):
        # Calculate the signed distance for each point in self.coords
        signed_distances = np.dot(self.coords, plane)
        signed_distances /= np.linalg.norm(plane)

        return signed_distances

    def animate(self, program_time, **kwargs):
        cycle, progress = self.cycle_progress(program_time)

        if self.state != cycle:
            # initialize a new random direction to sweep through
            self.direction = np.random.rand(3)
            self.direction /= np.linalg.norm(self.direction)
            self.hue = np.random.rand()
            self.state = cycle

        # get distances from the plane and normalize
        distances = self.signed_distance_from_plane(self.direction)
        distances = distances - distances.min()
        distances = distances / distances.max()

        # we want it to sweep through, so we need to have a sweep of a certain size,
        sweep_width = 2

        # and we have to run that sweep through over a certain period
        brightness = (distances * sweep_width - (progress * 3 - 1) * (sweep_width - 1)).clip(0, 1)

        brightness = np.where(brightness == 1, 0, brightness)

        # hsv = np.array([(self.hue, 1, val) for val in brightness])
        hsv = np.stack([
            np.ones(self.num_lights) * self.hue,
            np.ones(self.num_lights),
            brightness
        ], axis=1)
        return matplotlib.colors.hsv_to_rgb(hsv)
    pass


class FireAnimation(AnimationContainer):
    desc: str = "A pleasing gently flickering fire animation"

    def animate(self, program_time, **kwargs):
        x, y, z = self.get_xyz()

        angle = np.arctan2(x - 0.5, z - 0.5) / (2 * np.pi) + 0.5

        cycle_1 = (np.sin(program_time) + 1) / 2
        cycle_2 = (np.sin(5.777 * program_time) + 1) / 2
        cycle_3 = (np.sin(3 * program_time + 4 * angle) + 1) / 2

        brightness = np.power(y, 2.5 * cycle_1 + 0.5 * cycle_2 + 0.5 * cycle_3)

        hsv = np.stack([
            0.01 + 0.05 * brightness,
            np.ones(self.num_lights) * 0.9,
            brightness
        ], axis=1)
        return matplotlib.colors.hsv_to_rgb(hsv)


class RainbowFillAnimation(AnimationContainer):
    def animate(self, program_time, **kwargs):
        hue = (program_time/10) % 1
        hsv = np.stack([
            np.ones(self.num_lights) * hue,
            np.ones(self.num_lights) * 0.9,
            np.ones(self.num_lights) * 0.5
        ], axis=1)
        return matplotlib.colors.hsv_to_rgb(hsv)


def main():
    pass


if __name__ == '__main__':
    main()