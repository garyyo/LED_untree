import math
import time
from dataclasses import dataclass, field
from typing import Type

import numpy as np
import pandas as pd
import matplotlib


class AnimationContainer:
    desc: str = "A default description"

    def __init__(self, coords: np.ndarray, loop_time=10, num_cycles=1):
        self.coords = coords
        self.num_lights = len(coords)
        self.loop_time = loop_time
        self.num_cycles = num_cycles
        self.internal_time = 0
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

    def progress(self):
        return (self.internal_time % self.loop_time) / self.loop_time

    def cycle_progress(self):
        full_progress = self.progress() * self.num_cycles
        cycle = int(full_progress)
        return cycle, full_progress - cycle

    def __call__(self, delta_time, program_time, real_time, **kwargs):
        self.internal_time += delta_time
        # maybe change this to extract or only accept the 3 times?
        return self.animate(delta_time, program_time, real_time, **kwargs)

    # Turns off and exits
    # the standard things passed into the animate method are:
    #   delta_time: time since last frame in seconds (needed for anything that just tracks state)
    #   program_time: time since program start in seconds (needed for animations that ignore state and completely recalc from time)
    #   real_time: unix timestamp in seconds (needed for new year countdown)
    def animate(self, delta_time, program_time, real_time, **kwargs):
        # not sure why we make this 1d at first
        hsv_colors = np.tile([0, 0, 0], [self.num_lights, 1])

        # since it is 1d and in hsv, the hsv_to_rgb currently converts from hsv 1d to rgb:(num_lights, 3)
        colors = self.hsv_to_rgb_old(hsv_colors, self.num_lights)

        return np.zeros((self.num_lights, 3))

    pass


class SweepAnimation(AnimationContainer):
    desc: str = "Turns on the lights in a sweeping motion, using a random color, and in a random direction"

    def __init__(self, coords: np.ndarray, loop_time=6, num_cycles=2, sweep_ratio=.5, initial_direction=None, initial_hue=None, randomize_hue=True, randomize_direction=True):
        # todo: perhaps the loop_time should be multiplied byt the num_cycles so that a single cycle is of loop_time?
        super().__init__(coords, loop_time*num_cycles, num_cycles)
        self.sweep_ratio = sweep_ratio
        self.sweep_width = 1/sweep_ratio

        self.state = 0

        # default hue is random
        if initial_hue is None:
            initial_hue = np.random.rand()
        self.hue = initial_hue
        self.randomize_hue = randomize_hue

        # default direction is also random, and it needs to be normalized
        if initial_direction is None:
            initial_direction = np.random.rand(3)
        self.direction = np.array(initial_direction, dtype=np.float64) * 2 - 1
        self.direction /= np.linalg.norm(self.direction)
        self.randomize_direction = randomize_direction

    def signed_distance_from_plane(self, plane):
        # Calculate the signed distance for each point in self.coords
        plane /= np.linalg.norm(plane)
        signed_distances = np.dot(self.coords, plane)
        signed_distances /= np.linalg.norm(plane)

        return signed_distances

    def animate(self, delta_time, program_time, real_time, **kwargs):
        # update the time
        cycle, progress = self.cycle_progress()

        if self.state != cycle:
            self.set_new_cycle()
            self.state = cycle

        return self.get_colors(progress)

    def get_colors(self, progress):
        # get distances from the plane and normalize
        distances = self.signed_distance_from_plane(self.direction)
        distances = distances - distances.min()
        distances = distances / distances.max()

        # we want it to sweep through, and we have to run that sweep through over a certain period (including margins)
        brightness = (distances - (progress * (1 + self.sweep_ratio) - self.sweep_ratio)) / self.sweep_ratio

        brightness = brightness.clip(0, 1)
        # breakpoint()
        brightness = np.where(brightness == 1, 0, brightness)

        hsv = np.stack([
            np.ones(self.num_lights) * self.hue,
            np.ones(self.num_lights),
            brightness
        ], axis=1)
        return matplotlib.colors.hsv_to_rgb(hsv)

    def set_new_cycle(self):
        # initialize a new random direction to sweep through
        if self.randomize_direction:
            self.direction = np.random.rand(3) * 2 - 1
            self.direction /= np.linalg.norm(self.direction)
        if self.randomize_hue:
            self.hue = np.random.rand()

    pass


class TurningSweepAnimation(SweepAnimation):
    desc: str = "Turns on the lights in a sweeping motion, using a random color, and in two lerp'd random directions!"

    def __init__(self, coords: np.ndarray, loop_time=6, num_cycles=2, sweep_ratio=.5, initial_direction=None, initial_direction_2=None, initial_hue=None, randomize_hue=True, randomize_direction=True):
        super().__init__(coords, loop_time, num_cycles, sweep_ratio, initial_direction, initial_hue, randomize_hue, randomize_direction)

        if initial_direction_2 is None:
            initial_direction_2 = np.random.rand(3) * 2 - 1

        self.direction_2 = np.array(initial_direction_2, dtype=np.float64)
        self.direction_2 /= np.linalg.norm(self.direction_2)

        if initial_direction is None:
            initial_direction = np.random.rand(3) * 2 - 1

        self.direction_1 = np.array(initial_direction, dtype=np.float64)
        self.direction_1 /= np.linalg.norm(self.direction_1)

    def animate(self, delta_time, program_time, real_time, **kwargs):
        # update the time
        cycle, progress = self.cycle_progress()

        if self.state != cycle:
            self.set_new_cycle()
            self.state = cycle

        self.direction = (self.direction_2 - self.direction_1) * progress + self.direction_1

        return self.get_colors(progress)

    def set_new_cycle(self):
        # initialize a new random direction to sweep through
        if self.randomize_direction:
            self.direction_1 = np.random.rand(3) * 2 - 1
            self.direction_1 /= np.linalg.norm(self.direction_1)
            self.direction_2 = np.random.rand(3) * 2 - 1
            self.direction_2 /= np.linalg.norm(self.direction_2)
        if self.randomize_hue:
            self.hue = np.random.rand()
        pass
    pass


class SimpleConfettiAnimation(AnimationContainer):
    desc: str = "Ugly confetti, as confetti is"
    def animate(self, delta_time, program_time, real_time, **kwargs):
        np.random.seed(int(program_time*2))
        return np.random.random((self.num_lights, 3))


# todo: select some percentage (~5-15%) of a random set of points at some rate (~seconds) and fade those in and out with some frequency (~second)
#  potentially add in an overlap between periods, and offset the points randomly so they dont happen at all the same time if you dont want them to.
class FireflyAnimation(AnimationContainer):
    desc: str = "Firefly: Not Yet Implemented"
    def __init__(self):

        pass

    def animate(self, program_time, **kwargs):
        pass


class FireAnimation(AnimationContainer):
    desc: str = "A pleasing gently flickering fire animation"

    def animate(self, delta_time, program_time, real_time, **kwargs):
        x, y, z = self.get_xyz()

        angle = np.arctan2(x - 0.5, z - 0.5) / (2 * np.pi) + 0.5

        cycle_1 = (np.sin(self.internal_time) + 1) / 2
        cycle_2 = (np.sin(5.777 * self.internal_time) + 1) / 2
        cycle_3 = (np.sin(3 * self.internal_time + 4 * angle) + 1) / 2

        brightness = np.power(y, 2.5 * cycle_1 + 0.5 * cycle_2 + 0.5 * cycle_3)

        hsv = np.stack([
            0.01 + 0.05 * brightness,
            np.ones(self.num_lights) * 0.9,
            brightness
        ], axis=1)
        return matplotlib.colors.hsv_to_rgb(hsv)


class RainbowFillAnimation(AnimationContainer):
    desc: str = "Makes all the lights turn on and cycle through rainbow colors"
    def animate(self, delta_time, program_time, real_time, **kwargs):
        hue = (program_time/10) % 1
        hsv = np.stack([
            np.ones(self.num_lights) * hue,
            np.ones(self.num_lights) * 0.9,
            np.ones(self.num_lights) * 0.5
        ], axis=1)
        return matplotlib.colors.hsv_to_rgb(hsv)


# region sequencing
@dataclass
class SequenceItem:
    animation: Type[AnimationContainer]
    duration: float
    time: float = None
    kwargs: dict = field(default_factory=dict)
    seed: int = None
    offset: float = 0
    reset_state: bool = False


@dataclass
class SequenceInstance:
    animation: AnimationContainer
    duration: float
    time: float
    kwargs: dict
    seed: int
    offset: float
    reset_state: bool


# todo: rewrite, this implementation is a bit overcomplicated and full of bugs
#  instead I want it to keep state on which animation it is playing, and track which animation it should play next
#  there should be
#  - a special case for looping forever,
#  - a special initialization parameter for scheduled ones,
#  - scheduled ones should have a "continue with index"
#  none of this dumb calculating which one to play every time, just keep track of it.


class AnimationSequencer(AnimationContainer):
    def __init__(self, coords: np.ndarray, loop_time=10, sequence=(), does_loop=True):
        super().__init__(coords, loop_time, len(sequence))

        self.sequence_list = [
            SequenceInstance(
                item.animation(coords, **item.kwargs),
                item.duration,
                item.time,
                item.kwargs,
                item.seed,
                item.offset,
                item.reset_state
            )
            for item in sequence
        ]

        # build the data structures to calculate which one goes at what time
        self.sequenced_times = {}
        self.scheduled_times = {}
        self.total_duration = 0
        for i, item in enumerate(sequence):
            if item.time is None:
                self.sequenced_times[i] = [self.total_duration, 0]
                self.total_duration += item.duration
            else:
                self.scheduled_times[i] = [item.time, item.duration, 0]

        # update the sequenced_times with the next index to play
        self.first_sequenced_index = list(self.sequenced_times.keys())[0]
        self.sequence_index = self.first_sequenced_index
        last_i = self.first_sequenced_index
        for i in reversed(self.sequenced_times.keys()):
            self.sequenced_times[i][1] = last_i
            last_i = i

        # update the scheduled times, so they know where to pick back up
        for i in self.scheduled_times.keys():
            last_i = self.first_sequenced_index
            for found_index in range(i+1, len(self.sequence_list)):
                if self.sequenced_times.get(found_index, None) is not None:
                    last_i = found_index
                    break
            self.scheduled_times[i][2] = last_i

        # the time relative to the start of the entire sequence, resetting when the sequence loops
        self.sequenced_time = 0
        self.break_me = False
        pass

    def animate(self, delta_time, program_time, real_time, **kwargs):
        found_item = self.find_scheduled(delta_time, real_time)
        if found_item is None:
            found_item = self.find_sequenced(delta_time, real_time)
        colors = self.play_animation(*found_item, **kwargs)
        return colors

    def find_sequenced(self, delta_time, real_time):
        # find the next sequenced item that we need to play
        sequenced_item = self.sequence_list[self.sequence_index]

        # update variables to find the next one (if needed)
        next_sequenced_index = self.sequenced_times[self.sequence_index][1]
        sequence_start = self.sequenced_times[self.sequence_index][0]
        sequence_end = sequence_start + sequenced_item.duration

        # update the time
        self.sequenced_time += delta_time

        # if the animation is past its time, we reset the timer and go to the next one
        if not (sequence_start <= self.sequenced_time < sequence_end):
            self.sequence_index = next_sequenced_index
            sequenced_item = self.sequence_list[self.sequence_index]
            sequence_start = self.sequenced_times[self.sequence_index][0]
            # sequence_end = sequence_start + sequenced_item.duration

            # we also need to update the internal timer to reset it if we are restarting
            if next_sequenced_index == self.first_sequenced_index:
                self.sequenced_time -= self.total_duration
                # and if for god knows what reason it is less than 0 it is reset to 0
                if self.sequenced_time < 0:
                    self.sequenced_time = 0
            # breakpoint()
            pass

        # play it
        return sequenced_item, delta_time, self.sequenced_time - sequence_start, real_time

    def find_scheduled(self, delta_time, real_time):
        # todo: it currently searches for a scheduled one every single frame, would be nice to not do that...
        # find if there is a scheduled item that we need to play
        scheduled_item = None
        for index, (scheduled_time, scheduled_duration, next_sequenced_index) in self.scheduled_times.items():
            if scheduled_time <= real_time < (scheduled_time + scheduled_duration):
                scheduled_item = self.sequence_list[index]
                self.sequence_index = next_sequenced_index
                self.sequenced_time = self.sequenced_times[self.sequence_index][0]

        # if nothing is found, return
        if scheduled_item is None:
            return None

        return scheduled_item, delta_time, real_time - scheduled_item.time, real_time

    @staticmethod
    def play_animation(animation_item, delta_time, program_time, real_time, **kwargs):
        # set the seed if needed
        if animation_item.seed is not None:
            np.random.seed(animation_item.seed)

        # maybe I will add more things eventually?

        # play the animation
        return animation_item.animation(
            delta_time,
            program_time + animation_item.offset,
            real_time,
            **kwargs
        )
# endregion


def new_year_sequence(coords, **kwargs):
    red = 0
    green = 1/3
    blue = 2/3
    newyear = 1704085200
    sequence = AnimationSequencer(coords, **kwargs, sequence=[
        # SequenceItem(SimpleConfettiAnimation, 2, time.time() + 1),
        SequenceItem(FireAnimation, 5),
        # SequenceItem(SweepAnimation, 3, kwargs={"sweep_ratio": .4, "loop_time": 6})
        # SequenceItem(SimpleConfettiAnimation, 2, time.time() + 24),
        # SequenceItem(AnimationContainer, 5),
        # SequenceItem(RainbowFillAnimation, 5),
    #     SequenceItem(SweepAnimation, 12, kwargs={"sweep_ratio": .1}),
    #     SequenceItem(SweepAnimation, 6, kwargs={"sweep_ratio": -.5, "loop_time": 6}),
    #     SequenceItem(SweepAnimation, 12, kwargs={"sweep_ratio": .1, "loop_time": 12}),
    #     SequenceItem(SweepAnimation, 12, kwargs={"sweep_ratio": .4}),
    #     SequenceItem(SweepAnimation, math.inf)
    ])
    # sequence = AnimationSequencer(
    #     coords,
    #     **kwargs,
    #     sequence=[
    #         SequenceItem(SimpleConfettiAnimation, 2, time.time() + 5),
    #     ]
    #     + [SequenceItem(SweepAnimation, i, kwargs={"sweep_ratio": 1 - (.1 * i) ** 2, "loop_time": i}) for i in list(range(9, 1, -1))]
    #     # + [SequenceItem(SweepAnimation, i, kwargs={"sweep_ratio": 1 - (.1 * i) ** 2, "loop_time": i}) for i in list(range(1, 9, 1))]
    #     + [SequenceItem(FireAnimation, math.inf),]
    # )

    return sequence


def main():
    pass


if __name__ == '__main__':
    main()