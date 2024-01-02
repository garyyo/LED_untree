import datetime
import random
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

    def signed_distance_from_plane(self, plane):
        # Calculate the signed distance for each point in self.coords
        plane /= np.linalg.norm(plane)
        signed_distances = np.dot(self.coords, plane)
        signed_distances /= np.linalg.norm(plane)

        return signed_distances

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
    desc: str = "Firefly: it does fireflies!"

    def __init__(self, coords: np.ndarray, p_spawn: float = 0.07):
        super().__init__(coords)
        self.p_spawn = p_spawn
        self.fireflies: dict[int, FireFly] = {}
        self.hue = 0.085
        self.tau = 0.5

    def exp_profile(self, t: float) -> float:
        return np.exp(-t/self.tau)

    def exp_gauss_profile(self, t: float) -> float:
        t_ramp = self.tau/2
        t_ramp_speed = t_ramp/4
        if t<t_ramp:
            return np.exp(-(t-t_ramp)**2/(2*t_ramp_speed**2))
        else:
            return self.exp_profile(t-t_ramp)

    def spawn(self, loc: int):
        self.fireflies[loc] = FireFly(
            loc=loc,
            profile=lambda t: self.exp_gauss_profile(t=t),
            max_life=self.tau*4,
        )

    def kill_old(self):
        return {i: self.fireflies.pop(i) for i, f in list(self.fireflies.items()) if not f.alive}

    @property
    def available_flies(self) -> list[int]:
        return [i for i in range(self.num_lights) if i not in self.fireflies.keys()]

    def animate(self, delta_time, program_time, real_time, **kwargs):
        if np.random.rand() <= self.p_spawn:
            available_flies = self.available_flies
            if available_flies:
                fly_ind = np.random.choice(available_flies)
                self.spawn(loc=fly_ind)

        brightness = np.zeros(self.num_lights)
        for i, f in self.fireflies.items():
            brightness[i] = f.update(delta_time)

        self.kill_old()

        hsv = np.stack([
            np.ones(self.num_lights) * self.hue,
            np.ones(self.num_lights),
            brightness * 0.8
        ], axis=1)
        return matplotlib.colors.hsv_to_rgb(hsv)


class FireFly:
    def __init__(self, loc: int, profile, max_life: float):
        self.loc = loc
        self.profile = profile
        self.brightness = 1.0
        self.t = 0.0
        self.max_life = max_life

    @property
    def alive(self):
        return self.t < self.max_life

    def update(self, dt: float) -> float:
        self.t += dt
        return self.profile(self.t)


class SupernovaAnimation(AnimationContainer):
    desc: str = "A sky full of stars disappearing"

    def __init__(self, coords: np.ndarray, loop_time=72, num_cycles=1):
        super().__init__(coords, loop_time, num_cycles)
        # what is reset_time? seconds? seconds of what?
        self.reset_time = 6

        # what are all these magical constants?
        self.death_progress = np.random.rand(self.num_lights)
        self.death_progress = self.death_progress * (1 - 0.1 - (self.reset_time+3.5)/loop_time) + 0.1
        self.death_time = self.death_progress*loop_time

    def animate(self, delta_time, program_time, real_time, **kwargs):
        reset_progress = (program_time - self.loop_time + self.reset_time) / self.reset_time
        if reset_progress > 0:
            hue = np.ones(self.num_lights) * np.interp(reset_progress, [0, 0.2, 0.3, 1], [0.6, 0.6, 0.2, 0.2])
            sat = np.ones(self.num_lights) * np.interp(reset_progress, [0, 0.2, 0.3, 1], [1, 0, 0, 0.4])
            val = np.ones(self.num_lights) * np.interp(reset_progress, [0, 0.2, 1], [0, 1, 0.1])
        
            hsv = np.stack([hue, sat, val], axis=1) % 1
            return matplotlib.colors.hsv_to_rgb(hsv)
    
        star_progress = 1 - self.death_progress + self.progress()
        time_dead = program_time - self.death_time
        
        hue = np.zeros(self.num_lights)
        sat = np.zeros(self.num_lights)
        val = np.zeros(self.num_lights)
        
        # configs for lifetime
        default_brightness = 0.1
        max_brightness = 0.2
        red_fraction = 0.1
        default_sat = 0.4
        final_sat = 0.9
        default_hue = 0.2
        final_hue = 0.05
        
        # gradually brighten
        val = np.where(
            star_progress <= 1,
            self.progress() * (max_brightness - default_brightness) / self.death_progress + max_brightness,
            val
        )
        
        # redden towards the end
        sat = np.where(
            star_progress <= 1,
            np.interp(star_progress, [1-red_fraction, 1], [default_sat, final_sat]), 
            sat)
            
        hue = np.where(
            star_progress <= 1,
            np.interp(star_progress, [1-red_fraction, 1], [default_hue, final_hue]),
            hue
        )

        # todo: implement twinkling to simulate atmospheric effects

        # todo: implement different "sized" (colored) stars. color should affect brightness,
        #  but we may want to vary brightness to simulate distance? or just not do either of those

        # configs for supernova
        time_profile = [0,              0.5,        0.6,    2,      2.5,    3]
        hue_profile  = [final_hue,      final_hue,  0.6,    0.6,    0.6,    0.6]
        sat_profile  = [final_sat,      1,          1,      1,      1,      1]
        val_profile  = [max_brightness, 0,          0,      1,      0.8,    0]
        
        hue = np.where(star_progress > 1, np.interp(time_dead, time_profile, hue_profile), hue)
        sat = np.where(star_progress > 1, np.interp(time_dead, time_profile, sat_profile), sat)
        val = np.where(star_progress > 1, np.interp(time_dead, time_profile, val_profile), val)

        hsv = np.stack([hue, sat, val], axis=1) % 1
        return matplotlib.colors.hsv_to_rgb(hsv)


class CountdownAnimation(AnimationContainer):
    desc = "itsa countdown!"

    def __init__(self, coords: np.ndarray, loop_time=10, num_cycles=1):
        super().__init__(coords, loop_time, num_cycles)

    def animate(self, delta_time, program_time, real_time, **kwargs):
        progress = self.progress()
        progress2 = progress * 2 - 1
        sweep_ratio = 0.4
        distances = self.signed_distance_from_plane(np.array([0.0, 1.0, 0.0]))
        distances = distances - distances.min()
        distances = distances / distances.max()
        i = 0
        rings = [
            ((distances - ((progress * 1 - 0) - sweep_ratio)) / sweep_ratio),
            ((distances - ((progress * 2 - 1) - sweep_ratio)) / sweep_ratio),
            ((distances - ((progress * 4 - 3) - sweep_ratio)) / sweep_ratio)
            # for i in range(3, 0, -1)
            # , np.zeros(self.num_lights)
        ]

        # breakpoint()
        rings = [np.where(ring > 1, 0, ring.clip(0, 1)) for ring in rings]

        rgb = np.stack(rings, axis=1)

        return rgb


class FireAnimation(AnimationContainer):
    desc: str = "A pleasing gently flickering fire animation"

    def animate(self, delta_time, program_time, real_time, **kwargs):
        x, y, z = self.get_xyz()

        angle = np.arctan2(x - 0.5, z - 0.5) / (2 * np.pi) + 0.5

        cycle_1 = (np.sin(self.internal_time) + 1) / 2
        cycle_2 = (np.sin(np.e * 2 * self.internal_time) + 1) / 2
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


class RainAnimation(AnimationContainer):
    desc: str = "Makes it rain"

    def __init__(self, coords: np.ndarray, p_spawn: float=0.07, speed: float=15.0):
        super().__init__(coords)
        self.p_spawn = p_spawn
        self.speed = speed
        self.sequences: list[RainDropSequence] = []
        self.lines: list[list[int]] = [[]]
        self.setup_lines()

    def setup_lines(self):
        idx_list = [[0,49],[99,50],[100, 149],[199, 150],[200, 249],[299, 250],[300, 349],[399,350]]
        idx_list = [list(reversed(l)) for l in idx_list]
        for pair in idx_list:
            self.lines.append(np.arange(pair[0], pair[1], 1 if pair[1] > pair[0] else -1))

    def exp_gauss_linv_profile(self, n_idx: int, t: float, v: float, tau: float):
        x = np.arange(n_idx)
        tau_ramp = 0.1*tau
        t_r = t - tau_ramp - x/v
        return (t_r > 0) * (np.exp((-t_r/tau) * (t_r > 0))) + (t_r <= 0)*np.exp(-t_r**2/(2*tau_ramp**2))

    def exp_linv_profile(self, n_idx: int, t: float, v: float, tau: float):
        x = np.arange(n_idx)
        t_r = t - x/v
        return (t_r > 0) * np.exp((-t_r/tau) * (t_r > 0))

    def spawn(self, indices: list[int]):
        self.sequences.append(
            RainDropSequence(
                indices=indices,
                profile=lambda nx, t:
                    self.exp_gauss_linv_profile(
                        n_idx=nx,
                        t=t,
                        v=self.speed,
                        tau=0.5,
                    ),
                max_t=7,
                hue=2/3.0 + np.random.rand()/100,
                saturation=np.random.rand()/4+0.5,
            )
        )

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    def update(self, dt: float) -> dict[int, tuple[float, float, float]]:
        return_hsv: dict[int, tuple[float, float, float]] = {i: (0.0, 0.0, 0.0) for i in range(self.num_lights)}
        epsilon = 0.1

        # calculate brightnesses and add any overlaps
        for sequence in self.sequences:
            drop_hsv = sequence.update(dt)
            for i, hsv in drop_hsv.items():
                h = hsv[0] if return_hsv[i][2] < epsilon else return_hsv[i][0]
                s = hsv[1] if return_hsv[i][2] < epsilon else return_hsv[i][1]
                v = return_hsv[i][2] + hsv[2]
                return_hsv[i] = (
                    h,
                    s,
                    v if v <= 1 else 1
                )

        # return and ensure brightness doesn't exceed 1
        return return_hsv

    def remove(self):
        return [s for s in self.sequences if not s.finished]

    def animate(self, delta_time, program_time, real_time, **kwargs):
        # start new raindrop
        if np.random.rand() < self.p_spawn:
            line = random.choice(self.lines)
            # line = self.lines[1]
            self.spawn(indices=line)

        # update raindrops
        brightness_dict = self.update(dt=delta_time)

        # remove finished raindrops
        self.sequences = self.remove()

        return matplotlib.colors.hsv_to_rgb(np.stack(list(brightness_dict.values()), axis=1).T)

class RainDropSequence:

    def __init__(self, indices: list[int], profile, max_t: float, max_brightness = 0.8, hue: float = 0.0, saturation: float = 0.0):
        self.profile = profile
        self.indices = indices
        self.n_indices = len(self.indices)
        self.t = 0.0
        self.t_max = max_t
        self.max_brightness = max_brightness
        self.hue=hue
        self.saturation=saturation

    @property
    def finished(self) -> bool:
        return self.t > self.t_max

    def update(self, dt: float):
        self.t += dt
        x_t = self.profile(self.n_indices, self.t)
        return {i: (self.hue, self.saturation, v*self.max_brightness) for i, v in zip(self.indices, x_t)}

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
        # IF there are overlapping schedule animations, it prefers the latest one in the loop.
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


def str_time(str_date):
    return datetime.datetime.strptime(str_date, '%d/%m/%y %H:%M:%S %z').timestamp()
    pass


def new_year_sequence(coords, **kwargs):
    # red = 0
    # green = 1/3
    # blue = 2/3
    # newyear = 1704085200
    newyear = str_time("01/01/24 00:00:00 -0500")
    second = 1
    minute = 60 * second
    hour = 60 * minute
    day = hour * 24

    # debug, remove for the real one
    # newyear -= day
    # newyear = str_time("31/12/23 09:09:00 -0500")
    # print(newyear)

    sequence = AnimationSequencer(coords, **kwargs, sequence=[
        # SequenceItem(SimpleConfettiAnimation, 2, time.time() + 1),
        SequenceItem(FireAnimation, 5 * second),
        SequenceItem(SupernovaAnimation, hour - 10 * second, newyear - hour, {"loop_time": hour - 10 * second}),
        SequenceItem(CountdownAnimation, 10 * second, newyear - 10 * second, {"loop_time": 10 * second}),
        SequenceItem(SimpleConfettiAnimation, 5, newyear),
        # SequenceItem(SweepAnimation, 3, kwargs={"sweep_ratio": .4, "loop_time": 6})
        # SequenceItem(SimpleConfettiAnimation, 2, time.time() + 24),
        # SequenceItem(AnimationContainer, 5),
        # SequenceItem(RainbowFillAnimation, 5),
        # SequenceItem(SweepAnimation, 12, kwargs={"sweep_ratio": .1}),
        # SequenceItem(SweepAnimation, 6, kwargs={"sweep_ratio": -.5, "loop_time": 6}),
        # SequenceItem(SweepAnimation, 12, kwargs={"sweep_ratio": .1, "loop_time": 12}),
        # SequenceItem(SweepAnimation, 12, kwargs={"sweep_ratio": .4}),
        # SequenceItem(SweepAnimation, math.inf)
    ] + [
        SequenceItem(TurningSweepAnimation, 12 * second, str_time(f"31/12/23 {hour}:00:00 -0500"))
        for hour in range(10, 22)
    ])

    return sequence


def main():
    pass


if __name__ == '__main__':
    main()