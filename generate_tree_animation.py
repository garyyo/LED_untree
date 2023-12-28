import os.path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange


def color_correct(rgb, correction=(1, 1, 1)):
    return [color*correct for color, correct in zip(rgb, correction)]


def tree_coordinates_normalized(df):
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

def convert_video(video, flat_lights, num_frames, halved=False, correction=(1, 0.2, 0.05)):
    tree_animation = []

    for _ in trange(num_frames):
        ret, frame = video.read()

        if not ret:
            print("reached last frame")
            break
        if not video.isOpened():
            print("video file closed unexpectedly")
            break

        height, width, _ = frame.shape

        # apply the video mask to the coordinates
        lights_frame = [
            color_correct(
                np.flip(frame[int(np.floor(y * height)) - 1][int(np.floor(x * width)) - 1]),
                correction
            )
            if z > .5 or not halved
            else [0, 0, 0]
            for i, (x, y, z) in enumerate(flat_lights)
        ]

        tree_animation.append(lights_frame)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            print("q key was pressed, exiting")
            break
    return tree_animation


def main(video_file="bad_apple_2.mp4", max_seconds=None):
    # open video
    video = cv2.VideoCapture(video_file)

    # extract metadata
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    name = os.path.splitext(os.path.basename(video_file))[0]

    if max_seconds is not None:
        num_frames = int(frame_rate * max_seconds)
    print(f"{frame_rate=}\n{num_frames=}\n{name=}")

    # open tree light coordinates
    tree_df = pd.read_csv("final_coords.csv")
    x, y, z = tree_coordinates_normalized(tree_df)
    # flatten the lights on some axis
    # todo: switch this over to conical coordinate mapping?
    flat_lights = np.stack([x, y, z], axis=1)

    tree_animation = convert_video(video, flat_lights, num_frames, correction=(1,1,1), halved=True)

    # save to file
    tree_array = np.array(tree_animation)
    np.save(f"{name}_({frame_rate}).npy", tree_array)

    # close things up properly
    video.release()
    cv2.destroyAllWindows()
    pass


if __name__ == '__main__':
    main()
    pass
