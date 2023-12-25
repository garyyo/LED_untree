#!/home/pi/.virtualenvs/rpi_led_string/bin/python
import numpy as np

import spatial_patterns
from spatial_patterns import *


def main():
    parser = argparse.ArgumentParser(description="Render some cool effects for your ws281x lights, to file!")
    parser.add_argument("-n", "--name", required=False)
    parser.add_argument("--coords", required=False, default="final_coords.csv")
    parser.add_argument("-f", "--function", default="fire", required=False)
    parser.add_argument("--fps", default="30")
    parser.add_argument("--seconds", default="15")

    args = parser.parse_args()

    # what func do we need to render
    if args.name is None:
        name = args.function
    else:
        name = args.name
    render_func = globals()[args.function]

    # what frame rate?
    fps = float(args.fps)
    # how long
    seconds = float(args.seconds)

    # load up the positions
    df = pd.read_csv(args.coords)
    x, y, z = normalize_coordinates(df)

    spatial_patterns.NEW_YEAR_TIMESTAMP = 0
    spatial_patterns.TIME_OFFSET = -10

    tree_animation = []
    for frame in range(int(fps * seconds)):
        progress = frame/fps
        frame_data = (render_func(x, y, z, progress) * 255).astype("uint8")
        tree_animation.append(frame_data)

    tree_array = np.array(tree_animation)
    np.save(f"{name}_({fps}).npy", tree_array)
    pass


if __name__ == '__main__':
    main()
    pass
