import glob
import os
import numpy as np
import pandas as pd
import cv2
import scipy
import csv
import seaborn
import matplotlib as mpl
import matplotlib.pyplot as plt

to_csv_kwargs = {"index": False, "quoting": csv.QUOTE_NONNUMERIC, "encoding": "utf-8"}
mpl.use('Qt5Agg')


def plot_3d_scatter(df, title=""):
    # plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    ax.set_xlabel("X")
    # y and z are switched to make it look right on the graph
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")

    if "confidence" in df:
        ax.scatter(df.x, df.z, -df.y, c=1-df.confidence, cmap="rocket")
    else:
        ax.plot(df.x, df.z, -df.y)

    plt.show()
    pass


def rotate(degrees, df):
    # rotate
    t = np.radians(degrees)

    # must do it in one line if I don't want to use temp vars
    df.x, df.z = np.cos(t) * df.x - np.sin(t) * df.z, np.sin(t) * df.x + np.cos(t) * df.z

    return df


# region image stuff
def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def find_bright(img):
    kernel_size = 5
    brightness_thresh = 0.9

    img_copy = img.copy()

    # apply a (bad) blurring conv, this will remove any potential error bright spots
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size*kernel_size)
    output = scipy.signal.convolve2d(img_copy, kernel, "same")

    # visualize the sanity check
    # show(((output > output.max() * brightness_thresh) * 255).round().astype('uint8'))

    # gather all the points that pass the threshold
    points = (output > output.max() * brightness_thresh).nonzero()
    # todo: find the center, and then find how many points are outliers, and the filter out outliers
    center = [vals.sum() / len(vals) for vals in points]
    return np.flip(np.array(center)), output.max() / 255


# endregion


# region processing
def check_files(files):
    if files is None:
        files = glob.glob("pics/*.png")

    angles = set()
    light_nums = set()

    for file in files:
        index, angle = os.path.splitext(os.path.basename(file))[0].split("_")
        angles.add(angle)
        light_nums.add(index)
    light_nums = sorted([int(i) for i in light_nums])
    angles = sorted([int(i) for i in angles])

    return light_nums, angles


def process_files(files, save=False):
    if files is None:
        files = glob.glob("pics/*.png")

    lights_data = []
    for file in files:
        index, angle = os.path.splitext(os.path.basename(file))[0].split("_")

        # read the image
        img = cv2.imread(file)
        gray_img = gray(img)

        # calculate bright center
        center, confidence = find_bright(gray_img)

        # add entry to list
        lights_data.append({
            "light_index": int(index),
            "angle": int(angle),
            "confidence": confidence,
            "x": center[0],
            "y": center[1],
        })
    # into a df, and sorted for easier viewing
    df = pd.DataFrame(lights_data)
    df = df.sort_values(["angle", "light_index"])

    # optionally save it for later
    if save:
        df.to_csv("image_positions.csv", **to_csv_kwargs)

    return df


def process_positions(image_positions="image_positions.csv"):
    df = pd.read_csv(image_positions)

    # find unique angles, and get all pairs that are 90 degrees apart (and the ones that are -270 apart which is technically 90 degrees)
    angles = df.angle.unique()
    angle_pairs = [(a1, a2) for a1 in angles for a2 in angles if (a1 + 90) % 360 == a2]

    coord_dfs = []
    for a1, a2 in angle_pairs:
        coord_df = df[df.angle == a1].reset_index(drop=True)
        z_df = df[df.angle == a2].reset_index(drop=True)

        # reassign z
        coord_df["z"] = z_df.x

        # multiply confidence
        coord_df["confidence"] = coord_df.confidence * z_df.confidence

        # try to find the center to rotate around, all methods are problematic
        # anton: the dumb one
        # c_x, c_z = 0, 0
        # anton: this one assumes that the "top" point is always the same
        # c_x, c_z = coord_df.iloc[coord_df.y.idxmin()][["x", "z"]].tolist()
        # anton: this one finds the center of mass but is weighted by confidence
        # c_x, c_z = np.average(np.array(coord_df[["x", "z"]]), weights=coord_df.confidence, axis=0)
        # anton: this one takes only the top 10% of the tree and calculates center x,z based on that.
        c_x, c_z = coord_df[(coord_df.y < (coord_df.y.min() + (coord_df.y.max() - coord_df.y.min()) * .1)) & (coord_df.confidence > .96)][["x", "z"]].mean()

        coord_df.x -= c_x
        coord_df.z -= c_z

        coord_df = rotate(a1, coord_df)

        # plot_3d_scatter(coord_df, f"Rotate ({a1}°)")
        coord_dfs.append(coord_df)

        pass

    # going row by row, check which of the dfs pass the confidence threshold, and average only those that do.
    # todo: a second pass of confidence, if the final point is too far from any one of the inputs, ignore it and recalc average
    final_coords = []
    for i in range(len(coord_dfs[0])):
        coords_confs = [coord_dfs[c].iloc[i][["x", "y", "z", "confidence"]].tolist() for c in range(len(coord_dfs))]
        mean_coord = np.array([(x, y, z) for x, y, z, conf in coords_confs if conf > .9]).mean(axis=0)

        # todo: some second sort of filtering?

        final_coords.append({
            "light_index": i,
            "x": mean_coord[0],
            "y": mean_coord[1],
            "z": mean_coord[2],
        })

    # into a df
    final_df = pd.DataFrame(final_coords)

    # sanity check display it
    plot_3d_scatter(final_df, f"averaged and filtered")

    # save it
    final_df.to_csv("final_coords.csv", **to_csv_kwargs)

    return final_df


# endregion


def main():

    # light_nums, angles = check_files(None)
    # pos_df = process_files(None, save=True)
    # df = process_positions()
    df = pd.read_csv("final_coords.csv")

    # a sweep from bottom to top in y order


    breakpoint()

    pass


if __name__ == '__main__':
    main()
    pass
