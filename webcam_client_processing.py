import cv2 as cv
import numpy as np
import scipy
import scipy.signal
import socket
import atexit
import time
import pandas as pd
import csv

to_csv_kwargs = {"index": False, "quoting": csv.QUOTE_NONNUMERIC, "encoding": "utf-8"}

cam = cv.VideoCapture(0)

PORT = 5000
HOST = '192.168.1.141'


# region image stuff
def get_pic():
    s, img = cam.read()
    if s:
        return img
    raise Exception("No camera")


def gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# inputs an image, and outputs an x,y coord.
def find_bright(img):
    kernel_size = 5
    brightness_thresh = 0.99

    img_copy = img.copy()
    show(img_copy)

    # apply a (bad) blurring conv, this will remove any potential error bright spots
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size*kernel_size)
    output = scipy.signal.convolve2d(img_copy, kernel, "same")

    # visualize the sanity check
    show(((output > output.max() * brightness_thresh) * 255).round().astype('uint8'))

    # gather all the points that pass the threshold
    points = (output > output.max() * brightness_thresh).nonzero()
    # todo: find the center, and then find how many points are outliers, and the filter out outliers
    center = [vals.sum() / len(vals) for vals in points]
    return np.flip(np.array(center))


def get_compare_img():
    return gray(get_pic())


def process_image(img, compare_img=None):
    # if we have a comparison image, use that
    if compare_img is not None:
        diff2 = compare_img - img

        diff1 = np.abs(compare_img.astype(int) - img.astype(int))
        diff = np.clip(diff1, 0, 255).astype('uint8')
    else:
        diff = img.copy()

    # find center of the bright spot
    return find_bright(diff), diff


def show(img):
    cv.imshow("cam test", img)
    cv.waitKey(0)


def box_point(img, center):
    img_copy = img.copy()
    margin = 5
    color = (0, 0, 255)
    c1 = (center - margin).astype(int).tolist()
    c2 = (center + margin).astype(int).tolist()
    rect_img = cv.rectangle(img_copy, c1, c2, color, 1)
    rect_img = cv.putText(rect_img, f"center={center.astype(int).tolist()}", c1, cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return rect_img


# endregion


def send_light(s, i=-1):
    # send which light is to be changed
    s.sendall(bytes(str(i), 'utf-8'))

    # wait for done signal
    data = s.recv(1024)
    # todo: check if data is actually done and not some nonsense, eventually
    pass


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # connect
        s.connect((HOST, PORT))

        # handle unexpected closure case
        atexit.register(s.close)

        light_positions = []

        # Should match the LED_COUNT in positions server
        start_light = 0
        end_light = 400

        # get the ith light
        for i in range(start_light, end_light):
            # turn off all the lights for comparison
            send_light(s, -1)
            compare_img = get_pic()

            # turn on the light we want to test
            send_light(s, i)
            img = get_pic()

            time.sleep(0.1)

            # get gray versions
            compare_gray = gray(compare_img)
            img_gray = gray(img)

            # center, diff = process_image(img_gray)
            center, diff = process_image(img_gray, compare_gray)
            # lights_entry = dict(zip(("x", "y"), tuple(center)))
            lights_entry = {
                "index": i,
                "x": center[0],
                "y": center[1],
            }
            light_positions.append(lights_entry)

        df = pd.DataFrame(light_positions)
        df.to_csv("light_pos.csv", **to_csv_kwargs)
        breakpoint()

        # center, diff = process_image(img_gray, compare_img_gray)
        # show(box_point(img, center))
        # show(box_point(compare_img, center))
        # show(box_point(diff, center))


if __name__ == '__main__':
    main()
    pass
