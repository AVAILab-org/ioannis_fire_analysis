import sys
import os.path as path
import cv2
import numpy as np

"""PROGRAM CONSTANTS"""
# the height of the frame in pixels
FRAME_HEIGHT_PIXELS = 720
# the width of the frame in pixels
FRAME_WIDTH_PIXELS = 1280
# the height of the frame in cm
FRAME_HEIGHT_CM = 52.5
# the width of the frame in cm
FRAME_WIDTH_CM = 100.0
# the minimum threshold of the red pixel before considered a fire
RED_THRESHOLD = 128
# the size of the gaussian kernel to run over the image
GAUSSIAN_KERNEL_SIZE = 10
# the width of the moving average window to use to smooth the signal on each bin
MOVING_AVG_LENGTH = 1
# the number of bins to split the image into horizontally
NUM_BINS = 128
# the minimum number of pixel required in a bin before it is considered to be on fire
MIN_BIN_INTENSITY_PIXELS = 30

"""RUNTIME CONSTANTS"""
gaussian_kernel = np.ones((GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), np.float32) / (GAUSSIAN_KERNEL_SIZE ** 2)
height_pix_to_cm = FRAME_HEIGHT_CM / FRAME_HEIGHT_PIXELS
width_pix_to_cm = FRAME_WIDTH_CM / FRAME_WIDTH_PIXELS

def get_height(image: np.ndarray) -> np.ndarray:
    # get the red color
    image = image[..., -1]

    # gaussian blur it
    image = cv2.filter2D(image, -1, gaussian_kernel)

    # threshold it
    image = (image > RED_THRESHOLD) * 1.0

    # split the image up into many discrete bins
    # the dimensions are now [bins, height, width], range=[0, 1]
    assert image.shape[-1] % NUM_BINS == 0
    image = np.split(image, image.shape[-1] // NUM_BINS, axis=-1)
    image = np.array(image).transpose((2, 1, 0))

    # threshold the intensity
    # the dimensions are now [bins, height, width], range=[0, 1]
    image = image * (np.sum(image, axis=(1, 2), keepdims=True) > MIN_BIN_INTENSITY_PIXELS)

    # sum up over the width of each bin, crush all zeros
    # the dimensions are now [bins, height], range=[0, 1]
    image = np.sum(image, axis=-1) > 0

    # need to make the lowest line of pixels true because
    # np.max on an array of False gives the highest value
    # the dimensions are now [bins, height], range=[0, 1]
    image[..., -1] = True

    # get the pixel_height
    # the dimensions are now [bins, ], range=[0, FRAME_HEIGHT_PIXELS]
    pixel_heights = FRAME_HEIGHT_PIXELS - np.argmax(image, axis=-1)

    # find the highest (indexed as lowest in np) value
    # the dimensions are now [bins, ], range=[0, FRAME_HEIGHT_CM]
    return pixel_heights * height_pix_to_cm


def process_video(filename: str) -> np.ndarray:
    # start video capture
    vidcap = cv2.VideoCapture(path.join(dirname, "movs/", filename))

    heights = []
    while True:
        # try read a frame, if fail just break
        success, image = vidcap.read()
        if not success or image is None: break

        heights.append(get_height(image))

    # convert to np array
    return np.array(heights)


def process_heights(heights):
    # heights is of shape [time, bins], range=[0, FRAME_HEIGHT_CM]
    # run a moving average window
    convolve1d = lambda x: np.convolve(x, np.ones(MOVING_AVG_LENGTH) / MOVING_AVG_LENGTH, mode="valid")
    heights = np.apply_along_axis(convolve1d, axis=0, arr=heights)

    return heights

def process_fronts(heights):
    # heights is of shape [time, bins], range=[0, FRAME_HEIGHT_CM]
    # crush the bins without fire
    has_fire = heights > np.min(heights, axis=-1, keepdims=True)

    # find the right-most bin with fire in it
    # bin_front is of shape [time, ], range=[0, NUM_BINS]
    bin_front = np.argmax(heights[:, ::-1], axis=-1)

    return bin_front * width_pix_to_cm

if __name__ == "__main__":
    # handle names
    dirname = path.dirname(__file__)
    mov_path = path.basename(sys.argv[1])
    npy_path = f"{path.join(dirname, 'npys/', mov_path)}.npy"
    csv_heights_path = f"{path.join(dirname, 'csvs/', mov_path)}_heights.csv"
    csv_fronts_path = f"{path.join(dirname, 'csvs/', mov_path)}_fronts.csv"

    # check if we need to process the video if the pixel heights already exists
    if not path.exists(npy_path) or True:
        heights = process_video(mov_path)
        np.save(npy_path, heights)
    else:
        heights = np.load(npy_path)

    # run filters through the heights
    heights = process_heights(heights)
    fronts = process_fronts(heights)

    # save as npy array and csv
    np.savetxt(csv_heights_path, heights, delimiter=",")
    np.savetxt(csv_fronts_path, fronts, delimiter=",")
