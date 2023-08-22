import sys
import numpy as np
import matplotlib.pyplot as plt
from os import path
from process_video import process_heights, process_fronts

if __name__ == "__main__":
    # handle names
    dirname = path.dirname(__file__)
    mov_path = path.basename(sys.argv[1])
    npy_path = f"{path.join(dirname, 'npys/', mov_path)}.npy"

    heights = np.load(npy_path)
    heights = process_heights(heights)
    print(heights.shape)
