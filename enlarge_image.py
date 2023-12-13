import cv2
import numpy as np
import time
import multiprocessing
import argparse
from numba import jit
from PyQt5.QtGui import QImage, QImageReader
from energy_calculation import forward_energy
from energy_calculation import backward_energy



@jit
def find_seam(image, calculate_energy):

    energy = calculate_energy(image)
    rows, cols = energy.shape
    M = energy
    seam_energy=0

    backtrack = np.zeros_like(M, dtype=int)

    # populate DP matrix
    for i in range(1, rows):
        for j in range(0, cols):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    # backtrack to find path
    seam_idx = []
    j = np.argmin(M[-1])

    # seam_energy=0
    for i in range(rows - 1, -1, -1):
        seam_idx.append(j)
        seam_energy += energy[i, j]
        j = backtrack[i, j]
    seam_idx.reverse()
    return np.array(seam_idx)

@jit
def remove_seam(image, seam):
    rows, cols, _ = image.shape
    new_image = np.zeros((rows, cols - 1, 4), dtype=np.uint8)
    # print(new_image.shape)
    for i in range(rows):
        j = seam[i]
        new_image[i, :, :] = np.delete(image[i, :, :], j, axis=0)
    return new_image

@jit
def add_seam(image, seam):
    rows, cols = image.shape[:2]
    output = np.zeros((rows, cols + 1, 4))
    for i in range(rows):
        j = seam[i]
        for ch in range(3):
            if j == 0:
                new_pixel = np.average(image[i, j: j + 2, ch])
                output[i, j, ch] = image[i, j, ch]
                output[i, j + 1, ch] = new_pixel
                output[i, j + 1:, ch] = image[i, j:, ch]
            else:
                new_pixel = np.average(image[i, j - 1: j + 1, ch])
                output[i, : j, ch] = image[i, : j, ch]
                output[i, j, ch] = new_pixel
                output[i, j + 1:, ch] = image[i, j:, ch]
    return output


def enlarge(image, target, calculate_energy):

    seam_list = []
    temp = image.copy()
    for i in range(target):
        seam = find_seam(temp, calculate_energy)
        seam_list.append(seam)
        temp = remove_seam(temp,seam)
    seam_list.reverse()
    for i in range(target):
        seam = seam_list.pop()
        image = add_seam(image, seam)
        for remaining_seam in seam_list:
            remaining_seam[np.where(remaining_seam >= seam)] += 2
        print(image.shape)
    return image

def seam_carving(image, target_columns, target_rows, calculate_energy):
    image = image.astype(np.float64)
    print("Starting image size: ", image.shape)
    output = enlarge(image, target_columns, calculate_energy)
    print(output.shape)
    print("cols removed")
    rotated_image = rotate_image(output, True)
    print(rotated_image.shape)
    print("enlarge rows")
    output = enlarge(rotated_image, target_rows, calculate_energy)
    output = rotate_image(output, False)
    print("Output image size: ", output.shape)
    return output


def enlarge_algo(qImage, row, col, is_forward_energy):

    calculate_energy = forward_energy if is_forward_energy else backward_energy
    input_image = qimage_to_numpy(qImage)
    target_rows = row
    target_columns = col
    print("Starting")
    start_time = time.time()
    #entering seam_carving
    output_image = seam_carving(input_image, target_columns, target_rows, calculate_energy)

    end_time = time.time()
    print("Time of the normal way running in: ", (end_time - start_time))

    # print("Time of the normal way running in: ", time_all)
    red_channel=output_image[:, :, 0]
    green_channel = output_image[:, :, 1]
    blue_channel = output_image[:, :, 2]
    color_channels = [red_channel, green_channel, blue_channel ]
    color_image = np.array(color_channels)
    color_image = np.transpose(color_image, axes=(1, 2, 0))
    color_image = color_image.astype('uint8')
    # Ensure the correct order of color channels
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Convert the NumPy array to a QImage
    height, width, channel = color_image_rgb.shape
    bytes_per_line = 3 * width

    # Convert the color_image_rgb to contiguous array
    color_image_rgb_contiguous = np.ascontiguousarray(color_image_rgb)

    # Create QImage with Format_RGB888
    q_image = QImage(color_image_rgb_contiguous.data, width, height, bytes_per_line, QImage.Format_RGB888)
    print("done")
    return q_image

def qimage_to_numpy(qimage):
    if qimage.isNull():
        return None
    # Convert QImage to NumPy array
    buffer = qimage.bits()
    buffer.setsize(qimage.byteCount())
    numpy_array = np.frombuffer(buffer, dtype=np.uint8).reshape((qimage.height(), qimage.width(), 4))
    # Remove alpha channel if present
    if qimage.hasAlphaChannel():
        numpy_array = numpy_array[:, :, :3]

    return numpy_array

def rotate_image(image, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(image, k)

# def visualize(im, boolmask=None, rotate=False):
#     vis = im.astype(np.uint8)
#     if boolmask is not None:
#         vis[np.where(boolmask == False)] = SEAM_COLOR
#     if rotate:
#         vis = rotate_image(vis, False)
#     cv2.imshow("visualization", vis)
#     cv2.waitKey(1)
#     return vis