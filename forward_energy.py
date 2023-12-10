import cv2
import numpy as np
import time
import multiprocessing
import argparse
from numba import jit
from PIL import Image
from PyQt5.QtGui import QImage, QImageReader



def compute_energy(image):

    rows, cols = image.shape[:2]

    I = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(float)
    energy = np.zeros((rows, cols))
    m = np.zeros((rows, cols))

    up = np.roll(I, 1, axis=0)
    left = np.roll(I, 1, axis=1)
    right = np.roll(I, -1, axis=1)

    cost_Mid = np.abs(right - left)
    cost_Left = np.abs(up - left) + cost_Mid
    cost_Right = np.abs(up - right) + cost_Mid

    for i in range(1, rows):

            m_Up = m[i - 1]
            m_Left = np.roll(m_Up, 1)
            m_Right = np.roll(m_Up, -1)

            top = np.array([m_Left, m_Up, m_Right])
            costs = np.array([cost_Left[i], cost_Mid[i], cost_Right[i]])
            top += costs

            argmins = np.argmin(top, axis=0)
            m[i] = np.choose(argmins, top)
            energy[i] = np.choose(argmins, costs)

    return energy


def find_seam(image):
    energy = compute_energy(image)
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
    return np.array(seam_idx), seam_energy


def remove_seam(image, seam):
    rows, cols, _ = image.shape
    # channels_to_keep = [1, 2, 3]
    # image = image[:, :, channels_to_keep]
    new_image = np.zeros((rows, cols - 1, 4), dtype=np.uint8)
    # print(new_image.shape)
    for i in range(rows):
        j = seam[i]
        new_image[i, :, :] = np.delete(image[i, :, :], j, axis=0)
    return new_image




# def visualize_seam(image, seam, color=(0, 0, 255)):
#     visualized_image = image.copy()
#
#     for i in range(len(seam)):
#         visualized_image[i, seam[i]] = color
#
#     return visualized_image


def seam_carving(image, target_columns, target_rows):
    # visualized_image = image.copy()
    rows = 0
    cols = 0
    while rows <= target_rows or cols <= target_columns:
        seam_vertical, col_energy = find_seam(image)
        image_rotate = rotate_image(image, True)
        seam_horizontal, row_energy= find_seam(image_rotate)
        # print(row_energy,col_energy)
        if col_energy<row_energy:
            if  cols<=target_columns:
                image = remove_seam(image, seam_vertical)
                cols += 1
            else:
                image_rotate = remove_seam(image_rotate, seam_horizontal)
                image = rotate_image(image_rotate, False)
                rows += 1
                # print("null1")
        else:
            # print("null2")
            if rows <= target_rows:
                image_rotate = remove_seam(image_rotate, seam_horizontal)
                image = rotate_image(image_rotate, False)
                rows += 1
            else:
                image = remove_seam(image, seam_vertical)
                cols += 1

        print(image.shape)
    return image



    # return visualized_image,
    # return output_image


def algo1(qImage, row, col):

    input_image = qimage_to_numpy(qImage)
    target_rows = row
    target_columns = col
    print("Starting")
    start_time = time.time()
    output_image = seam_carving(input_image, target_columns, target_rows)
    end_time = time.time()
    print("Time of the normal way running in: ", (end_time - start_time))

    # Display the original image with highlighted seams
    # cv2.imshow("Original Image with Seams", visualized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print("Time of the normal way running in: ", time_all)
    red_channel=output_image[:, :, 0]
    green_channel = output_image[:, :, 1]
    blue_channel = output_image[:, :, 2]
    color_channels = [red_channel, green_channel, blue_channel ]
    # print(color_channels)
    # print(color_channels.shape)
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