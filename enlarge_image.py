import cv2
import numpy as np
import time
import multiprocessing
import argparse
from numba import jit
from PIL import Image
from numba import jit
from PyQt5.QtGui import QImage, QImageReader


@jit
def compute_energy(image):
    # print("Seam4")
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

@jit
def find_seam(image):
    # print("Seam3")
    energy = compute_energy(image)
    # print("Seam5")
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
    # channels_to_keep = [1, 2, 3]
    # image = image[:, :, channels_to_keep]
    new_image = np.zeros((rows, cols - 1, 4), dtype=np.uint8)
    # print(new_image.shape)
    for i in range(rows):
        j = seam[i]
        # print(i,j)
        # print(image.shape)
        # print(new_image.shape)
        new_image[i, :, :] = np.delete(image[i, :, :], j, axis=0)
    return new_image




# def visualize_seam(image, seam, color=(0, 0, 255)):
#     visualized_image = image.copy()
#
#     for i in range(len(seam)):
#         visualized_image[i, seam[i]] = color
#
#     return visualized_image

# def recursive_remove(image, target_rows, target_columns):
#     if target_rows==0 and target_columns==0:
#         return image
#
#     seam_vertical, col_energy = find_seam(image)
#     image_rotate = rotate_image(image, True)
#     seam_horizontal, row_energy = find_seam(image_rotate)

@jit
def add_seam(image, seam):
    # print("Here adding")
    rows, cols = image.shape[:2]
    output = np.zeros((rows, cols + 1, 4))
    for i in range(rows):
        j = seam[i]
        for ch in range(3):
            if j == 0:
                # print("Adding at j=0")
                new_pixel = np.average(image[i, j: j + 2, ch])
                # print("pass")
                output[i, j, ch] = image[i, j, ch]
                # print("pass")
                output[i, j + 1, ch] = new_pixel
                # print("pass")
                output[i, j + 1:, ch] = image[i, j:, ch]
            else:
                # print("Adding at else")
                new_pixel = np.average(image[i, j - 1: j + 1, ch])
                # print("pass")
                output[i, : j, ch] = image[i, : j, ch]
                # print("pass")
                output[i, j, ch] = new_pixel
                # print("pass")
                output[i, j + 1:, ch] = image[i, j:, ch]
    # print("adding done")
    return output


def enlarge(image, target):

    seam_list = []
    temp = image.copy()
    for i in range(target):
        seam = find_seam(temp)
        seam_list.append(seam)
        temp = remove_seam(temp,seam)
    # print("Seams found")
    seam_list.reverse()
    # print(seam_list)
    for i in range(target):
        seam = seam_list.pop()
        # print("Here to add")
        image = add_seam(image, seam)
        # update the remaining seam indices
        for remaining_seam in seam_list:
            remaining_seam[np.where(remaining_seam >= seam)] += 2
        print(image.shape)
    return image

def seam_carving(image, target_columns, target_rows):
    rows = 0
    cols = 0
    image = image.astype(np.float64)
    print("Starting image size: ", image.shape)
    output = enlarge(image, target_columns)
    print(output.shape)
    print("cols removed")
    rotated_image = rotate_image(output, True)
    print(rotated_image.shape)
    print("enlarge rows")
    output = enlarge(rotated_image, target_rows)
    output = rotate_image(output, False)
    print("Output image size: ", output.shape)
    return output

#
# while rows < target_rows or cols < target_columns:
#     if (target_rows == 0):
#         seam_vertical_record, col_energy = find_seam(image)
#         image = add_seam(image, seam_vertical)
#         cols += 1
#
#     elif (target_columns == 0):
#         image_rotate = rotate_image(image, True)
#         seam_horizontal, row_energy = find_seam(image_rotate)
#         image_rotate = add_seam(image_rotate, seam_horizontal)
#         image = rotate_image(image_rotate, False)
#         rows += 1
#
#     else:
#         # print("Seam2")
#         seam_vertical, col_energy = find_seam(image)
#         image_rotate = rotate_image(image, True)
#         seam_horizontal, row_energy = find_seam(image_rotate)
#         # print(row_energy,col_energy)
#         if col_energy < row_energy:
#             if cols < target_columns:
#                 image = add_seam(image, seam_vertical)
#                 cols += 1
#             else:
#                 image_rotate = add_seam(image_rotate, seam_horizontal)
#                 image = rotate_image(image_rotate, False)
#                 rows += 1
#         else:
#             if rows < target_rows:
#                 image_rotate = add_seam(image_rotate, seam_horizontal)
#                 image = rotate_image(image_rotate, False)
#                 rows += 1
#             else:
#                 image = add_seam(image, seam_vertical)
#                 cols += 1




    # return visualized_image,
    # return output_image


def algo1(qImage, row, col):

    input_image = qimage_to_numpy(qImage)
    target_rows = row
    target_columns = col
    print("Starting")
    start_time = time.time()
    #entering seam_carving
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