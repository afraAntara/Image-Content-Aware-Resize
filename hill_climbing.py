import cv2
import numpy as np
import time
import random
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
    return np.array(seam_idx), seam_energy

@jit
def remove_seam(image, seam):
    rows, cols, _ = image.shape
    new_image = np.zeros((rows, cols - 1, 4), dtype=np.uint8)
    for i in range(rows):
        j = seam[i]
        new_image[i, :, :] = np.delete(image[i, :, :], j, axis=0)
    return new_image


@jit
def random_restart_hill_climbing(image, k1, k2, calculate_energy, num_restarts, greedy_seams, mutation_rate=0.2):
    best_energy_increase = np.inf
    best_image = None
    for _ in range(num_restarts):
        current_image = image.copy()
        current_sequence = greedy_seams.copy()
        current_k1, current_k2 = k1, k2
        energy_increase = 0
        seam_sequence = []  # To store the current seam removal sequence
        seam_arrays = []    # To store the seam arrays for each step

        while current_k1 > 0 or current_k2 > 0:
            # Choose whether to remove a column or row based on the current sequence
            next_action = current_sequence.pop()

            if next_action == 'column':
                seam, seam_energy = find_seam(current_image, calculate_energy)
                current_image = remove_seam(current_image, seam)
                current_k1 -= 1
            else:
                image_rotate = rotate_image(current_image, True)
                seam, seam_energy = find_seam(image_rotate, calculate_energy)
                current_image = remove_seam(image_rotate, seam)
                current_image = rotate_image(current_image, False)

                current_k2 -= 1

            energy_increase += seam_energy
            seam_sequence.append(next_action)
            seam_arrays.append(seam.tolist())

            if random.random() < mutation_rate:
                current_sequence = mutate_by_swapping(current_sequence)


        if energy_increase < best_energy_increase:
            best_energy_increase = energy_increase
            best_image = current_image
            best_seam_sequence = seam_sequence
            best_seam_arrays = seam_arrays

    return best_image, best_energy_increase, best_seam_sequence, best_seam_arrays


@jit
def mutate_by_swapping(sequence):
    row_indices = [i for i, x in enumerate(sequence) if x == 'row']
    col_indices = [i for i, x in enumerate(sequence) if x == 'column']

    # Ensure there are both rows and columns to swap
    if row_indices and col_indices:
        row_index = random.choice(row_indices)
        col_index = random.choice(col_indices)

        # Swap the row and column
        sequence[row_index], sequence[col_index] = sequence[col_index], sequence[row_index]

    return sequence


@jit
def seam_carving(image, target_columns, target_rows, calculate_energy):
    initial_image = image.copy()
    initial_min_energy = 0
    print(image.shape)
    greedy_seams = []
    rows = 0
    cols = 0
    start_time = time.time()
    # Greedy Approach
    while rows < target_rows or cols < target_columns:
        seam_vertical, col_energy = find_seam(image, calculate_energy)
        image_rotate = rotate_image(image, True)
        seam_horizontal, row_energy = find_seam(image_rotate, calculate_energy)
        if (target_rows == 0):
            seam_vertical, col_energy = find_seam(image, calculate_energy)
            image = remove_seam(image, seam_vertical)
            initial_min_energy +=col_energy
            cols += 1

        elif (target_columns == 0):
            image_rotate = rotate_image(image, True)
            seam_horizontal, row_energy = find_seam(image_rotate, calculate_energy)
            image_rotate = remove_seam(image_rotate, seam_horizontal)
            image = rotate_image(image_rotate, False)
            initial_min_energy += row_energy
            rows += 1

        else:
            if col_energy < row_energy:
                if cols < target_columns:
                    image = remove_seam(image, seam_vertical)
                    greedy_seams.append("column")
                    cols += 1
                    initial_min_energy += col_energy
                else:
                    image_rotate = remove_seam(image_rotate, seam_horizontal)
                    image = rotate_image(image_rotate, False)
                    greedy_seams.append("row")
                    rows += 1
                    initial_min_energy += row_energy
            else:
                if rows < target_rows:
                    image_rotate = remove_seam(image_rotate, seam_horizontal)
                    image = rotate_image(image_rotate, False)
                    greedy_seams.append("row")
                    rows += 1
                    initial_min_energy += row_energy
                else:
                    image = remove_seam(image, seam_vertical)
                    greedy_seams.append("column")
                    cols += 1
                    initial_min_energy += col_energy

    end_time = time.time()
    print("Time of the normal way running in: ", (end_time - start_time))

    print(image.shape)

    start_time2 = time.time()
    initial_image, energy, seam_sequence, seam_arrays = random_restart_hill_climbing(initial_image, target_columns, target_rows, calculate_energy, 20, greedy_seams)
    #
    end_time2 = time.time()
    print("Time of the hill climbing: ", (end_time2 - start_time2))

    print(initial_image.shape)

    concatenated_image = np.concatenate((image, initial_image), axis=1)

    cv2.imshow("Greedy/Randomized", concatenated_image)
    print(initial_min_energy > energy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return initial_image

def random_hill(qImage, row, col, is_forward_energy ):

    calculate_energy = forward_energy if is_forward_energy else backward_energy
    input_image = qimage_to_numpy(qImage)
    target_rows = row
    target_columns = col
    print("Starting")
    start_time = time.time()
    output_image = seam_carving(input_image, target_columns, target_rows, calculate_energy)
    end_time = time.time()
    print("Time of the normal way running in: ", (end_time - start_time))

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

@jit
def rotate_image(image, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(image, k)