import cv2
import numpy as np
import time
import multiprocessing
from PIL import Image
from PyQt5.QtGui import QImage, QImageReader

# Main function to run the system
def algo1(qImage, row, col):
    image = qimage_to_numpy(qImage)
    target_rows = row
    target_columns = col
    # change image to gray scale
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # change color of image to gray
    gray_image = np.array(gray_scale)  # convert openCV image to numpy array
    # print("input_image", gray_image.shape)
    # Initialize variables
    color_channels = []
    removed_rows = 0
    removed_columns = 0
    # Defining color channels of the original RGB image
    red_channel = np.array(image[:, :, 0])
    green_channel = np.array(image[:, :, 1])
    blue_channel = np.array(image[:, :, 2])
    # images = [gray_image, red_channel, green_channel, blue_channel]
    # removed_images = [gray_image, red_channel, green_channel, blue_channel]
    # Initialize start time
    start_time = time.time()
    # Iterate to remove columns and rows
    while removed_columns < target_columns or removed_rows < target_rows:
        print("Here")
        # Call energy function which calculates gradient and energy
        energy_map_column = calculate_energy_for_column(gray_image)
        print("Here")
        # Call to search the column_path, returns an array that has a column indices path running through the image
        column_path = find_column_seam(energy_map_column, gray_image)

        # Call energy function which calculates gradient and energy
        # energy_map_row = calculate_energy_for_row(gray_image)
        # Call to search the row_path, returns an array that has a row indices path running through the image
        # row_path = find_row_seam(energy_map_row, gray_image)

        # if column_insert_energy < row_insert_energy:
        # if removed_columns < target_columns:
        red_channel = remove_column(red_channel, column_path)
        green_channel = remove_column(green_channel, column_path)
        blue_channel = remove_column(blue_channel, column_path)
        removed_columns += 1
        # elif removed_rows < target_rows:
        # else:
        #     red_channel = remove_row(red_channel, row_path)
        #     green_channel = remove_row(green_channel, row_path)
        #     blue_channel = remove_row(blue_channel, row_path)
        #     removed_rows += 1
        # elif row_insert_energy < column_insert_energy:
        #     if removed_rows < target_rows:
        #         images = remove_rows(images, row_path)
        #         removed_rows += 1
        #     else:
        #         images = remove_cols(images, column_path)
        #         removed_columns += 1
        # Print the current shape of the grayscale image after each iteration
        print(red_channel.shape)

    end_time = time.time()
    time_all = end_time-start_time
    print("Time of the normal way running in: ", time_all)
    color_channels.append(red_channel)
    color_channels.append(green_channel)
    color_channels.append(blue_channel)
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


# Energy function calculates Gx and Gy using filters filter_x and filter_y, then it calculates energy
def calculate_energy_for_column(gray_image):
    rows, columns = gray_image.shape
    gray_image = np.pad(gray_image, pad_width=1, mode='constant',constant_values=0)  # Add padding to grayscale image with all values surrounding as 0
    initial_energy = np.zeros((rows, columns))
    print("a", gray_image.shape())
    print("a", initial_energy.shape())
    # Initialize loop 1 to iterate through rows
    for i in range(1, rows+1):
        # Initialize loop 2 to iterate through columns
        for j in range(1, columns+1):
            initial_energy[i-1][j-1] = np.abs(gray_image[i][j - 1] - gray_image[i][j + 1])
    print(initial_energy)
    return initial_energy



# Search for column path with lowest energy
def find_column_seam(energy_map, gray_image):
    rows, columns = energy_map.shape  # Get image size
    column_path = np.zeros(rows, dtype=int)  # Create numpy array named column_path
    dp_table = energy_map.copy()  # Create DP table same as energy map array
    print("b",energy_map.shape())
    for i in range(1, rows):
        for j in range(0, columns):
            print("row-col", i, j)
            if j == 0:
                top = energy_map[i - 1][j]
                top_right =  energy_map[i - 1][j + 1] + np.abs(gray_image[i - 1][j] - gray_image[i][j + 1])
                dp_table[i, j] = dp_table[i, j] + min(top, top_right)
            elif j == columns - 1:
                top_left = energy_map[i - 1][j - 1]  + np.abs(gray_image[i - 1][j] - gray_image[i][j - 1])
                top = energy_map[i - 1][j]
                dp_table[i, j] = dp_table[i, j] + min(top_left, top)
            else:
                top_left = energy_map[i - 1][j - 1]  + np.abs(gray_image[i - 1][j] - gray_image[i][j - 1])
                top = energy_map[i - 1][j]
                top_right = energy_map[i - 1][j + 1] + np.abs(gray_image[i - 1][j] - gray_image[i][j + 1])
                dp_table[i, j] = dp_table[i, j] + min(top_left, top, top_right)
    column_path[-1] = np.argmin(dp_table[-1])  # Set last element of column_path as the last value which is minimum value from DP table

    # Loop iterates from row-2 to 0 while decrementing at each step
    # The column_path array stores the indices of columns with the lowest energy that is connected and is a path through the image
    for i in range(rows - 2, -1, -1):
        min_col_in_next_row = column_path[i + 1]

        if min_col_in_next_row == 0:
            column_path[i] = np.argmin(dp_table[i, :2])

        elif min_col_in_next_row == columns - 1:
            # find min col in current row and adjust index
            column_path[i] = min_col_in_next_row - 1 + np.argmin(dp_table[i, columns - 2:])

        else:
            # find min col in current row and adjust index
            column_path[i] = min_col_in_next_row - 1 + np.argmin(
                dp_table[i, min_col_in_next_row - 1:min_col_in_next_row + 2])

    return column_path

# Energy function calculates Gx and Gy using filters filter_x and filter_y, then it calculates energy
def calculate_energy_for_row(gray_image):
    rows, columns = gray_image.shape
    gray_image = np.pad(gray_image, pad_width=1, mode='constant',constant_values=0)  # Add padding to grayscale image with all values surrounding as 0
    initial_energy = np.zeros((rows, columns))
    for i in range(1, rows+1):
        for j in range(1, columns+1):
            initial_energy[i-1][j-1] = np.abs(gray_image[i-1][j] - gray_image[i+1][j])
    return initial_energy

# Search for row seam in the given energy map
def find_row_seam(energy_map, gray_image):
    rows, columns = energy_map.shape
    # print("energy",energy_map.shape)
    # print("gray",gray_image.shape)
    row_path = np.zeros(columns, dtype=int)
    dp_table = energy_map.copy()  # Create DP table same as energy map array
    for i in range(0, rows):
        for j in range(1, columns):
            if i == 0:
                top = energy_map[i][j - 1]
                top_left =  energy_map[i + 1][j - 1] + np.abs(gray_image[i][j - 1] - gray_image[i + 1][j])
                dp_table[i, j] = dp_table[i, j] + min(top, top_left)
            elif i == rows - 1:
                top = energy_map[i][j - 1]
                top_right = energy_map[i - 1][j - 1]  + np.abs(gray_image[i][j - 1] - gray_image[i - 1][j])
                dp_table[i, j] = dp_table[i, j] + min(top_right, top)
            else:
                top_left =  energy_map[i + 1][j - 1] + np.abs(gray_image[i][j - 1] - gray_image[i + 1][j])
                top = energy_map[i][j - 1]
                top_right = energy_map[i - 1][j - 1] + np.abs(gray_image[i][j - 1] - gray_image[i - 1][j])
                dp_table[i, j] = dp_table[i, j] + min(top_left, top, top_right)

    row_path[-1] = np.argmin(dp_table[:, -1])
    for i in range(columns - 2, -1, -1):
        min_row_in_next_col = row_path[i + 1]
        if min_row_in_next_col == 0:
            row_path[i] = np.argmin(dp_table[:2, i])
        elif min_row_in_next_col == rows - 1:
            row_path[i] = min_row_in_next_col - 1 + int(np.argmin(dp_table[rows - 2:, i]))
        else:
            row_path[i] = min_row_in_next_col - 1 + int(
                np.argmin(dp_table[min_row_in_next_col - 1:min_row_in_next_col + 2, i]))

    return row_path

# Calculate the energy for inserting a column based on the given seam
def calculate_column_energy(energy, column_path):
    row_num, col_num = energy.shape
    insert_energy = 0
    for i in range(1, row_num):
        j = column_path[i]
        if column_path[i - 1] == column_path[i] and j > 0 and j < col_num - 1:
            insert_energy += (energy[i, j - 1] - energy[i, j + 1])
        elif column_path[i - 1] < column_path[i] and j < col_num - 1:  # right -1
            insert_energy += (energy[i, j - 1] - energy[i - 1, j - 1]) + (energy[i, j - 1] - energy[i, j + 1])
        elif column_path[i - 1] > column_path[i] and j > 0:  # right +1
            insert_energy += (energy[i, j - 1] - energy[i, j + 1]) + (energy[i, j + 1] - energy[i - 1, j + 1])
    return insert_energy

# Calculate the energy for inserting a row based on the given seam
def calculate_row_energy(energy, row_path):
    row_num, col_num = energy.shape
    insert_energy = 0
    for j in range(1, col_num):
        i = row_path[j]
        if row_path[j - 1] == row_path[j] and i > 0 and i < row_num - 1:
            insert_energy += energy[i + 1, j] - energy[i - 1, j]
        elif row_path[j - 1] < row_path[j]  and i < row_num - 1:
            insert_energy += (energy[i - 1, j] - energy[i + 1, j]) + (energy[i - 1, j] - energy[i, j - 1])
        elif row_path[j - 1] > row_path[j] and i > 0:
            insert_energy += (energy[i + 1, j] - energy[i, j - 1]) + (energy[i + 1, j] - energy[i - 1, j])
    return insert_energy


# Delete the column from the given gray scale image based on the seam by iterating through column path and removing that pixel which falls in column
def remove_column(gray_image, column_path):
    row_num, col_num = gray_image.shape
    result = np.zeros((row_num, col_num-1))
    for i in range(row_num):
        col_index = column_path[i]
        result[i, :] = np.delete(gray_image[i, :], col_index)
    return result


# Delete the row from the given gray scale image based on the seam by iterating through row path and removing that pixel which falls in row
def remove_row(gray_image, row_path):
    row_num, col_num = gray_image.shape
    result = np.zeros((row_num-1, col_num))
    for j in range(0, col_num):
        row_index = row_path[j]
        result[:, j] = np.delete(gray_image[:, j], row_index)
    return result

# def remove_cols(images, column_path):
#     images[0] = remove_column(images[0], column_path)
#     images[1] = remove_column(images[1], column_path)
#     images[2] = remove_column(images[2], column_path)
#     images[3] = remove_column(images[3], column_path)
#     return images
#
# def remove_rows(images, row_path):
#     images[0] = remove_row(images[0], row_path)
#     images[1] = remove_row(images[1], row_path)
#     images[2] = remove_row(images[2], row_path)
#     images[3] = remove_row(images[3], row_path)
#     return images

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



