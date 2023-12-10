import cv2
import numpy as np
import time

# Energy function calculates Gx and Gy using filters filter_x and filter_y, then it calculates energy
def calculate_energy(gray_image):
    rows, columns = gray_image.shape  # Get rows and columns for grayscale image
    gradient_x = np.zeros((rows, columns))  # Create numpy array gradient_x for Gx
    gradient_y = np.zeros((rows, columns))  # Create numpy array gradient_y for Gy
    gray_image = np.pad(gray_image, pad_width=1, mode='constant', constant_values=0)  # Add padding to grayscale image with all values surrounding as 0

    # Initialize loop 1 to iterate through rows
    for i in range(1, rows+1):
        # Initialize loop 2 to iterate through columns
        for j in range(1, columns+1):
            # Carrying out element-wise multiplication followed by summing values for Gx and Gy
            gradient_x[i - 1][j - 1] = np.sum(filter_x * gray_image[i - 1:i + 2, j - 1:j + 2])
            gradient_y[i - 1][j - 1] = np.sum(filter_y * gray_image[i - 1:i + 2, j - 1:j + 2])
    
    # Calculate energy using the Gx and Gy values, energy is the numpy array with the values
    energy = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    return energy

# Search for column path with lowest energy
def find_column_seam(energy_map):
    rows, columns = energy_map.shape  # Get image size
    column_path = np.zeros(rows, dtype=int)  # Create numpy array named column_path 
    dp_table = energy_map.copy()  # Create DP table same as energy map array

    # Loop 1 to iterate through rows
    for i in range(1, rows):
        # Loop 2 to iterate through columns
        for j in range(0, columns):

            if j == 0:
                dp_table[i, j] = dp_table[i, j] + min(dp_table[i-1, j], dp_table[i-1, j+1])
            elif j == columns - 1:
                dp_table[i, j] = dp_table[i, j] + min(dp_table[i-1, j], dp_table[i-1, j-1])
            else:
                dp_table[i, j] = dp_table[i, j] + min(dp_table[i-1, j-1], dp_table[i-1, j], dp_table[i-1, j+1])

    column_path[-1] = np.argmin(dp_table[-1])  # Set last element of column_path as the last value which is minimum value from DP table
    index_i=rows-1
    index_j=column_path[index_i]
    column_insert_energy=energy_map[index_i, index_j]

    # Loop iterates from row-2 to 0 while decrementing at each step
    # The column_path array stores the indices of columns with the lowest energy that is connected and is a path through the image
    for i in range(rows-2, -1, -1):
        min_col_in_next_row = column_path[i+1]
    
        if min_col_in_next_row == 0:  
            column_path[i] = np.argmin(dp_table[i, :2])
            
        elif min_col_in_next_row == columns-1:
            #find min col in current row and adjust index
            column_path[i] = min_col_in_next_row - 1 + np.argmin(dp_table[i, columns-2:])
            
        else:
            #find min col in current row and adjust index 
            column_path[i] = min_col_in_next_row - 1 + np.argmin(dp_table[i, min_col_in_next_row-1:min_col_in_next_row+2])
        index_j=column_path[i]
        column_insert_energy+=energy_map[i,index_j]
    return column_path, column_insert_energy
    

# Search for row seam in the given energy map
def find_row_seam(energy_map):
    rows, columns = energy_map.shape
    row_path = np.zeros(columns, dtype=int)
    dp_table = energy_map.copy()

    for j in range(1, columns):
        for i in range(0, rows):
            if i == 0:
                dp_table[i, j] = dp_table[i, j] + min(dp_table[i, j-1], dp_table[i+1, j-1])
            elif i == rows-1:
                dp_table[i, j] = dp_table[i, j] + min(dp_table[i, j-1], dp_table[i-1, j-1])
            else:
                dp_table[i, j] = dp_table[i, j] + min(dp_table[i-1, j-1], dp_table[i, j-1], dp_table[i+1, j-1])

    row_path[-1] = np.argmin(dp_table[:, -1])
    index_j=columns-1
    index_i = row_path[-1]
    row_insert_energy = energy_map[index_i, index_j]
    for i in range(columns - 2, -1, -1):
        min_row_in_next_col = row_path[i + 1]
        if min_row_in_next_col == 0:
            row_path[i] = np.argmin(dp_table[:2, i])
        elif min_row_in_next_col == rows-1:
            row_path[i] = min_row_in_next_col - 1 + int(np.argmin(dp_table[rows-2:, i]))
        else:
            row_path[i] = min_row_in_next_col - 1 + int(np.argmin(dp_table[min_row_in_next_col-1:min_row_in_next_col+2, i]))
        index_i = row_path[i]
        row_insert_energy+=energy_map[index_i,i]
    
    return row_path, row_insert_energy

# # Calculate the energy for inserting a column based on the given seam
# def calculate_column_energy(energy_map, column_path):
#     row_num, col_num = gray_image.shape
#     insert_energy = 0
#     for i in range(0, row_num):
#         j= column_path[i]
#         insert_energy += energy_map[i,j]
#     return insert_energy

# # Calculate the energy for inserting a column based on the given seam
# def calculate_row_energy(energy_map, row_path):
#     row_num, col_num = gray_image.shape
#     insert_energy = 0
#     for j in range(0, col_num):
#         i= row_path[j]
#         insert_energy += energy_map[i,j]
#     return insert_energy

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



# Initialize and set values for filters fx and fy
filter_x = np.array([[-1, 0, 1],
                     [-2, 0, 2], 
                     [-1, 0, 1]])

filter_y = np.array([[-1, -2, -1], 
                     [0, 0, 0], 
                     [1, 2, 1]])

# Main function to run the system
if __name__ == "__main__":
    # Set the number of rows and columns to be removed
    target_rows = target_columns = 40

    # Load image
    image_name = 'Image2.bmp'  # Write the name of the image you are changing
    image = cv2.imread('E:/UBC/Advanced Algo/520 Project/Images/{}'.format(image_name))  # Add the location of the image

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Change the color of the image to gray
    gray_image = np.array(gray_image)  # Convert OpenCV image to numpy array
    print("input_image", gray_image.shape)

    # Initialize variables
    color_channels = []
    removed_rows = 0
    removed_columns = 0

    # Defining color channels of the original RGB image
    red_channel = np.array(image[:, :, 0])
    green_channel = np.array(image[:, :, 1])
    blue_channel = np.array(image[:, :, 2])

    # Initialize start time
    start_time = time.time()

    # Iterate to remove columns and rows
    while removed_columns < target_columns or removed_rows < target_rows:

        # Print the current shape of the grayscale image after each iteration
        print(red_channel.shape)

        # Call energy function which calculates gradient and energy
        energy_map = calculate_energy(gray_image)

        # Call to search the column_path, returns an array that has a column indices path running through the image
        column_path, column_insert_energy = find_column_seam(energy_map)

        # Call to search the row_path, returns an array that has a row indices path running through the image
        row_path, row_insert_energy = find_row_seam(energy_map)

        # column_insert_energy = calculate_column_energy(energy_map, column_path)

        # row_insert_energy = calculate_row_energy(energy_map, row_path)
        # print(row_insert_energy, column_insert_energy)
        
        if column_insert_energy < row_insert_energy:
            if removed_columns < target_columns:
                gray_image = remove_column(gray_image, column_path)
                red_channel = remove_column(red_channel, column_path)
                green_channel = remove_column(green_channel, column_path)
                blue_channel = remove_column(blue_channel, column_path)
                removed_columns += 1
            else:
                gray_image = remove_row(gray_image, row_path)
                red_channel = remove_row(red_channel, row_path)
                green_channel = remove_row(green_channel, row_path)
                blue_channel = remove_row(blue_channel, row_path)
                removed_rows += 1
        elif row_insert_energy < column_insert_energy:
            if removed_rows < target_rows:
                gray_image = remove_row(gray_image, row_path)
                red_channel = remove_row(red_channel, row_path)
                green_channel = remove_row(green_channel, row_path)
                blue_channel = remove_row(blue_channel, row_path)
                removed_rows += 1
            else:
                # gray_image = remove_column(gray_image, column_path)
                red_channel = remove_column(red_channel, column_path)
                green_channel = remove_column(green_channel, column_path)
                blue_channel = remove_column(blue_channel, column_path)
                removed_columns += 1


    end_time = time.time()
    print("Time of the normal way running in {}: {}".format(image_name, end_time - start_time))
    color_channels.append(red_channel)
    color_channels.append(green_channel)
    color_channels.append(blue_channel)
    color_image = np.array(color_channels)
    color_image = np.transpose(color_image, axes=(1, 2, 0))
    color_image = color_image.astype('uint8')
    print("output_image", color_image.shape)
    cv2.imwrite('E:/UBC/Advanced Algo/520 Project/Images/outputs/{}.jpg'.format(image_name), color_image)
    print("done")
