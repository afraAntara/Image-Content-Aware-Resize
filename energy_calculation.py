import cv2
import numpy as np
from numba import jit


filter_x = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

filter_y = np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]])

@jit
def backward_energy(image):

    rows, columns = image.shape[:2]
    gray_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(float)
    energy = np.zeros((rows, columns))

    gradient_x = np.zeros((rows, columns))  # Create numpy array gradient_x for Gx
    gradient_y = np.zeros((rows, columns))  # Create numpy array gradient_y for Gy
    gray_image = np.pad(gray_image, pad_width=1, mode='constant',constant_values=0)  # Add padding to grayscale image with all values surrounding as 0

    # Initialize loop 1 to iterate through rows
    for i in range(1, rows + 1):
        # Initialize loop 2 to iterate through columns
        for j in range(1, columns + 1):
            # Carrying out element-wise multiplication followed by summing values for Gx and Gy
            gradient_x[i - 1][j - 1] = np.sum(filter_x * gray_image[i - 1:i + 2, j - 1:j + 2])
            gradient_y[i - 1][j - 1] = np.sum(filter_y * gray_image[i - 1:i + 2, j - 1:j + 2])

    # Calculate energy using the Gx and Gy values, energy is the numpy array with the values
    energy = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    return energy

@jit
def forward_energy(image):
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