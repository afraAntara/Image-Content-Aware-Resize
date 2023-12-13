import cv2
import numpy as np
import multiprocessing
from numba import jit
import concurrent.futures

def process_row(i):
    m_Up = m[i - 1]
    m_Left = np.roll(m_Up, 1)
    m_Right = np.roll(m_Up, -1)

    top = np.array([m_Left, m_Up, m_Right])
    costs = np.array([cost_Left[i], cost_Mid[i], cost_Right[i]])
    top += costs

    argmins = np.argmin(top, axis=0)
    m[i] = np.choose(argmins, top)
    energy[i] = np.choose(argmins, costs)

def compute_energy(image):
    global m, energy, cost_Left, cost_Mid, cost_Right

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

    rows_to_process = range(1, rows)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_row, rows_to_process)

    return energy
