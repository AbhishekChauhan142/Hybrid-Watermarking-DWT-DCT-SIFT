import cv2
import numpy as np
import pywt
import os


def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def save_gray(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def dwt_haar_1level(img):
    return pywt.dwt2(img.astype(np.float32), 'haar')


def idwt_haar_1level(LL, HL, LH, HH):
    return pywt.idwt2((LL, (HL, LH, HH)), 'haar')


def block_process(mat, block_size=8):
    h, w = mat.shape
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if i + block_size <= h and j + block_size <= w:
                yield i, j, mat[i:i+block_size, j:j+block_size]


def get_midband_mask():
    mask = np.zeros((8, 8), dtype=bool)
    positions = [
        (0, 3), (0, 4), (1, 2), (1, 3), (1, 4),
        (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 0), (3, 1), (3, 2), (3, 3),
        (4, 0), (4, 1), (4, 2),
        (5, 0), (5, 1),
        (2, 5), (3, 4), (4, 3), (5, 2),
    ]
    for (r, c) in positions:
        mask[r, c] = True
    return mask


def generate_pn_sequences(length, key):
    rng = np.random.RandomState(key)
    pn_zero = rng.choice([-1, 1], size=length)
    pn_one = rng.choice([-1, 1], size=length)
    return pn_zero, pn_one
