import cv2
import numpy as np


def add_gaussian_noise(img, var=0.001):
    img_f = img.astype(np.float32) / 255.0
    noise = np.random.normal(0, var ** 0.5, img.shape)
    noisy = np.clip(img_f + noise, 0, 1)
    return (noisy * 255).astype(np.uint8)


def rotate(img, angle):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def jpeg_compress(path_in, path_out, quality=60):
    img = cv2.imread(path_in)
    cv2.imwrite(path_out, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
