import cv2
import numpy as np
from utils import (
    dwt_haar_1level, block_process,
    get_midband_mask, generate_pn_sequences
)
from sift_registration import register_with_sift


def _extract_from_image(img, sift_data):
    LL1, (HL1, LH1, HH1) = dwt_haar_1level(img)
    hl = HL1

    mask = get_midband_mask()
    band_indices = np.argwhere(mask)
    num_coeffs = band_indices.shape[0]

    pn_zero, pn_one = generate_pn_sequences(
        num_coeffs,
        sift_data["key"]
    )

    num_bits = sift_data["num_bits"]
    bits = []

    for i, j, block in block_process(hl, 8):
        if len(bits) >= num_bits:
            break

        dct_block = cv2.dct(block.astype(np.float32))
        coeffs = np.array([dct_block[r, c] for (r, c) in band_indices])

        corr0 = float(np.dot(coeffs, pn_zero))
        corr1 = float(np.dot(coeffs, pn_one))

        bits.append(0 if corr0 > corr1 else 1)

    bits = np.array(bits, dtype=np.uint8)
    wm_h, wm_w = sift_data["wm_shape"]
    bits = bits[:wm_h * wm_w]
    wm_img = (bits.reshape(wm_h, wm_w) * 255).astype(np.uint8)
    return wm_img


def extract_watermark(attacked_img, sift_data, use_sift=True):
    if use_sift:
        aligned = register_with_sift(
            attacked_img,
            sift_data["keypoints"],
            sift_data["descriptors"]
        )
    else:
        aligned = attacked_img

    return _extract_from_image(aligned, sift_data)
