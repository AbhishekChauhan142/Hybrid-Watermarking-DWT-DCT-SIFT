import cv2
import numpy as np

from utils import (
    read_gray, save_gray,
    dwt_haar_1level, idwt_haar_1level,
    block_process, get_midband_mask,
    generate_pn_sequences,
)


def embed_watermark(cover_path, wm_path, out_path,
                    key: int = 1234, lam: float = 0.4):
    """
    DWT-DCT watermark embedding (HL1, 8x8 blocks, 22 mid-band coeffs).
    - cover_path: path to 512x512 grayscale cover image
    - wm_path:    path to watermark image (we'll resize to 32x32)
    - out_path:   path to save watermarked image
    - key:        seed for PN sequence
    - lam:        embedding strength λ
    """
    print(f"[embed_watermark] Using lambda = {lam}")

    # ---- Load & prepare images ----
    cover = read_gray(cover_path)
    cover = cv2.resize(cover, (512, 512))

    wm = read_gray(wm_path)
    wm = cv2.resize(wm, (32, 32))        # 32x32 = 1024 bits
    wm_bits = (wm > 127).astype(np.uint8).flatten()

    # ---- 1-level DWT ----
    LL1, (HL1, LH1, HH1) = dwt_haar_1level(cover)
    hl = HL1.copy()

    # ---- PN sequences & mid-band mask ----
    mask = get_midband_mask()
    band_indices = np.argwhere(mask)          # shape (22, 2)
    num_coeffs = band_indices.shape[0]        # 22

    pn_zero, pn_one = generate_pn_sequences(num_coeffs, key)

    h, w = hl.shape
    bit_idx = 0

    # ---- Block-wise DCT & embedding ----
    for i, j, block in block_process(hl, 8):
        if bit_idx >= len(wm_bits):
            break

        dct_block = cv2.dct(block.astype(np.float32))
        bit = int(wm_bits[bit_idx])
        seq = pn_zero if bit == 0 else pn_one

        # Y = X + λ * PN  on 22 mid-band coefficients
        for k, (r, c) in enumerate(band_indices):
            dct_block[r, c] = dct_block[r, c] + float(lam) * float(seq[k])

        hl[i:i+8, j:j+8] = cv2.idct(dct_block)
        bit_idx += 1

    # ---- Rebuild image via IDWT ----
    watermarked = idwt_haar_1level(LL1, hl, LH1, HH1)
    watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
    save_gray(out_path, watermarked)

    # ---- SIFT keypoints/descriptors on watermarked image ----
    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(watermarked, None)

    if kps is None:
        kp_arr = np.empty((0, 4), dtype=np.float32)
    else:
        kp_arr = np.array(
            [kp.pt + (kp.size, kp.angle,) for kp in kps],
            dtype=np.float32
        )

    sift_data = {
        "keypoints": kp_arr,
        "descriptors": des,
        "wm_shape": wm.shape,      # (32, 32)
        "num_bits": bit_idx,       # should be 1024 if all blocks used
        "key": key,
        "lam": lam,
    }

    print(f"[embed_watermark] Embedded {bit_idx} bits.")
    return watermarked, sift_data
