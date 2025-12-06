import cv2
import numpy as np
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from typing import Tuple

# Optional ECC (reedsolo)
try:
    from reedsolo import RSCodec
    ECC_AVAILABLE = True
except Exception:
    ECC_AVAILABLE = False


def register_images_sift(src_gray: np.ndarray, dst_gray: np.ndarray, min_matches=8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Register src_gray onto dst_gray using SIFT + FLANN + RANSAC.
    Returns (aligned_src_gray, homography) or (src_gray, None) if registration fails.
    """
    # Ensure uint8
    s = (src_gray.astype(np.uint8) if src_gray.dtype != np.uint8 else src_gray)
    d = (dst_gray.astype(np.uint8) if dst_gray.dtype != np.uint8 else dst_gray)

    try:
        sift = cv2.SIFT_create()
    except Exception:
        # fallback to ORB if SIFT unavailable
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(s, None)
        kp2, des2 = orb.detectAndCompute(d, None)
        if des1 is None or des2 is None:
            return src_gray, None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < min_matches:
            return src_gray, None
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return src_gray, None
        h, w = dst_gray.shape
        aligned = cv2.warpPerspective(src_gray, H, (w, h))
        return aligned, H

    kp1, des1 = sift.detectAndCompute(s, None)
    kp2, des2 = sift.detectAndCompute(d, None)
    if des1 is None or des2 is None:
        return src_gray, None

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) < min_matches:
        return src_gray, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return src_gray, None

    h, w = dst_gray.shape
    aligned = cv2.warpPerspective(src_gray, H, (w, h))
    return aligned, H


def embed_watermark(cover: np.ndarray,
                    wm_bits: np.ndarray,
                    alpha: float = 12,
                    wavelet: str = "haar",
                    subband: str = "HH",
                    block_size: int = 8) -> np.ndarray:
    """
    Embed a binary watermark (wm_bits) into cover image.
    wm_bits: 2D numpy array of 0/1 (height, width)
    block_size: size of DCT blocks in subband (change to 4 to increase capacity)
    """
    # Convert to Y channel
    if cover.ndim == 3:
        ycrcb = cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        Y = ycrcb[:, :, 0]
    else:
        Y = cover.astype(np.float32)

    # Single-level DWT
    cA, (cH, cV, cD) = pywt.dwt2(Y, wavelet)

    # select band
    if subband == "HH":
        band = cD.copy()
    elif subband == "HL":
        band = cV.copy()
    elif subband == "LH":
        band = cH.copy()
    else:
        raise ValueError("subband must be one of 'HH','HL','LH'")

    h, w = band.shape
    wm_flat = wm_bits.flatten()
    max_blocks = (h // block_size) * (w // block_size)
    if wm_flat.size > max_blocks:
        raise ValueError(f"Watermark too large for chosen parameters. capacity={max_blocks}, required={wm_flat.size}")

    idx = 0
    for i in range(0, h - (h % block_size), block_size):
        for j in range(0, w - (w % block_size), block_size):
            if idx >= wm_flat.size:
                break
            block = band[i:i+block_size, j:j+block_size].astype(np.float32)
            d = cv2.dct(block)
            # embed into a mid-frequency coefficient
            coef_pos = (1, 2) if block_size >= 4 else (2, 2)
            d[coef_pos] += alpha * (1 if wm_flat[idx] else -1)
            band[i:i+block_size, j:j+block_size] = cv2.idct(d)
            idx += 1
        if idx >= wm_flat.size:
            break

    # put band back
    if subband == "HH":
        cD = band
    elif subband == "HL":
        cV = band
    else:
        cH = band

    # inverse DWT
    rec = pywt.idwt2((cA, (cH, cV, cD)), wavelet)
    rec = np.clip(rec, 0, 255)

    if cover.ndim == 3:
        ycrcb[:, :, 0] = rec
        out = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    else:
        out = rec.astype(np.uint8)

    return out


def extract_watermark(attacked: np.ndarray,
                      original_cover: np.ndarray,
                      wm_shape: Tuple[int, int],
                      alpha: float = 12,
                      wavelet: str = "haar",
                      subband: str = "HH",
                      block_size: int = 8,
                      use_registration: bool = True) -> np.ndarray:
    """
    Extract watermark bits from attacked image using original cover for reference.
    wm_shape: (h, w)
    """
    # Prepare grayscale Y channels
    if attacked.ndim == 3:
        attacked_gray = cv2.cvtColor(attacked, cv2.COLOR_BGR2GRAY)
    else:
        attacked_gray = attacked.copy()
    if original_cover.ndim == 3:
        orig_gray = cv2.cvtColor(original_cover, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = original_cover.copy()

    aligned = attacked_gray
    if use_registration:
        aligned, H = register_images_sift(attacked_gray, orig_gray)
        if H is None:
            aligned = attacked_gray  # fallback

    # DWT on aligned attacked and original
    cA_a, (cH_a, cV_a, cD_a) = pywt.dwt2(aligned.astype(np.float32), wavelet)
    cA_o, (cH_o, cV_o, cD_o) = pywt.dwt2(orig_gray.astype(np.float32), wavelet)

    if subband == "HH":
        band_att, band_orig = cD_a, cD_o
    elif subband == "HL":
        band_att, band_orig = cV_a, cV_o
    elif subband == "LH":
        band_att, band_orig = cH_a, cH_o
    else:
        raise ValueError("subband must be one of 'HH','HL','LH'")

    h, w = band_att.shape
    expected_bits = wm_shape[0] * wm_shape[1]
    bits = []

    for i in range(0, h - (h % block_size), block_size):
        for j in range(0, w - (w % block_size), block_size):
            if len(bits) >= expected_bits:
                break
            block_a = band_att[i:i+block_size, j:j+block_size].astype(np.float32)
            block_o = band_orig[i:i+block_size, j:j+block_size].astype(np.float32)
            d_a = cv2.dct(block_a)
            d_o = cv2.dct(block_o)
            coef_pos = (1, 2) if block_size >= 4 else (2, 2)
            diff = d_a[coef_pos] - d_o[coef_pos]
            bits.append(1 if diff > 0 else 0)
        if len(bits) >= expected_bits:
            break

    bits = np.array(bits, dtype=np.uint8)
    if bits.size < expected_bits:
        # pad with zeros if insufficient bits found
        pad = np.zeros(expected_bits - bits.size, dtype=np.uint8)
        bits = np.concatenate([bits, pad])

    return bits[:expected_bits].reshape(wm_shape)


# --- attack helpers ---
def attack_jpeg(img: np.ndarray, quality: int = 80) -> np.ndarray:
    _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def attack_gaussian_noise(img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    noisy = img.astype(np.float32) + np.random.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def attack_rotate(img: np.ndarray, angle: float = 5.0) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def attack_crop(img: np.ndarray, crop_frac: float = 0.1) -> np.ndarray:
    h, w = img.shape[:2]
    ch = int(h * (1 - crop_frac))
    cw = int(w * (1 - crop_frac))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    cropped = img[y0:y0+ch, x0:x0+cw]
    return cv2.resize(cropped, (w, h))


# --- metrics ---
def compute_ber(original_bits: np.ndarray, extracted_bits: np.ndarray) -> float:
    total = original_bits.size
    error = np.count_nonzero(original_bits.flatten() != extracted_bits.flatten())
    return float(error) / float(total)


def evaluate_images(original: np.ndarray, watermarked: np.ndarray) -> dict:
    """
    Compute PSNR and SSIM in a robust way:
    - Ensure same shape (resize watermarked to original if necessary)
    - Convert dtypes to uint8 if needed
    - Use channel_axis for color images (skimage newer API)
    - On any SSIM failure, return PSNR and ssim=None instead of crashing
    """
    try:
        # Convert to uint8 if float
        def to_uint8(img):
            if img.dtype == np.float32 or img.dtype == np.float64:
                # assume range [0,255]
                img_u8 = np.clip(img, 0, 255).astype(np.uint8)
                return img_u8
            return img

        orig = to_uint8(original)
        wm = to_uint8(watermarked)

        # If different sizes, resize watermarked to match original
        if orig.shape[:2] != wm.shape[:2]:
            wm = cv2.resize(wm, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_AREA)

        # PSNR
        p = psnr(orig, wm, data_range=255)

        # SSIM: handle color vs grayscale and library differences
        s = None
        try:
            if orig.ndim == 3 and orig.shape[2] in (3, 4):
                # color image; use channel_axis (preferred) or fallback to multichannel
                try:
                    s = ssim(orig, wm, channel_axis=2, data_range=255)
                except TypeError:
                    # older skimage versions expect multichannel arg
                    s = ssim(orig, wm, multichannel=True, data_range=255)
            else:
                # grayscale
                s = ssim(orig, wm, data_range=255)
        except Exception:
            # SSIM failed (e.g., image too small or other edge case)
            s = None

        return {'psnr': float(p), 'ssim': (float(s) if s is not None else None)}

    except Exception as e:
        # Worst case fallback: only return PSNR if possible
        try:
            orig = np.clip(original, 0, 255).astype(np.uint8)
            wm = np.clip(watermarked, 0, 255).astype(np.uint8)
            p = psnr(orig, wm, data_range=255)
            return {'psnr': float(p), 'ssim': None}
        except Exception:
            # give up gracefully
            return {'psnr': None, 'ssim': None}

