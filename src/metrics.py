import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(orig, wm):
    return peak_signal_noise_ratio(orig, wm, data_range=255)


def ssim(orig, wm):
    return structural_similarity(orig, wm, data_range=255)


def nc(w, w_rec):
    w = w.astype(np.float32).flatten()
    w_rec = w_rec.astype(np.float32).flatten()
    num = float((w * w_rec).sum())
    den = float((w ** 2).sum() ** 0.5 * (w_rec ** 2).sum() ** 0.5)
    return num / den if den != 0 else 0.0


def ber(w, w_rec):
    wb = (w > 127).astype(np.uint8).flatten()
    wr = (w_rec > 127).astype(np.uint8).flatten()
    diff = (wb ^ wr).sum()
    return diff / len(wb)
