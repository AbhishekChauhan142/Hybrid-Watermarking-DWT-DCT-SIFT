import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_ber(original_bits, extracted_bits):
    errors = np.count_nonzero(original_bits.flatten() != extracted_bits.flatten())
    return errors / original_bits.size

def image_metrics(img1, img2):
    return {
        "psnr": psnr(img1, img2, data_range=255),
        "ssim": ssim(img1, img2, multichannel=True, data_range=255)
    }
