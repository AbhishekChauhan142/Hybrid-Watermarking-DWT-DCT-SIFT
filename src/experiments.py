import os
import cv2
import numpy as np
from robust_watermark import embed_watermark, extract_watermark, attack_jpeg
from metrics import image_metrics, compute_ber

COVER_DIR = "data/cover"
WM_DIR = "data/watermark"
RESULTS = "data/results"

def run_experiment(cover_path, wm_path, alpha=12):
    cover = cv2.imread(cover_path)
    wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
    wm_bin = (cv2.resize(wm, (32, 32)) > 128).astype(np.uint8)

    watermarked = embed_watermark(cover, wm_bin, alpha=alpha)
    wm_file = os.path.join(RESULTS, "watermarked", "watermarked.png")
    cv2.imwrite(wm_file, watermarked)

    attacked = attack_jpeg(watermarked, quality=30)
    atk_file = os.path.join(RESULTS, "attacked", "attacked.jpg")
    cv2.imwrite(atk_file, attacked)

    extracted = extract_watermark(attacked, cover, wm_bin.shape, alpha=alpha)
    ext_file = os.path.join(RESULTS, "extracted", "extracted.png")
    cv2.imwrite(ext_file, (extracted * 255).astype(np.uint8))

    ber = compute_ber(wm_bin, extracted)
    metrics = image_metrics(cover, watermarked)

    print("\n--- Results ---")
    print("BER:", ber)
    print("PSNR:", metrics["psnr"])
    print("SSIM:", metrics["ssim"])

if __name__ == '__main__':
    covers = os.listdir(COVER_DIR)
    watermarks = os.listdir(WM_DIR)

    if not covers or not watermarks:
        print("No images found in cover/ or watermark/.")
        exit()

    cover_path = os.path.join(COVER_DIR, covers[0])
    wm_path = os.path.join(WM_DIR, watermarks[0])

    run_experiment(cover_path, wm_path)
