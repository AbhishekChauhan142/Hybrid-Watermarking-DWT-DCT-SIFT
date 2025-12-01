import cv2
import numpy as np

from utils import read_gray, save_gray
from dwt_dct_sift_embed import embed_watermark
from dwt_dct_sift_extract import extract_watermark
from metrics import psnr, ssim, nc, ber
from attacks import add_gaussian_noise, rotate, jpeg_compress


def main():
    # ---------- Config ----------
    LAMBDA = 0.8          # embedding strength λ (try 0.4, 0.8, 3.0, etc.)
    USE_ATTACK = False    # set True when you want to test attacks
    USE_SIFT = USE_ATTACK # use SIFT only when there is geometric distortion

    # ---------- Paths ----------
    cover_path = "../data/cover/lena.png"
    wm_path = "../data/watermark/logo32.png"   # 32x32 watermark

    watermarked_path = "../results/watermarked/lena_wm.png"
    attacked_path    = "../results/attacked/lena_attacked.png"
    extracted_path   = "../results/extracted/lena_extracted.png"

    # ---------- Embedding ----------
    print("Embedding...")
    wm_img, sift_data = embed_watermark(
        cover_path, wm_path, watermarked_path,
        key=1234, lam=LAMBDA
    )

    orig  = read_gray(cover_path)
    wm_im = read_gray(watermarked_path)

    # Imperceptibility metrics
    psnr_val = psnr(orig, wm_im)
    ssim_val = ssim(orig, wm_im)
    print(f"PSNR (no attack): {psnr_val:.2f} dB")
    print(f"SSIM (no attack): {ssim_val:.4f}")

    # Debug: max absolute pixel difference
    diff = np.abs(wm_im.astype(np.float32) - orig.astype(np.float32))
    print(f"Max |pixel diff|: {diff.max()}")

    # ---------- Scenario: with or without attack ----------
    if USE_ATTACK:
        print("Applying attack (rotation + Gaussian noise)...")
        attacked = rotate(wm_im, 10)                         # rotation 10°
        attacked = add_gaussian_noise(attacked, var=0.001)   # Gaussian noise
        save_gray(attacked_path, attacked)
    else:
        print("No attack (just copy watermarked image)...")
        attacked = wm_im.copy()
        save_gray(attacked_path, attacked)

    # ---------- Extraction ----------
    print("Extracting watermark...")
    extracted = extract_watermark(attacked, sift_data, use_sift=USE_SIFT)
    save_gray(extracted_path, extracted)

    # ---------- Metrics on watermark ----------
    wm_orig = read_gray(wm_path)
    # resize ground truth watermark to same shape used in embedding
    wm_orig = cv2.resize(
        wm_orig,
        (sift_data["wm_shape"][1], sift_data["wm_shape"][0])
    )

    nc_val = nc(wm_orig, extracted)
    ber_val = ber(wm_orig, extracted)
    print(f"NC:  {nc_val:.4f}")
    print(f"BER: {ber_val:.6f}")

    print("Done. Check results/ folder.")


if __name__ == "__main__":
    main()
