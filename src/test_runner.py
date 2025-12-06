# src/test_runner.py
"""
Test runner: embed -> attack -> extract -> metrics
Prints BER and NC (Normalized Correlation), PSNR and SSIM.
"""

import argparse
import os
import cv2
import numpy as np
from robust_watermark import embed_watermark, extract_watermark, attack_jpeg, attack_gaussian_noise, attack_rotate, attack_crop, compute_ber, evaluate_images

def compute_nc(original_bits: np.ndarray, extracted_bits: np.ndarray) -> float:
    """
    Normalized correlation for binary watermarks.
    Map {0,1} -> {-1, +1} then compute mean of product.
    Returns value in [-1, 1], with 1 = perfect match.
    """
    a = original_bits.flatten().astype(np.int8)
    b = extracted_bits.flatten().astype(np.int8)
    a = 2 * a - 1
    b = 2 * b - 1
    nc = float(np.sum(a * b)) / float(a.size)
    return nc

def run_pipeline(cover_path, wm_path, out_wm_path, attacked_path, extracted_path,
                 alpha=12, size=64, block_size=4, subband="HH",
                 attack_type="jpeg", quality=80, sigma=10.0, angle=5.0, crop_frac=0.1,
                 use_registration=True):
    os.makedirs(os.path.dirname(out_wm_path), exist_ok=True)
    os.makedirs(os.path.dirname(attacked_path), exist_ok=True)
    os.makedirs(os.path.dirname(extracted_path), exist_ok=True)

    # load
    cover = cv2.imread(cover_path)
    if cover is None:
        raise FileNotFoundError("Cover not found: " + cover_path)
    wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
    if wm is None:
        raise FileNotFoundError("Watermark not found: " + wm_path)

    # prepare watermark bits
    wm_resized = cv2.resize(wm, (size, size))
    wm_bin = (wm_resized > 128).astype(np.uint8)

    # embed
    watermarked = embed_watermark(cover, wm_bin, alpha=alpha, subband=subband, block_size=block_size)
    cv2.imwrite(out_wm_path, watermarked)
    print("Saved watermarked:", out_wm_path)

    # attack
    if attack_type == "jpeg":
        attacked = attack_jpeg(watermarked, quality=quality)
    elif attack_type == "noise":
        attacked = attack_gaussian_noise(watermarked, sigma=sigma)
    elif attack_type == "rotate":
        attacked = attack_rotate(watermarked, angle=angle)
    elif attack_type == "crop":
        attacked = attack_crop(watermarked, crop_frac=crop_frac)
    else:
        raise ValueError("Unsupported attack type")

    cv2.imwrite(attacked_path, attacked)
    print("Saved attacked:", attacked_path)

    # extract
    extracted_bits = extract_watermark(attacked, cover, (size, size), alpha=alpha, subband=subband, block_size=block_size, use_registration=use_registration)
    cv2.imwrite(extracted_path, (extracted_bits * 255).astype(np.uint8))
    print("Saved extracted:", extracted_path)

    # metrics
    ber = compute_ber(wm_bin, extracted_bits)
    nc = compute_nc(wm_bin, extracted_bits)
    img_metrics = evaluate_images(cover, watermarked)

    print("\n--- METRICS ---")
    print(f"BER: {ber:.4f}")
    print(f"NC : {nc:.4f}")
    print(f"PSNR: {img_metrics['psnr']:.2f}")
    print(f"SSIM: {img_metrics['ssim']:.4f}")

    return {
        "ber": ber,
        "nc": nc,
        "psnr": img_metrics['psnr'],
        "ssim": img_metrics['ssim']
    }

def main():
    parser = argparse.ArgumentParser(description="Test runner for watermarking pipeline")
    parser.add_argument("--cover", required=True)
    parser.add_argument("--wm", required=True)
    parser.add_argument("--out_wm", default="results/watermarked/test_wm.png")
    parser.add_argument("--attacked", default="results/attacked/test_attacked.jpg")
    parser.add_argument("--extracted", default="results/extracted/test_extracted.png")
    parser.add_argument("--alpha", type=float, default=12)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--subband", choices=["HH","HL","LH"], default="HH")
    parser.add_argument("--attack", choices=["jpeg","noise","rotate","crop"], default="jpeg")
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--angle", type=float, default=5.0)
    parser.add_argument("--crop_frac", type=float, default=0.1)
    parser.add_argument("--no_register", action="store_true", help="disable SIFT registration during extraction")

    args = parser.parse_args()

    run_pipeline(
        args.cover, args.wm, args.out_wm, args.attacked, args.extracted,
        alpha=args.alpha, size=args.size, block_size=args.block_size, subband=args.subband,
        attack_type=args.attack, quality=args.quality, sigma=args.sigma, angle=args.angle, crop_frac=args.crop_frac,
        use_registration=not args.no_register
    )

if __name__ == "__main__":
    main()
