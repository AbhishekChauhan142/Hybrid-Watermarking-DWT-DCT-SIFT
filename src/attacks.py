# src/attacks.py
import argparse
import cv2
import numpy as np
import os
from robust_watermark import attack_jpeg, attack_gaussian_noise, attack_rotate, attack_crop

def main():
    parser = argparse.ArgumentParser(description="Apply attacks to an image (wrapping robust_watermark attack functions).")
    parser.add_argument("--img", required=True, help="Input image path")
    parser.add_argument("--type", choices=["jpeg","noise","rotate","crop"], default="jpeg", help="Attack type")
    parser.add_argument("--out", default="results/attacked/attacked.png", help="Output attacked image path")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality (only for jpeg attack)")
    parser.add_argument("--sigma", type=float, default=10.0, help="Gaussian noise sigma (only for noise)")
    parser.add_argument("--angle", type=float, default=5.0, help="Rotation angle in degrees (only for rotate)")
    parser.add_argument("--crop_frac", type=float, default=0.1, help="Crop fraction (only for crop)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(args.img)

    if args.type == "jpeg":
        out_img = attack_jpeg(img, quality=args.quality)
    elif args.type == "noise":
        out_img = attack_gaussian_noise(img, sigma=args.sigma)
    elif args.type == "rotate":
        out_img = attack_rotate(img, angle=args.angle)
    elif args.type == "crop":
        out_img = attack_crop(img, crop_frac=args.crop_frac)
    else:
        raise ValueError("Unsupported attack type")

    cv2.imwrite(args.out, out_img)
    print("Saved attacked image to:", args.out)

if __name__ == "__main__":
    main()
