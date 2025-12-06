from robust_watermark import embed_watermark
import cv2
import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cover', required=True)
    parser.add_argument('--watermark', required=True)
    parser.add_argument('--out', default='data/results/watermarked/output.png')
    parser.add_argument('--alpha', type=float, default=12)
    args = parser.parse_args()

    cover = cv2.imread(args.cover)
    wm = cv2.imread(args.watermark, cv2.IMREAD_GRAYSCALE)
    wm_bin = (cv2.resize(wm, (32, 32)) > 128).astype(np.uint8)

    result = embed_watermark(cover, wm_bin, alpha=args.alpha)
    cv2.imwrite(args.out, result)
    print("Saved watermarked image to:", args.out)
