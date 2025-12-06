from robust_watermark import extract_watermark
import cv2
import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--attacked', required=True)
    parser.add_argument('--original', required=True)
    parser.add_argument('--out', default='data/results/extracted/output.png')
    parser.add_argument('--alpha', type=float, default=12)
    args = parser.parse_args()

    attacked = cv2.imread(args.attacked)
    original = cv2.imread(args.original)

    extracted = extract_watermark(attacked, original, (32, 32), alpha=args.alpha)
    cv2.imwrite(args.out, (extracted * 255).astype(np.uint8))
    print("Saved extracted watermark to:", args.out)
