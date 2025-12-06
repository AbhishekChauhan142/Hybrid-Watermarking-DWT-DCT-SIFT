# src/majority_vote_combine.py
import numpy as np
import cv2
import os
from robust_watermark import compute_ber

# Paths (adjust if you use different names)
lh_path = "results/extracted/lena_extracted_LH.png"
hl_path = "results/extracted/lena_extracted_HL.png"
wm_path = "data/watermark/logo64.png"
out_path = "results/extracted/lena_extracted_majority.png"

def load_bin_image(path, size=(64,64)):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    # Resize if necessary
    if im.shape[:2] != size:
        im = cv2.resize(im, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    # Convert to binary 0/1
    return (im > 128).astype(np.uint8)

def compute_nc(original_bits: np.ndarray, extracted_bits: np.ndarray) -> float:
    a = original_bits.flatten().astype(np.int8)
    b = extracted_bits.flatten().astype(np.int8)
    a = 2 * a - 1
    b = 2 * b - 1
    return float(np.sum(a * b)) / float(a.size)

def main():
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    a = load_bin_image(lh_path)
    b = load_bin_image(hl_path)
    wm = load_bin_image(wm_path, size=a.shape)

    # Majority vote: since only two sources, majority is equivalent to taking bitwise OR when they disagree.
    # We'll do: if a==b -> keep; else -> choose bit from the one that is 1 (OR). If you prefer tie=0, replace logic.
    voted = np.where(a == b, a, (a | b)).astype(np.uint8)

    # Save combined image (0/1 -> 0/255)
    cv2.imwrite(out_path, (voted * 255).astype("uint8"))

    # Compute metrics
    ber = compute_ber(wm, voted)
    nc = compute_nc(wm, voted)

    print("Saved majority-vote extraction to:", out_path)
    print(f"BER: {ber:.4f}")
    print(f"NC : {nc:.4f}")

if __name__ == "__main__":
    main()
