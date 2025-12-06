# quick runner to extract a 64x64 watermark embedded with block_size=4
import os
import cv2
from robust_watermark import extract_watermark

attacked_path = "results/attacked/lena_attacked_64_bs4.jpg"
orig_path = "data/cover/lena.png"
out_path = "results/extracted/lena_extracted_64_bs4.png"

size = 64
block_size = 4
subband = "HH"
alpha = 12

os.makedirs(os.path.dirname(out_path), exist_ok=True)
att = cv2.imread(attacked_path)
orig = cv2.imread(orig_path)
extracted = extract_watermark(att, orig, (size, size), alpha=alpha, subband=subband, block_size=block_size)
cv2.imwrite(out_path, (extracted * 255).astype("uint8"))
print("Saved extracted:", out_path)
