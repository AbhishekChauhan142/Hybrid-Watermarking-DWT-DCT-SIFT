# quick runner to embed a 64x64 watermark using block_size=4
import os
import cv2
from robust_watermark import embed_watermark

cover_path = "data/cover/lena.png"
wm_path = "data/watermark/logo64.png"
out_path = "results/watermarked/lena_wm_64_bs4.png"

alpha = 12
size = 64
block_size = 4
subband = "HH"

os.makedirs(os.path.dirname(out_path), exist_ok=True)

cover = cv2.imread(cover_path)
wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
wm = cv2.resize(wm, (size, size))
wm_bin = (wm > 128).astype("uint8")

out = embed_watermark(cover, wm_bin, alpha=alpha, subband=subband, block_size=block_size)
cv2.imwrite(out_path, out)
print("Saved:", out_path)
