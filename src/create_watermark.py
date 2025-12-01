import cv2
import numpy as np
import os

out_dir = "../data/watermark"
os.makedirs(out_dir, exist_ok=True)

# 32x32 blank
img = np.ones((32, 32), dtype=np.uint8) * 255

cv2.putText(img, "WM", (1, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,), 1, cv2.LINE_AA)

out_path = os.path.join(out_dir, "logo32.png")
cv2.imwrite(out_path, img)

print("✅ Saved 32x32 watermark.")
