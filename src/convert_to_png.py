import cv2
import os

cover_dir = "../data/cover"  # Path to your cover folder

for file in os.listdir(cover_dir):
    if file.lower().endswith(".tif") or file.lower().endswith(".tiff"):
        img_path = os.path.join(cover_dir, file)
        img = cv2.imread(img_path)

        if img is None:
            print("Failed to load:", file)
            continue

        png_name = os.path.splitext(file)[0] + ".png"
        png_path = os.path.join(cover_dir, png_name)
        cv2.imwrite(png_path, img)
        print("Converted:", file, "→", png_name)

print("✅ All possible images are converted to PNG!")
