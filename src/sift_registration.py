from robust_watermark import register_images_sift
import cv2

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', required=True)
    parser.add_argument('--out', default='aligned.png')
    args = parser.parse_args()

    src = cv2.imread(args.src, cv2.IMREAD_GRAYSCALE)
    dst = cv2.imread(args.dst, cv2.IMREAD_GRAYSCALE)

    aligned, H = register_images_sift(src, dst)

    if H is None:
        print("Registration failed. Outputting unaligned image.")
        cv2.imwrite(args.out, src)
    else:
        cv2.imwrite(args.out, aligned)
        print("Aligned image saved to:", args.out)
