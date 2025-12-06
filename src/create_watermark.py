import cv2

def create_logo(src_path, out_path, size=(32, 32)):
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, size)
    _, binary = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite(out_path, binary)
    print("Saved watermark:", out_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--out', default='data/watermark/logo.png')
    parser.add_argument('--size', type=int, default=32)
    args = parser.parse_args()

    create_logo(args.src, args.out, size=(args.size, args.size))
