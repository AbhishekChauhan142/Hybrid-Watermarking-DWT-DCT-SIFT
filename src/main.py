import argparse
import os
import cv2
import numpy as np
from robust_watermark import embed_watermark, extract_watermark

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="cmd")

embed_cmd = sub.add_parser("embed")
embed_cmd.add_argument("--cover", required=True)
embed_cmd.add_argument("--wm", required=True)
embed_cmd.add_argument("--out", required=True)
embed_cmd.add_argument("--alpha", type=float, default=12)
embed_cmd.add_argument("--size", type=int, default=64, help="watermark size (square)")
embed_cmd.add_argument("--block_size", type=int, default=4, help="DCT block size inside DWT subband")
embed_cmd.add_argument("--subband", type=str, default="HH", choices=["HH", "HL", "LH"])

extract_cmd = sub.add_parser("extract")
extract_cmd.add_argument("--attacked", required=True)
extract_cmd.add_argument("--orig", required=True)
extract_cmd.add_argument("--out", required=True)
extract_cmd.add_argument("--alpha", type=float, default=12)
extract_cmd.add_argument("--size", type=int, default=64, help="watermark size (square)")
extract_cmd.add_argument("--block_size", type=int, default=4, help="DCT block size inside DWT subband")
extract_cmd.add_argument("--subband", type=str, default="HH", choices=["HH", "HL", "LH"])
extract_cmd.add_argument("--no_register", action="store_true", help="disable SIFT registration during extraction")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        exit(0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.cmd == "embed":
        cover = cv2.imread(args.cover)
        if cover is None:
            raise FileNotFoundError(args.cover)
        wm = cv2.imread(args.wm, cv2.IMREAD_GRAYSCALE)
        if wm is None:
            raise FileNotFoundError(args.wm)
        wm = cv2.resize(wm, (args.size, args.size))
        wm_bin = (wm > 128).astype(np.uint8)
        out = embed_watermark(cover, wm_bin, alpha=args.alpha, subband=args.subband, block_size=args.block_size)
        cv2.imwrite(args.out, out)
        print("Watermarked image saved:", args.out)

    elif args.cmd == "extract":
        attacked = cv2.imread(args.attacked)
        if attacked is None:
            raise FileNotFoundError(args.attacked)
        orig = cv2.imread(args.orig)
        if orig is None:
            raise FileNotFoundError(args.orig)
        extracted = extract_watermark(attacked, orig, (args.size, args.size),
                                      alpha=args.alpha, subband=args.subband,
                                      block_size=args.block_size,
                                      use_registration=not args.no_register)
        extracted_img = (extracted * 255).astype(np.uint8)
        cv2.imwrite(args.out, extracted_img)
        print("Extracted watermark saved:", args.out)
