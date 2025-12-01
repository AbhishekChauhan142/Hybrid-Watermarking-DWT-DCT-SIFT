import cv2
import numpy as np

def register_with_sift(attacked_img, ref_kp_array, ref_des):
    sift = cv2.SIFT_create()
    kps2, des2 = sift.detectAndCompute(attacked_img, None)

    if des2 is None or ref_des is None or ref_kp_array.size == 0:
        return attacked_img

    # ✅ Correct way to rebuild KeyPoint objects
    kps1 = []
    for kp in ref_kp_array:
        x, y, size, angle = kp
        kps1.append(cv2.KeyPoint(float(x), float(y), float(size), float(angle)))

    # SIFT matching using BFMatcher + ratio test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(ref_des, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 10:
        return attacked_img

    src_pts = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    h, w = attacked_img.shape
    if H is None:
        return attacked_img

    aligned = cv2.warpPerspective(attacked_img, H, (w, h))
    return aligned
