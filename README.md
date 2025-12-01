# Hybrid Robust Image Watermarking (DWT–DCT + SIFT)

**Research Paper Replicated:**  
*A Hybrid Robust Image Watermarking Method Based on DWT-DCT and SIFT for Copyright Protection*  
(Algorithmic pipeline replicated academically for copyright protection and robustness testing)

---

## 📌 Overview

This project embeds and extracts a copyright watermark into grayscale images using a hybrid transform + feature-based alignment technique:

- **1-Level Haar DWT** (decomposes image into LL1, HL1, LH1, HH1)
- **Block-wise 2D-DCT (8×8)** on **HL1 band**
- **22 mid-frequency DCT coefficients** modified using PN-generated spread-spectrum sequences
- **IDCT → IDWT reconstruction**
- **Blind watermark extraction via correlation**
- **SIFT keypoint-based geometric alignment** for attacked images (rotation/scale/crop etc.)
- **Robustness testing** using common watermark attacks

---

## 🧪 Evaluation Metrics

| Metric | Meaning |
|---|---|
| **PSNR** | Measures visual distortion (higher = watermark more invisible) |
| **SSIM** | Structural similarity with original (1.0 = identical) |
| **NC** | Normalized correlation for watermark match (higher = more robust detection) |
| **BER** | Bit error rate of extracted watermark (lower = fewer bit flips) |

---

## 📁 Project Structure

DWT_DCT_SIFT_Watermarking/
├── data/
│ ├── cover/ # Cover images (.png only, grayscale 512×512)
│ │ ├── lena.png
│ │ ├── baboon.png
│ │ ├── peppers.png
│ │ └── house.png …
│ └── watermark/ # Watermark images
│ └── logo32.png # (32×32 binary watermark = 1024 bits)
│
├── src/ # Source code modules
│ ├── utils.py # Image/DWT/DCT utilities + PN generators
│ ├── sift_registration.py # SIFT-based geometric alignment
│ ├── dwt_dct_sift_embed.py # Watermark embedding function
│ ├── dwt_dct_sift_extract.py # Watermark extraction function
│ ├── metrics.py # PSNR, SSIM, NC, BER implementations
│ ├── attacks.py # Noise, rotation, JPEG compression & more attacks
│ └── experiments.py # 🚀 Execute watermark pipeline from here
│
├── results/
│ ├── watermarked/ # Watermarked output images
│ ├── attacked/ # Attacked images for experiment tests
│ └── extracted/ # Extracted watermark images
│
├── requirements.txt
└── README.md

yaml
Copy code

---

## ⚙️ Setup

Ensure Python 3.10+ is installed.

Install dependencies:

```powershell
pip install opencv-python opencv-contrib-python pywavelets scikit-image matplotlib
▶️ Run Watermark Pipeline
Run from inside src/ directory:

powershell
Copy code
cd src
python experiments.py
🛡️ Supported Attacks (in src/attacks.py)
✅ Gaussian noise

✅ Rotation

✅ JPEG compression

(Extendable: blur, crop, scaling, color jitter, translation, etc.)

📈 Expected Behavior
Watermarked images are nearly identical to original when λ is small-to-moderate.

Extracted watermark may show speckles proportional to BER (bit flips).

For no-attack tests, aim for high NC (~1.0) and very low BER (~0.0).

For geometric attacks, enable SIFT alignment before correlation extraction.