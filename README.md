# Hybrid Robust Image Watermarking (DWT–DCT + SIFT)
**Based on research paper:**  
*“A Hybrid Robust Image Watermarking Method Based on DWT-DCT and SIFT for Copyright Protection”*

---

## 📌 Project Goal
This project replicates the high-level technique from the paper to embed a secure and robust copyright watermark into images using:

- **Discrete Wavelet Transform (DWT – 1 level, Haar)**
- **Discrete Cosine Transform (DCT on 8×8 blocks)**
- **22 Mid-band frequency coefficients**
- **PN (Pseudo-random noise) Spread Spectrum embedding**
- **SIFT-based geometric registration for watermark recovery after attacks**
- **Watermark extraction via correlation (Blind watermarking)**
- **Quality metrics: PSNR, SSIM, Normalized Correlation (NC), Bit Error Rate (BER)**

---

## 🧠 Paper Method Replication Level
| Feature | Replicated? |
|---|:---:|
| 1-Level Haar DWT | ✅ |
| DCT on 8×8 blocks from HL1 band | ✅ |
| 22 mid-frequency coefficient selection | ✅ (mask included in code) |
| PN-sequence watermark bit embedding Y = X + λPN | ✅ |
| Correlation-based blind watermark extraction | ✅ |
| SIFT-based alignment for rotation/scale/crop | ✅ |
| Robustness attacks testing | ✅ (extendable in `experiments.py`) |

---

## 📁 Project Structure

DWT_DCT_SIFT_Watermarking/
├── data/
│ ├── cover/
│ │ ├── lena.png, baboon.png, peppers.png, sailboat.png, house.png...
│ └── watermark/
│ └── logo32.png (32×32 binary watermark)
├── src/
│ ├── utils.py
│ ├── sift_registration.py
│ ├── dwt_dct_sift_embed.py
│ ├── dwt_dct_sift_extract.py
│ ├── metrics.py
│ ├── attacks.py
│ ├── create_watermark.py
│ └── experiments.py → Run this file
├── results/
│ ├── watermarked/
│ ├── attacked/
│ └── extracted/
├── requirements.txt
└── README.md


---

## ⚙️ Setup & Installation

Make sure Python 3.10+ is installed.

Install dependencies:

```powershell
pip install opencv-python opencv-contrib-python pywavelets scikit-image matplotlib

▶️ Run Watermark Pipeline

Navigate inside src:

cd src
python experiments.py

🧪 Key Experiment Controls



Open experiments.py and modify:

LAMBDA = 2.0       # embedding strength
USE_ATTACK = True  # watermark robustness test
USE_SIFT = True   # only enable for geometric attacks

📊 Supported Attacks (Currently Implemented)

✅ Gaussian Noise

✅ Rotation

✅ JPEG Compression

You can extend more such as:

Blurring

Scaling

Cropping

Translation

Color jitter

📈 Metrics Expected
Metric	Meaning
PSNR	Watermark invisibility quality
SSIM	Visual similarity (1.0 = identical)
NC	Watermark robustness detection accuracy (should be high)
BER	Bit error, lower is better (0 = no bit flipped)
📌 Notes

Extracted watermark may look noisy if λ is too low.

Larger λ increases robustness but reduces PSNR slightly.

For no-attack case, NC should be ~1.0 and BER ~0.0.

For rotation/scale/crop tests, enable SIFT to realign before correlation.