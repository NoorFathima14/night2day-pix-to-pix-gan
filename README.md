# 🌙→☀️ Night-to-Day Image Translation using Pix2Pix GAN

This project uses a **Pix2Pix GAN (Generative Adversarial Network)** to convert **night-time images** into **day-time images**. The model learns a mapping between paired night and day scenes using adversarial training.

---

## 📌 Table of Contents

1. [Project Overview](#project-overview)
2. [How to Run](#how-to-run)
3. [Dataset](#dataset)
4. [Training](#training)
5. [Inference & Testing](#inference--testing)
6. [Challenges & Lessons Learned](#challenges--lessons-learned)
7. [Future Work](#future-work)
8. [References](#references)

---

## 🌍 Project Overview

This project demonstrates how a **Pix2Pix GAN** can learn to transform **night scenes into realistic day images**.

### 🧠 Architecture

* **Generator**: U-Net architecture with skip connections
* **Discriminator**: PatchGAN classifier
* **Loss Functions**:

  * Adversarial Loss (GAN loss)
  * L1 Loss (pixel-wise similarity)

👉 The model learns:

> “Given a night image → generate its corresponding day version”

---

## ⚙️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/night2day-gan.git
cd night2day-gan
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### Key libraries:

* `tensorflow`
* `datasets` (HuggingFace)
* `pillow`
* `matplotlib`

---

## 📂 Dataset

This project uses the **Night2Day dataset** from HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("huggan/night2day", split="train")
```

### Dataset structure:

* `imageA` → Night image 🌙
* `imageB` → Corresponding day image ☀️

---

## 🏋️ Training

To train the model:

```bash
python train.py
```

### Training details:

* Input size: `256×256`
* Batch size: `1` (as per Pix2Pix paper)
* Optimizer: Adam (`lr=2e-4`, `beta_1=0.5`)
* Loss:

  * GAN loss
  * L1 loss

### During training:

* Model checkpoints are saved periodically
* Sample outputs are generated to track progress
* Generator & discriminator losses are logged

---

## 🔮 Inference & Testing

After training:

```bash
python predict.py
```

### Output:

* Night input image
* Ground truth day image
* Generated day image

👉 Helps visually evaluate model performance

---

## ⚠️ Challenges & Lessons Learned

* **Training Stability**
  GANs are unstable — balancing generator & discriminator was tricky

* **Resource Constraints**
  Training for many epochs required checkpointing and session recovery

* **Dataset Quality**
  Paired alignment between night and day images is crucial

* **Kaggle Limitations**
  Session timeouts required frequent checkpoint saving

---

## 🚀 Future Work

* Improve realism using:

  * **Perceptual Loss**
  * **Attention mechanisms**
* Try **CycleGAN** for unpaired datasets
* Optimize model for faster training
* Deploy as a web app for real-time transformation

---

## � References

1. **Image-to-Image Translation with Conditional Adversarial Networks**
2. TensorFlow Documentation
3. Hugging Face Datasets
4. Original Pix2Pix implementation

---

# 🧠 One-line summary

> This project trains a Pix2Pix GAN to convert night images into realistic day images using paired data and adversarial learning.

