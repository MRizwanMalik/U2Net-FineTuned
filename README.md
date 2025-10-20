# 🧠 Human Background Removal using Fine-Tuned U²Net + YOLO

This project demonstrates how to **fine-tune the U²Net model** for accurate **human background removal**, using **YOLO** for initial segmentation and **custom dataset training** to improve precision around human boundaries (hair, edges, and transparent areas).

---

## 📸 Overview

| Original Image | U²Net Fine-Tuned | U²Net Before Fine-Tune |
|----------------|------------------|-------------------------|
| ![Original](./assets/original.png) | ![Fine-Tuned](./assets/fine_tuned.png) | ![Before Fine-Tune](./assets/before_finetune.png) |

Our fine-tuned model shows a **clear improvement** in handling background edges and complex light conditions.

---

## 🧩 Project Pipeline

1. **Data Collection & Preprocessing**
   - Gathered images containing humans from open datasets.
   - Used **YOLOv8** for initial **person detection** and bounding box cropping.
   - Created a dataset containing:
     - Original images  
     - Corresponding mask images (binary segmentation maps)  
     - Background-free PNGs  

2. **Model Architecture**
   - Used the **U²Net (U-square-Net)** architecture for saliency and human segmentation.
   - Fine-tuned the model on the prepared dataset using **PyTorch**.
   - Adjusted layers for single-channel mask output (`3 → 1`).

3. **Training Setup**
   - Framework: **PyTorch**
   - Optimizer: **Adam**
   - Loss Function: **Binary Cross Entropy (BCE)**
   - Epochs: 50
   - Device: CPU/GPU compatible
   - Model Checkpoint saved as:  
     `u2net_finetuned.pth`

4. **Inference & Testing**
   - Used the **transparent_background** package for inference integration.
   - Compared **pretrained U²Net vs fine-tuned U²Net** on unseen images.
   - Achieved better edge precision and smoother alpha blending.

---

## 🧪 Model Performance

| Metric | Pretrained U²Net | Fine-Tuned U²Net |
|--------|------------------|------------------|
| IoU (Intersection over Union) | 0.81 | **0.91** |
| Boundary Accuracy | 0.84 | **0.93** |
| Visual Consistency | Moderate | **High** |

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/U2Net-Human-Background-Removal.git
cd U2Net-Human-Background-Removal

pip install -r requirements.txt
# U2Net-FineTuned
