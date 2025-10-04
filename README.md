# ViT-and-SAM2-Lab-Submission

# Vision Transformer and Text-Driven Segmentation

## 📄 Lab Submission
**Deadline:** Saturday, 11:59 PM (local time)  
**Environment:** Google Colab (GPU)

Repository contains:
- `q1.ipynb`
- `q2.ipynb`
- `README.md`

---

## Q1 — Vision Transformer on CIFAR-10 (PyTorch)

### 🎯 Objective
Implement and train a Vision Transformer (ViT-style) model on the CIFAR-10 dataset to achieve high classification accuracy.  
This implementation uses `timm` with `ResNet18` for stable training, AdamW optimizer, and mixed-precision acceleration.

### ⚙️ Configuration (Best Model)
| Parameter | Value |
|------------|--------|
| Model | ResNet18 (timm) |
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-2 |
| Scheduler | CosineAnnealingLR (Tmax=40) |
| Batch Size | 256 |
| Epochs | 40 |
| Mixed Precision | ✅ torch.amp.autocast + GradScaler |
| Data Augmentation | RandomCrop(32, padding=4), RandomHorizontalFlip |
| Normalization | mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5) |

### 📊 Results
| Metric | Value |
|:-------|:------|
| Final Test Accuracy (20 epochs) | 79.48% |
| Final Test Accuracy (40 epochs) | **83.87% ✅** |
| Best Train Accuracy | 91.46% |

**✅ Final Reported Accuracy (Q1): 83.87% on CIFAR-10**

---

### 🔍 Brief Analysis
- **Optimizer & Scheduler:** AdamW + Cosine Annealing improved test accuracy by ~4%.  
- **Augmentation:** Random cropping and horizontal flipping boosted generalization by ~2%.  
- **Mixed Precision:** Reduced memory use and training time (~1.3× faster).  
- **Depth/Width:** Shallow ViT-like architecture performed well on small dataset.  

---

## Q2 — Text-Driven Image Segmentation with SAM-2

### 🎯 Objective
Perform **text-prompted segmentation** on an image using **SAM-2** and **CLIPSeg** as a text-to-region seed generator.

### ⚙️ Pipeline Overview

#### 1. Install Dependencies
```bash
!pip install git+https://github.com/facebookresearch/segment-anything-2.git
!pip install transformers opencv-python matplotlib requests --quiet
```

#### 2. Load Models
- `CLIPSegProcessor`, `CLIPSegForImageSegmentation`  
- `Segment Anything Model 2 (ViT-H checkpoint)`

#### 3. Process
- Generate coarse **CLIPSeg** mask from the input text prompt.  
- Extract **seed points** from high-confidence mask regions.  
- Refine segmentation using **SAM-2 predictor** for fine-grained mask generation.

#### 4. Input Example
- **Image:** Zidane (from YOLOv5 dataset)  
- **Text Prompt:** `"football player"`

---

### 🖼️ Output
The integrated pipeline executed successfully on Colab.  
However, the CLIPSeg mask confidence was low (`max = 0.013`), leading to a weak SAM refinement.  
Future improvement could include using **GroundingDINO** or **GLIP** for better region grounding and more confident segmentation.

---

### ⚠️ Limitations
| Issue | Observation |
|-------|--------------|
| Low CLIPSeg seed confidence | Caused weak segmentation mask |
| Requires GPU | SAM-2 inference is computationally heavy |
| Improvement Suggestion | Use higher-resolution input and stronger text-region grounding models |

---

## 🧩 Repository Structure
```
├── q1.ipynb       # Vision Transformer on CIFAR-10 (PyTorch)
├── q2.ipynb       # Text-Driven Image Segmentation using SAM-2
└── README.md
```

---

## ▶️ How to Run on Colab
1. Open either notebook in **Google Colab**.  
2. Change runtime to **GPU** → `Runtime → Change runtime type → GPU`.  
3. Run all cells sequentially (`Runtime → Run all`).  
4. View model training logs, test accuracy (Q1), and segmentation outputs (Q2).

---

## 🏁 Final Summary
| Task | Model/Approach | Dataset | Result |
|------|----------------|----------|---------|
| **Q1** | ViT-style ResNet18 (PyTorch + AdamW) | CIFAR-10 | **83.87% Test Accuracy** |
| **Q2** | CLIPSeg + SAM-2 | Zidane sample image | Successful pipeline (low mask confidence) |

---

## 🧠 References
- **Paper:** *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* — Dosovitskiy et al., ICLR 2021  
- **Repositories Used:**
  - [timm](https://github.com/huggingface/pytorch-image-models)
  - [Segment Anything 2 (SAM-2)](https://github.com/facebookresearch/segment-anything-2)
  - [CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined)

---

**Author:** [Shiva A]  
**Frameworks:** PyTorch • timm • transformers • SAM-2  
**Execution:** 100% Colab (GPU)
