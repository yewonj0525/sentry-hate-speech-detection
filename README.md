# SENTRY: Set Embeddings for Robust Hate Speech Detection

> A set-based NLP framework that detects hate speech by analyzing groups of semantically related tweets — rather than isolated posts — using permutation-equivariant architectures and hierarchical contrastive optimization.

📄 [Full Paper (PDF)](./SENTRY_FinalReport.pdf) | 👥 Team Project (UNC Chapel Hill, COMP 562)

---

## 🔍 Project Overview

Traditional hate speech detection classifies each post in isolation. **SENTRY** takes a different approach: it groups semantically related tweets into *sets* and learns context-aware representations across them.

Key ideas:
- **Set-based modeling**: tweets are grouped via unsupervised K-means clustering
- **Permutation-equivariant architecture**: set representations are order-independent
- **Hierarchical DPO loss**: contrastive objectives applied at both tweet and set levels
- Evaluated on two datasets: **Davidson** (tweet classification) and **DIALOCONAN** (dialogue classification)

**Best result**: Macro-F1 **0.847** on Davidson dataset (Equivariant + Attention Pooling + DPO)

---

## 👩‍💻 My Contributions

This was a 4-person team project. I was responsible for the following:

### 1. Data Preprocessing & Augmentation
- Cleaned raw Davidson dataset (24,783 tweets): removed nulls, duplicates, and whitespace; encoded labels
- Addressed severe **class imbalance** (hate speech = only 5.77% of data) using **back-translation augmentation**
  - Translated minority-class tweets to French/German/Spanish and back to English using MarianMT
  - Applied similarity filtering to remove low-quality outputs
- Constructed tweet sets of size K=5, yielding 4,957 sets total
- Performed 70/15/15 train/val/test split

### 2. Davidson Experiment Design & Execution (Exp 1–6)
Designed and ran 6 progressive experiments to isolate the contribution of each component:

| Experiment | Configuration | Purpose |
|---|---|---|
| Exp 1 | Sets + Cross Entropy + Mean Pooling | Baseline |
| Exp 2 | Sets + DPO | Does contrastive loss help? |
| Exp 3 | Sets + Attention Pooling + DPO | Does attention pooling help? |
| Exp 4 | Equivariant + Mean Pooling | Does equivariant architecture help? |
| Exp 5 | Equivariant + Attention Pooling + DPO | Full model (best) |
| Exp 6 | Equivariant + Cross Entropy only | Ablation: DPO vs. no DPO |

**Default configuration I used:**
- Encoder: DistilBERT (66M params)
- Set size: K = 5
- Pooling: Mean (Exp 1–2) → Attention (Exp 3, 5)
- Loss weights: λ_CE = 1.0, λ_tweet = 0.1, λ_set = 0.5, β = 0.1
- Optimizer: AdamW | Epochs: 10 | Batch size: 8

**Key results (Exp 5, best model):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Hate | 79.2% | 44.2% | 56.7% |
| Offensive | 96.8% | 100% | 98.4% |
| Neutral | 96.0% | 96.0% | 96.0% |
| **Macro** | | | **83.7%** |

---

## 🛠️ Tech Stack

- **Language**: Python
- **Models**: DistilBERT (HuggingFace Transformers)
- **Frameworks**: PyTorch
- **Tools**: scikit-learn (K-means clustering), MarianMT (back-translation), NumPy, Git
- **Hardware**: NVIDIA A100 GPU (40GB)

---

## 📁 Repository Structure

```
├── config/          # Experiment configuration files
├── data/            # Preprocessed datasets
├── experiments/     # Experiment results and logs
├── losses/          # DPO loss implementations
├── models/          # Model architectures
├── training/        # Training pipeline
├── utils/           # Helper functions
├── main.py          # Entry point
├── requirements.txt # Dependencies
└── run_experiments.sh  # Script to reproduce experiments
```

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/yewonj0525/sentry-hate-speech-detection.git
cd sentry-hate-speech-detection

# Install dependencies
pip install -r requirements.txt

# Run default experiment (Exp 5: Equivariant + Attention + DPO)
python main.py
```

---

## 👥 Team

| Name | Contribution |
|---|---|
| Yewon Joung | Data preprocessing & augmentation, Davidson experiment design & execution |
| Amogh Gupta | Model architecture (equivariant layers) |
| Yuvraj Jain | DPO loss implementation, DIALOCONAN experiments |
| Paritosh Pandey | Set construction, evaluation pipeline |

*UNC Chapel Hill — COMP 562: Machine Learning (Fall 2024)*
