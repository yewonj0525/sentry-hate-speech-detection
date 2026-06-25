# SENTRY: Set Embeddings for Robust Hate Speech Detection

> A set-based NLP framework that detects hate speech by analyzing groups of semantically related tweets — rather than isolated posts — using permutation-equivariant architectures and hierarchical contrastive optimization.

---

## 📌 Project Overview

Traditional hate speech detection classifies each post in isolation, missing contextual patterns across related messages. **SENTRY** addresses this by grouping semantically related tweets into *sets* and learning context-aware representations across them.

**Key ideas:**
- Tweets are grouped into sets via unsupervised K-means clustering
- Permutation-equivariant architecture ensures order-independent set representations
- Hierarchical DPO loss applies contrastive objectives at both tweet and set levels
- Evaluated on two datasets: **Davidson** (tweet classification) and **DIALOCONAN** (dialogue classification)

**Best result:** Macro-F1 **0.847** on Davidson dataset (Equivariant + Attention Pooling + DPO)

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
| Exp 2 | Sets + DPO + Mean Pooling | Does contrastive loss help? |
| Exp 3 | Sets + Attention Pooling + DPO | Does attention pooling help? |
| Exp 4 | Equivariant + Mean Pooling | Does equivariant architecture help? |
| Exp 5 | Equivariant + Attention Pooling + DPO | Full model (best) |
| Exp 6 | Equivariant + Cross Entropy only | Ablation: DPO vs. no DPO |

---

## 📊 Key Results

**Davidson Dataset (Exp 5 — Best Model):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Hate | 79.2% | 44.2% | 56.7% |
| Offensive | 96.8% | 100% | 98.4% |
| Neutral | 96.0% | 96.0% | 96.0% |
| **Macro** | | | **83.7%** |

**Key finding:** Set-based aggregation with equivariant architecture and hierarchical contrastive optimization substantially outperforms single-instance baselines across all classes.

---

## 📁 Project Structure

```
├── data/
│   └── preprocess.py        # Data cleaning, set construction, train/val/test split
├── experiments/
│   └── run_davidson.py      # Experiment 1–6 design and execution
├── configs/
│   └── davidson_config.yaml # Hyperparameter settings
├── augmentation.py          # Back-translation data augmentation
├── requirements.txt
└── README.md
```

> **Note:** Raw data not included. Download the Davidson dataset from [Kaggle](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset).

---

## ⚙️ Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
```bash
# Place labeled_data.csv in data/
python data/preprocess.py
```

### 3. Run augmentation
```bash
python augmentation.py
```

### 4. Run experiments
```bash
# Run all 6 experiments
python experiments/run_davidson.py
```

---

## 🛠️ Tech Stack

- **Language:** Python
- **Models:** DistilBERT (HuggingFace Transformers)
- **Frameworks:** PyTorch
- **Tools:** scikit-learn (K-means), MarianMT (back-translation), NumPy, Git
- **Hardware:** NVIDIA A100 GPU (40GB)

---

## 👥 Team

| Name | Role |
|---|---|
| Yewon Joung | Data preprocessing & augmentation, Davidson experiment design & execution |
| Amogh Gupta | — |
| Yuvraj Jain | — |
| Paritosh Pandey | — |

*UNC Chapel Hill — COMP 755: Machine Learning (Fall 2025)*
