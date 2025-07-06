# SMART-SBERT with SimCSE: A Robust Defense Against Catastrophic Forgetting in BERT

This repository contains the code for our Final Year Project at UET Lahore:  
**"SMART-SBERT with SimCSE: A Robust Defense Against Catastrophic Forgetting in BERT"**.  
We propose a novel fine-tuning strategy that combines **Siamese SBERT**, **SimCSE**, and **SMART regularization** to improve robustness and multi-task performance in NLP.

---

## 📌 Project Overview

Large pre-trained models like BERT struggle with **catastrophic forgetting** when fine-tuned sequentially on multiple tasks. Our goal was to build a **robust and compact multi-task model** that maintains high performance across:

- **Paraphrase Detection (QQP)**  
- **Semantic Textual Similarity (STS-B)**  
- **Sentiment Classification (SST-5)**

We propose a robust fine-tuning framework that integrates the following components:
- **Smoothness-Inducing Adversarial Regularization:** Enhances model robustness by training it to resist small perturbations in the input space.
- **Bregman Proximal Point Optimization:** Prevents large and unstable parameter updates during fine-tuning, ensuring smoother convergence and reducing overfitting.
- **Contrastive Learning:** Enhances sentence representations by pulling embeddings of semantically related sentence pairs closer together, while pushing apart those of unrelated pairs within the embedding space.
- We also adopt the **Siamese Sentence-BERT (SBERT) Architecture** to create meaningful sentence embeddings that capture semantic similarity, enabling efficient comparison of sentence pairs.
---

## 👥 Authors

- Rooshan Khan — 2021-EE-067  
- Hussnain Amjad — 2021-EE-063  
- Abdul Samad — 2021-EE-191  
- Areesha Noor — 2021-EE-103  
- **Supervisor**: Dr. Irfan Ullah Chaudhary
Department of Electrical Engineering, UET Lahore

---
## 🙏 Acknowledgments
- We would like to express our heartfelt gratitude to our supervisor, Dr. Irfan Ullah Chaudhry, for his invaluable guidance, continuous support, and encouragement throughout the course of this project.
- We are also deeply thankful to Dr. Haroon Babri for his insightful suggestions and constructive feedback, which significantly enhanced our understanding and progress.
- Lastly, we extend our sincere appreciation to Mr. Asad Ullah Khan for his consistent assistance and helpful input during the course of this work.
---
## 🔬 Phase 1: Single-Task Fine-Tuning for Sentiment Analysis

Phase 1 of our project focused on fine-tuning BERT for the sentiment analysis task using the **SST-5** and **CFIMDB** datasets. This phase aimed to understand the effectiveness of different fine-tuning strategies on a single task.

We experimented with:
1. **Last linear layer fine-tuning**
2. **Full model fine-tuning**

### 📊 Development Accuracies

| Fine-Tuning Strategy       | Ours (SST) | Ours (CFIMDB) | Reference (SST) | Reference (CFIMDB) |
|---------------------------|------------|----------------|------------------|----------------------|
| Last linear layer         | 0.409      | 0.788          | 0.390            | 0.780                |
| Full model fine-tuning    | 0.524      | 0.967          | 0.515            | 0.966                |

These results confirmed that full model fine-tuning provides a significant boost in accuracy for sentiment analysis tasks.

---
## 🧠 Model Architecture

We use a shared BERT-base encoder and three task-specific heads:
- **Regression Head** for STS-B
- **Classification Head** for QQP
- **Classification Head** for SST-5

Training is done **sequentially across tasks**, and performance is evaluated using a composite metric.

---

## 📊 Evaluation Metrics

| Task                      | Metric                |
|---------------------------|------------------------|
| Paraphrase Detection (QQP) | Accuracy, F1 Score     |
| Semantic Similarity (STS-B)| Pearson Correlation    |
| Sentiment Classification (SST-5) | Accuracy, F1 Score |

**Overall performance** is computed as:  
`((STS_Corr + 1)/2 + QQP_Acc + SST_Acc) / 3`

---

## 📈 Results

### SMART-SBERT (SS)
- SST-5 Accuracy: **0.537**
- QQP Accuracy: **0.864**
- STS-B Pearson Corr: **0.819**
- **Overall Score**: **0.770**

### SMART-SBERT + SimCSE (SSS)
- SST-5 Accuracy: **0.507**
- QQP Accuracy: **0.864**
- STS-B Pearson Corr: **0.843**
- **Overall Score**: **0.764**

Choose SS or SSS based on whether SST or STS performance is more critical.

---

## 🗃️ Dataset Summary

| Dataset      | Train   | Dev   | Test  |
|--------------|---------|-------|-------|
| QQP          | 283k    | 40k   | 81k   |
| STS-B        | 6k      | 863   | 1.7k  |
| SST-5        | 8.5k    | 1.1k  | 2.2k  |
| SNLI (SimCSE)| 550k    | -     | -     |

---

## 📦 Download Fine-Tuned Models

The following models were fine-tuned as part of this Final Year Project using our proposed SMART-SBERT and SMART-SBERT + SimCSE framework. Due to GitHub LFS limitations, they are hosted externally:

🔗 [Download from Google Drive](https://drive.google.com/drive/folders/17Ia3Q82RDLN1AelTeSwjSZLaunfxv2AB?usp=drive_link)

### Included:
- `SMART_SBERT.pt`: Our fine-tuned model using SMART regularization
- `SMART_SBERT_SIMCSE.pt`: Our fine-tuned model using SMART + SimCSE contrastive learning

## 🛠️ Setup Instructions

1. **Clone Repo**
```bash
git clone https://github.com/RooshanKhan/FYP_Group27_Session2021.git
cd FYP_Group27_Session2021
