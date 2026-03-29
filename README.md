# Efficiency Over Complexity: Optimal Neural Architectures for FOG Forecasting

[![Conference](https://img.shields.io/badge/DMD-2026-blue)](https://dmd.umn.edu/)
[![Topic](https://img.shields.io/badge/Research-Parkinson's%20Disease-orange)](#)
[![Method](https://img.shields.io/badge/Evaluation-LOSO--CV-green)](#)

This repository contains the official implementation and comparative analysis presented in our paper: **"Efficiency Over Complexity: Optimal Neural Architectures for Near-Term Forecasting of Parkinsonian Freezing of Gait"**.

## 📖 Abstract
Freezing of Gait (FOG) is a debilitating Parkinson's symptom linked to falls and loss of independence. While deep learning (DL) has improved detection, near-term **forecasting**—predicting an onset before it happens—remains a challenge. 

We benchmark **eight different architectures** (CNN, LSTM, Transformer, and Hybrids) using a unified preprocessing pipeline and a rigorous **Leave-One-Subject-Out Cross-Validation (LOSO-CV)** protocol. Our findings reveal that lightweight 1D-CNNs often outperform more complex, high-capacity models in generalizing to unseen patients.

---

## 🛠️ System Overview
We utilize high-fidelity gait dynamics from 16 individuals, captured via synchronized 3D acceleration and angular velocity sensors (200 Hz) mounted on the feet.

*(Consider adding an image of the sensor placement here: `![Sensor Placement](link_to_your_image.png)`)*

### The Forecasting Pipeline
* **Input:** 4-second sliding window of IMU data.
* **Horizon:** 2-second future prediction window.
* **Labeling:** "Any-event" strategy—if a freeze occurs within the 2s horizon, the input is labeled as **Impending FOG**.

*(Consider adding an image of the LOSO-CV process here: `![Pipeline Diagram](link_to_your_image.png)`)*

---

## 🏗️ Evaluated Architectures
We organized our models into three blocks to isolate the benefits of spatial vs. temporal modeling:

| Category | Architectures | Key Characteristics |
| :--- | :--- | :--- |
| **Simple/Baselines** | FD, LSTM (64), CNN (64) | Fundamental single-modality processing. |
| **Spatiotemporal Hybrids** | CNN-LSTM, LSTM-CNN | Synergy between feature extraction and temporal modeling. |
| **High-Capacity** | Deeper CNN, Transformer, LSTM-TCN | Investigating if increased complexity improves generalization. |

---

## 📊 Key Results
Our study highlights a significant **"Generalization Gap"** between standard random splits and the more realistic LOSO-CV protocol.

### LOSO-CV Performance (Subject Robustness)
The **CNN-64** and **Deeper CNN (128)** emerged as the most robust, achieving the best balance for clinical use.

| Architecture | Recall (Sensitivity) | F1-Score |
| :--- | :--- | :--- |
| **CNN (64)** | **0.9159** | **0.8415** |
| Deeper CNN (128) | 0.8920 | 0.8415 |
| LSTM-CNN | 0.8433 | 0.8183 |
| Transformer | 0.8176 | 0.7870 |
| FD (Baseline) | 0.0588 | 0.1110 |

> **The Takeaway:** For short-horizon forecasting (4s windows), 1D convolutions capture the essential local pre-freeze kinematics better than heavy sequential models, which are more prone to overfitting subject-specific noise.

---

## 🚀 Getting Started
### Prerequisites
* Python 3.10+
* PyTorch / TensorFlow *(update based on your framework)*
* NumPy, Pandas, Scikit-learn

### Usage
1. **Clone the repo:** ```bash
   git clone [https://github.com/ethan527w/FOG_Algorithm_Comparison_LOSOCV.git](https://github.com/ethan527w/FOG_Algorithm_Comparison_LOSOCV.git)
