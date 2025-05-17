# Neurodegenerative Disorder Detection via Voice & Typing Analysis

![GitHub](https://img.shields.io/github/license/yourusername/repo-name?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![ML Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20PyTorch-orange?style=flat-square)

A machine learning project to detect early signs of neurodegenerative disorders (e.g., Alzheimer’s, Parkinson’s) using **voice analysis** and **typing pattern analysis**. This repository contains code for data preprocessing, feature extraction, model training, and evaluation.

---

## 📌 Overview
Neurodegenerative disorders often manifest subtle changes in speech and motor coordination long before clinical diagnosis. This project leverages:
- **Voice Analysis**: Pitch, tone, speech pauses, and vocal tremors.
- **Typing Analysis**: Keystroke dynamics, typing speed, and error patterns.

The goal is to build a non-invasive, low-cost screening tool for early detection.

---

## 🚀 Features
- **Voice Module**:
  - Preprocessing of audio recordings (noise removal, segmentation).
  - Extraction of MFCCs, prosodic features, and spectral characteristics.
  - CNN/LSTM models for voice-based classification.
  
- **Typing Module**:
  - Capture typing patterns (key hold time, latency between keystrokes).
  - Feature engineering for motor coordination metrics.
  - Random Forest/GRU models for keystroke dynamics classification.

- **Fusion Model**:
  - Late-fusion architecture combining voice and typing modalities for improved accuracy.

---

## 📦 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/repo-name.git
   cd repo-name
