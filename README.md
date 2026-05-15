# AMC-RadioML-ESP32
### Lightweight CNN-based Automatic Modulation Classification for Edge AI

> **6G Signal Intelligence | ESP32 + Raspberry Pi | PyTorch**

---

## Project Overview

This project implements an **Automatic Modulation Classification (AMC)** system using a lightweight CNN model trained on the RadioML 2016.10a dataset. The system is designed for edge deployment using ESP32 (feature extraction) and Raspberry Pi (inference).

---

## Target Modulations (8 types)

| Type | Category |
|------|----------|
| BPSK | Linear Digital |
| QPSK | Linear Digital |
| 8PSK | Linear Digital |
| 16QAM | Non-linear Digital |
| 64QAM | Non-linear Digital |
| PAM4 | Linear Digital |
| WBFM | Analog |
| AM-DSB | Analog |

---

## System Architecture

\\\
Laptop (Signal Source)
        |
       WiFi
        |
   ESP32 (Feature Extraction: DC offset removal, Normalization, SNR Estimation)
        |
      Cable
        |
  Raspberry Pi (CNN Inference: SNR Classifier + Specialist Models)
        |
     USB Port
        |
  Serial Monitor (Output)
\\\

---

## Model Versions

| Version | Description |
|---------|-------------|
| CNN v1 | Single lightweight CNN — baseline |
| CNN v2 | 4-model pipeline (Low/Mid/High SNR specialists + SNR classifier) |

### Target Accuracy

| SNR Range | Target Accuracy |
|-----------|----------------|
| Low SNR | 70-75% |
| Mid SNR | 75-85% |
| High SNR | 90-95% |
| Overall | 85-92% |

---

## Repository Structure

\\\
AMC-RadioML-ESP32/
├── data/           # Dataset storage (RadioML 2016.10a)
├── models/         # Saved model checkpoints
├── notebooks/      # Jupyter notebooks for EDA and experiments
├── src/            # Core Python source code
├── results/        # Evaluation results, plots, confusion matrices
├── esp32/          # ESP32 firmware (Arduino/C++)
├── rpi/            # Raspberry Pi inference scripts
└── docs/           # Project documentation
\\\

---

## Hardware

- **Training:** Gigabyte G5 (NVIDIA RTX 4050)
- **Feature Extraction:** ESP32
- **Inference:** Raspberry Pi
- **Dataset:** RadioML 2016.10a

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red)
![ESP32](https://img.shields.io/badge/ESP32-Arduino-green)

---

## Author

**YeariedJim2004** — Avionics/Aeronautical Engineering Student  
Research interests: Cognitive Radio, 6G, RF Signal Intelligence
