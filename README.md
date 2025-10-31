# Hidden Markov Model for Human Activity Recognition

## Project Overview
This project implements a Hidden Markov Model (HMM) to classify human activities (Jumping, Standing, Still, Walking) using smartphone sensor data from accelerometer and gyroscope.

## Group Members
- **Excel** - Data Collection, Feature Engineering, HMM Implementation
- **Lesly** - Data Collection, Model Evaluation, Report Writing

## Project Structure
```
HMM-Activity-Recognition/
├── data/                          # Raw sensor data (258 files)
│   ├── jumping/                   # 10 sessions
│   ├── standing/                  # 10 sessions  
│   ├── still/                     # 10 sessions
│   ├── walking/                   # 12 sessions
│   └── test/                      # Unseen test data
├── notebook/
│   └── hmm_activity_recognition_final.ipynb
├── models/                        # Saved models and scalers
├── reports/
│   └── figures/                   # Generated visualizations
├── evaluation_metrics.csv         # Performance metrics
```

## Key Results
- **Overall Test Accuracy**: 92.11%
- **Activities**: Jumping, Standing, Still, Walking
- **Best Performing**: Walking (94.74% sensitivity)
- **Model Generalization**: Excellent (92.11% on unseen data)

## Features Extracted
- **Time-domain**: Mean, STD, RMS, SMA, Correlations
- **Frequency-domain**: Dominant frequency, Spectral energy, Spectral entropy

## Requirements
```bash
numpy, pandas, matplotlib, seaborn, scipy, scikit-learn
```

## Usage
Run the Jupyter notebook `hmm_activity_recognition_final.ipynb` to:
1. Load and preprocess sensor data
2. Extract time and frequency domain features
3. Train Gaussian HMM with Baum-Welch algorithm
4. Evaluate model performance
5. Generate visualizations and metrics
---

### **Training Metrics Table Overview**

| **State (Activity)** | **Number of Samples** | **Sensitivity** | **Specificity** | **Overall Accuracy** |
| -------------------: | --------------------: | --------------: | --------------: | -------------------: |
|              jumping |                   406 |          1.0000 |          1.0000 |               1.0000 |
|             standing |                   402 |          1.0000 |          1.0000 |               1.0000 |
|                still |                   360 |          1.0000 |          1.0000 |               1.0000 |
|              walking |                   508 |          1.0000 |          1.0000 |               1.0000 |
---
## Test Performance

**Overall Test Accuracy**: 92.11%

| Activity | Samples | Sensitivity | Specificity |
|----------|---------|-------------|-------------|
| Jumping  | 38      | 0.8947      | 0.9474      |
| Walking  | 38      | 0.9474      | 0.8947      |

