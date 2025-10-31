# UPDATED README.md

```markdown
# Hidden Markov Model for Human Activity Recognition

## Project Overview
This project implements a Hidden Markov Model (HMM) to classify human activities using smartphone sensor data from accelerometer and gyroscope. The model successfully distinguishes between different motion patterns with 84.90% training accuracy and 65.79% test accuracy on unseen data.

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
│   ├── hmm_model_20251031_180138.pkl
│   ├── scaler_20251031_180138.pkl
│   ├── feature_names_20251031_180138.json
│   └── training_metrics_20251031_180138.json
├── reports/
│   └── figures/                   # Generated visualizations
├── evaluation_metrics.csv         # Performance metrics
└── collaboration_log.xlsx        # Task allocation
```

## Key Results
- **Training Accuracy**: 84.90% (all 4 activities)
- **Test Accuracy**: 65.79% (on available Jumping & Walking data)
- **Best Performing**: Walking (81.58% sensitivity in test)
- **Activities**: Jumping, Standing, Still, Walking
- **Model Saved**: Complete HMM with feature pipeline

## Performance Summary

### Training Performance
| Activity | Samples | Sensitivity | Specificity |
|----------|---------|-------------|-------------|
| Jumping  | 406     | 50.99%      | 99.92%      |
| Standing | 402     | 95.52%      | 97.25%      |
| Still    | 360     | 97.50%      | 100.00%     |
| Walking  | 508     | 94.69%      | 81.42%      |

### Test Performance (Unseen Data)
| Activity | Samples | Sensitivity | Specificity |
|----------|---------|-------------|-------------|
| Jumping  | 38      | 70.00%      | 100.00%     |
| Walking  | 38      | 81.58%      | 50.00%      |

## Model Architecture
- **Hidden Markov Model** with Gaussian emissions
- **4 States** (Jumping, Standing, Still, Walking)
- **23 Features** per analysis window
- **Baum-Welch algorithm** for training
- **Viterbi algorithm** for decoding

## Features Extracted
- **Time-domain**: Mean, STD, RMS, SMA, Correlations, Magnitude
- **Frequency-domain**: Dominant frequency, Spectral energy, Spectral entropy

## Technical Implementation
- **Window Size**: 100 samples (1 second at 100Hz)
- **Overlap**: 50% between windows
- **Normalization**: Z-score standardization
- **Training Samples**: 1,676 feature vectors
- **Test Samples**: 76 feature vectors

## Usage
```bash
# Run the complete pipeline
jupyter notebook hmm_activity_recognition_final.ipynb
```

## Requirements
```python
numpy, pandas, matplotlib, seaborn, scipy, scikit-learn
```

## Model Files Generated
- `hmm_model_20251031_180138.pkl` - Trained HMM parameters
- `scaler_20251031_180138.pkl` - Feature normalization scaler
- `feature_names_20251031_180138.json` - Feature names
- `training_metrics_20251031_180138.json` - Training statistics
- `evaluation_metrics.csv` - Performance metrics table

