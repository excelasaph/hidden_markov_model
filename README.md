# Human Activity Recognition using Hidden Markov Models

## 📋 Project Overview
This project implements a Hidden Markov Model (HMM) for human activity recognition using smartphone sensor data. The system classifies four human activities: **Standing, Walking, Jumping, and Still** using accelerometer and gyroscope data with **74.07% accuracy**.

---
## 📁 File Structure

```
HMM_Activity_Recognition/
├──  Code/
│   ├── 1_Data_Collection_Preprocessing.ipynb
│   ├── 2_Feature_Engineering.ipynb
│   ├── 3_HMM_Implementation.ipynb
│   └── 4_Evaluation_Analysis.ipynb
├──  Data/
│   ├── excel/
│   │   ├── jumping/*.csv
│   │   ├── standing/*.csv
│   │   ├── walking/*.csv
│   │   └── still/*.csv
│   └── lesly/
│       ├── jumping/*.csv
│       ├── standing/*.csv
│       ├── walking/*.csv
│       └── still/*.csv
├──  Results/
│   ├── models/hmm_activity_model.pkl
│   ├── metrics/performance_summary.json
│   ├── metrics/cross_validation_results.csv
│   └── plots/
│       ├── confusion_matrix.png
│       ├── transition_matrix.png
│       └── feature_importance.png
└── Documentation/
    ├── README.md
    ├── requirements.txt
    └── CONTRIBUTION.md
```

---

## 🚀 Quick Start

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```txt
# requirements.txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
hmmlearn>=0.2.7
scipy>=1.7.0
jupyter>=1.0.0
```

### Run Pipeline
```python
# 1. Data Loading
from data_loader import UniversalDataLoader
loader = UniversalDataLoader()
df = loader.load_all_data()

# 2. Feature Extraction  
from feature_extractor import AdvancedFeatureExtractor
extractor = AdvancedFeatureExtractor()
features_df = extractor.create_enhanced_feature_dataset(df)

# 3. Model Training
from hmm_model import HMMActivityRecognizer
hmm = HMMActivityRecognizer()
model = hmm.train_hmm(train_sequences, train_lengths)

# 4. Evaluation
from evaluator import HMNEvaluator
evaluator = HMNEvaluator()
metrics, accuracy = evaluator.calculate_detailed_metrics(y_true, y_pred)
```

---

## 📊 Results Summary

### Performance Metrics
| State (Activity) | Samples | Sensitivity | Specificity | Overall Accuracy |
|------------------|---------|-------------|-------------|------------------|
| Standing | 54 | 0.852 | 0.933 | 0.852 |
| Walking | 54 | 0.778 | 0.944 | 0.778 |
| Jumping | 54 | 0.963 | 0.981 | 0.963 |
| Still | 54 | 0.741 | 0.963 | 0.741 |
| **Overall** | **216** | **0.833** | **0.955** | **0.833** |

### Cross-User Validation
| User | Accuracy | Samples |
|------|----------|---------|
| Excel | 73.15% | 216 |
| Lesly | 73.15% | 216 |
| **Mean** | **73.15%** | **432** |

### Model Training
- **Convergence**: ✅ Achieved
- **Iterations**: 16
- **Final Log-Likelihood**: 8589.15
- **States**: 4 (one per activity)

---

## 🔧 Implementation Details

### Data Collection
- **Sensors**: Accelerometer (x,y,z) + Gyroscope (x,y,z)
- **Sampling Rate**: 50 Hz
- **Duration**: 5-10 seconds per activity
- **Total Files**: 48 files (24 per user)
- **Format**: CSV with timestamp, sensor data, labels

### Feature Engineering
**Time-domain Features:**
- Statistical: Mean, STD, Variance, RMS, Range
- Shape-based: Skewness, Kurtosis, SMA, Energy
- Advanced: MAD, IQR, Median

**Frequency-domain Features:**
- Spectral energy, Dominant frequency
- Spectral centroid, Spectral entropy
- Band energy ratios (Low/Mid/High frequency)

**Cross-axis Features:**
- Signal magnitude
- Inter-axis correlations

### HMM Architecture
- **States**: 4 (Standing, Walking, Jumping, Still)
- **Emissions**: Gaussian with diagonal covariance
- **Training**: Baum-Welch algorithm (EM)
- **Decoding**: Viterbi algorithm
- **Initialization**: Structured parameters

### State Mapping
```python
# Optimal state-activity mapping
state_mapping = {
    0: 'walking',    # High detection accuracy
    1: 'jumping',    # Excellent sensitivity (0.963)
    2: 'still',      # Good specificity (0.963)  
    3: 'standing'    # Reliable performance
}
```

---

## 📈 Key Visualizations

### Generated Plots:
1. **Sensor Data Visualization** - Raw accelerometer/gyroscope signals
2. **Confusion Matrix** - Normalized and count versions
3. **Transition Probability Matrix** - Activity state transitions
4. **Feature Importance** - Top discriminative features
5. **Activity Transitions** - Temporal sequence predictions
6. **Performance Comparison** - Original vs Improved models

### Analysis Charts:
- Class distribution across activities
- Error analysis with common misclassifications
- Cross-user performance comparison
- Feature effectiveness by activity

--- 
