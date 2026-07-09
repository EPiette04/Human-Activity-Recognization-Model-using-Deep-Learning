# Human Activity Recognition — Deep Learning Classifier

Group project (with [zachzs1](https://github.com/zachzs1)) for **CSDS 340
(Case Study 2)**. Classifies human activity from raw wearable-sensor time series
into 4 activity classes using a multitask CNN + LSTM + attention model. Full
write-up and results are in [`CSDS 340 Case Study 2.pdf`](./CSDS%20340%20Case%20Study%202.pdf).

## Approach

- **Input** — six-channel time-series windows (3-axis accelerometer + 3-axis
  gyroscope) per sample.
- **Feature engineering** — alongside the raw signal, hand-crafted features are
  extracted per window: signal magnitude, high-frequency FFT coefficients,
  spectral entropy, and low/mid-frequency power ratios.
- **Class imbalance** — the minority activity class is oversampled with SMOTE
  before training.
- **Model** (`Classify_activity.py`) — a multitask Keras network with two input
  branches:
  - raw signal → Conv1D layers → LSTM → self-attention → pooled embedding
  - engineered features → dense layers
  
  The branches are concatenated and fed to a main 4-class softmax head, plus two
  auxiliary binary heads that help the model separate the two hardest classes.
- **Post-processing** — a confidence threshold is applied to the hardest class,
  and predictions are temporally smoothed with a rolling-mode filter.
- **Evaluation** (`Evaluate_classifier.py`) — reports micro/macro-averaged F1,
  per-class F1, a confusion matrix, and a classification report.

## Tech stack

Python · TensorFlow / Keras · scikit-learn · imbalanced-learn (SMOTE) · NumPy · SciPy · Matplotlib

## Repository structure

```
Classify_activity.py                 # Feature extraction + multitask model + predict_test()
Evaluate_classifier.py               # Evaluation harness: trains + scores + plots
Evaluate_classifier_test.py          # Variant evaluation harness
customEval.py                        # Additional evaluation utilities
Data/                                 # Accelerometer/gyroscope + label CSVs (two splits)
EDA and previous iterations/         # Exploratory scripts and earlier modeling attempts
EDA on data 1/, EDA on data 2/       # Saved EDA plots
CSDS 340 Case Study 2.pdf            # Assignment write-up / report
```

## Running it

```bash
pip install numpy tensorflow scikit-learn imbalanced-learn scipy matplotlib
python "Evaluate_classifier.py"
```

This loads the sensor CSVs from `Data/`, trains the model via
`Classify_activity.predict_test`, and prints/plots the F1 scores, confusion
matrix, and per-class breakdown.
