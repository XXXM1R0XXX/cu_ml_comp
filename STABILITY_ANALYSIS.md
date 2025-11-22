# Model Stability Analysis Documentation

## Overview

This document describes the model stability analysis features added to address the following concerns:
1. The test sample is very different from the training sample
2. Need to monitor model stability by month
3. Need to control overfitting
4. Need to evaluate the stability of feature separating power

## Features

### 1. Monthly Stability Analysis

The `analyze_monthly_stability()` function performs time-series cross-validation to assess model performance stability across different time periods.

**What it does:**
- Splits the training data into multiple temporal folds
- Trains models on earlier months and validates on later months
- Calculates train and validation AUC for each fold
- Monitors overfitting through train-validation gap analysis

**Key Metrics:**
- **Mean Validation AUC**: Average performance across time periods
- **Std Validation AUC**: Variability in performance (lower is more stable)
- **AUC Range**: Difference between best and worst performance
- **Train-Val Gap**: Indicator of overfitting (higher gap = more overfitting)

**Interpretation:**
- Low standard deviation in validation AUC indicates stable performance
- Large train-val gaps (>0.05-0.10) suggest overfitting
- Declining validation AUC over time may indicate distribution drift

### 2. Feature Importance Stability Analysis

The `analyze_feature_stability()` function tracks how feature importance changes across different time periods.

**What it does:**
- Trains models on different time windows
- Extracts feature importances for each period
- Calculates stability metrics for each feature
- Identifies consistently important vs. variable features

**Key Metrics:**
- **Mean Importance**: Average importance across time periods
- **CV (Coefficient of Variation)**: Std/Mean ratio (lower is more stable)
- **Range**: Difference between max and min importance

**Interpretation:**
- Features with low CV are stable and reliable
- Features with high CV but high mean importance may be time-dependent
- Features with high CV and low mean importance are unreliable

### 3. Generated Outputs

After running the analysis, the following files are generated:

#### CSV Files
- **monthly_stability_results.csv**: Detailed monthly performance metrics
  - Columns: month, train_auc, val_auc, train_val_gap, n_train, n_val
  
- **feature_stability_metrics.csv**: Feature stability metrics
  - Columns: feature, mean_importance, std_importance, cv_importance, min_importance, max_importance, range_importance

#### JSON File
- **stability_summary.json**: Summary statistics for quick review
  - Monthly stability summary (mean, std, min, max AUC)
  - Overfitting metrics
  - Top 10 most important features
  - Most and least stable features

#### Visualization Files
- **monthly_stability_analysis.png**: Four-panel plot showing:
  1. Train vs Val AUC over time
  2. Train-Val gap (overfitting indicator)
  3. Distribution of validation AUC
  4. Sample size by month

- **feature_stability_analysis.png**: Two-panel plot showing:
  1. Top 10 feature importances over time
  2. Scatter plot of importance vs. stability

## Usage

The analysis runs automatically as part of the marimo notebook execution. The stability analysis cells are executed after the model training is complete.

### Running the Analysis

```bash
# Run the marimo notebook
marimo edit main.py
```

The notebook will:
1. Load and prepare the data
2. Run hyperparameter optimization
3. Train the final model
4. **Execute monthly stability analysis**
5. **Execute feature stability analysis**
6. Generate predictions and submission file

### Interpreting Results

#### Good Model Stability
- Validation AUC std < 0.02
- Train-Val gap < 0.05
- Consistent feature importances (low CV for top features)

#### Warning Signs
- High variance in validation AUC (std > 0.05)
- Large train-val gaps (> 0.10)
- Top features have high CV (> 0.5)
- Declining performance over time

#### Recommended Actions Based on Results

**If overfitting is detected (large train-val gap):**
- Increase regularization (l2_leaf_reg)
- Reduce model complexity (depth)
- Use more aggressive early stopping
- Add more data augmentation

**If feature stability is low:**
- Consider feature engineering for more robust features
- Use feature selection to remove unstable features
- Investigate why features vary over time
- Consider interaction features or aggregations

**If temporal drift is detected:**
- Use more recent data for training
- Consider time-based features
- Implement model retraining schedule
- Use time-weighted samples

## Technical Details

### Time-Series Cross-Validation Approach

The analysis uses a progressive training approach:
- Fold 1: Train on months 1-N, validate on month N+1
- Fold 2: Train on months 1-N+1, validate on month N+2
- Fold 3: Train on months 1-N+2, validate on month N+3
- ...

This respects temporal ordering and simulates how the model would perform on future data.

### Performance Considerations

- GPU is required for efficient training (task_type: "GPU")
- Analysis time increases with number of time periods
- Expect ~5-10 minutes for complete analysis with 5 folds

## References

- CatBoost documentation: https://catboost.ai/
- Time Series Cross-Validation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
