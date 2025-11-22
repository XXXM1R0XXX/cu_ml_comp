# CU ML Competition - Scoring Model

Binary classification model using CatBoost for predicting `a6_flg` with temporal stability analysis.

## Features

- **Hyperparameter Optimization**: Uses Optuna for automated hyperparameter tuning
- **Temporal Validation**: Time-series aware data splitting
- **Model Stability Analysis**: Comprehensive monthly performance tracking
- **Feature Stability Analysis**: Monitors feature importance consistency over time
- **Overfitting Control**: Train-validation gap monitoring and early stopping

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for CatBoost training)

### Installation

```bash
pip install -e .
```

### Usage

Run the marimo notebook:

```bash
marimo edit main.py
```

The notebook will:
1. Load training and test data from `data/` directory
2. Perform hyperparameter optimization (50 trials or 10 minutes)
3. Train the final model
4. **Run stability analysis** (new feature)
5. Generate submission file

## Model Stability Analysis

The notebook now includes comprehensive stability analysis to address concerns about:
- Test samples being very different from training samples
- Model performance consistency across time periods
- Overfitting control
- Feature importance stability

### Generated Outputs

After running the notebook, you'll get:

**Analysis Files:**
- `monthly_stability_results.csv` - Detailed monthly metrics
- `feature_stability_metrics.csv` - Feature stability metrics
- `stability_summary.json` - Quick summary statistics

**Visualization Files:**
- `monthly_stability_analysis.png` - Monthly performance plots
- `feature_stability_analysis.png` - Feature importance plots

For detailed documentation, see [STABILITY_ANALYSIS.md](STABILITY_ANALYSIS.md).

## Data Format

Expected data files in `data/` directory:
- `train.parquet` - Training data with `a6_flg` target and `month_dt` timestamp
- `test.parquet` - Test data for predictions
- `sample_submission.csv` - Submission format template

## Project Structure

```
.
├── main.py                      # Main marimo notebook
├── STABILITY_ANALYSIS.md        # Stability analysis documentation
├── pyproject.toml               # Project dependencies
├── data/                        # Data directory (gitignored)
│   ├── train.parquet
│   ├── test.parquet
│   └── sample_submission.csv
└── submission.csv               # Generated predictions (gitignored)
```

## Key Improvements

### Stability Analysis Features (NEW)

1. **Monthly Stability Analysis**
   - Time-series cross-validation with progressive training
   - Performance metrics by month (train/val AUC)
   - Overfitting detection via train-val gap
   - Statistical stability metrics

2. **Feature Importance Stability**
   - Feature importance tracking across time periods
   - Stability metrics (mean, CV, range)
   - Identifies stable vs. unstable features

3. **Comprehensive Reporting**
   - CSV exports for detailed analysis
   - JSON summary for quick review
   - Visualizations for easy interpretation

## License

This project is for competition use.
