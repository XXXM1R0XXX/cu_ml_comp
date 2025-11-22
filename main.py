import marimo

__generated_with = "0.18.0"
app = marimo.App(width="full")


@app.cell
def _():
    from catboost import CatBoostClassifier, Pool
    import pandas as pd
    import optuna
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    from dataclasses import dataclass
    from simple_parsing import parse
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    from collections import defaultdict

    @dataclass
    class Args:
        train_path: str = "data/train.parquet"
        test_path: str = "data/test.parquet"
        ssub_path: str = "data/sample_submission.csv"

    args = parse(Args)

    SEED = 228
    
    # Stability analysis configuration constants
    MAX_TIME_SPLITS = 5  # Maximum number of time-series cross-validation splits
    FOLD_TRAIN_VAL_SPLIT = 0.8  # Train/val split ratio within each fold
    CV_MIN_MEAN_THRESHOLD = 1e-6  # Minimum mean importance for CV calculation
    
    return CatBoostClassifier, Pool, SEED, args, optuna, pd, np, plt, json, defaultdict, MAX_TIME_SPLITS, FOLD_TRAIN_VAL_SPLIT, CV_MIN_MEAN_THRESHOLD


@app.cell
def _(args, pd):
    train = pd.read_parquet(args.train_path, engine="fastparquet")
    train
    return (train,)


@app.cell
def _(args, pd):
    test = pd.read_parquet(args.test_path, engine="fastparquet")
    test
    return (test,)


@app.cell
def _(train):
    split_idx = int(len(train) * 0.8)

    train_df = train.iloc[:split_idx]
    val_df = train.iloc[split_idx:]
    return train_df, val_df


@app.cell
def _(Pool, test, train, train_df, val_df):
    train_pool = Pool(
        data=train_df.drop(["a6_flg"], axis=1),
        label=train_df["a6_flg"],
        cat_features=["product"],
        timestamp=train_df["month_dt"],
    )
    val_pool = Pool(
        data=val_df.drop(["a6_flg"], axis=1),
        label=val_df["a6_flg"],
        cat_features=["product"],
        timestamp=val_df["month_dt"],
    )
    full_train_pool = Pool(
        data=train.drop(["a6_flg"], axis=1),
        label=train["a6_flg"],
        cat_features=["product"],
        timestamp=train["month_dt"],
    )
    test_pool = Pool(data=test, cat_features=["product"], timestamp=test["month_dt"])
    return full_train_pool, test_pool, train_pool, val_pool


@app.cell
def _(CatBoostClassifier, SEED, train_pool, val_pool):
    def objective(trial):
        param = {
            "eval_metric": "AUC",
            "task_type": "GPU",
            "verbose": False,
            "random_seed": SEED,
            "iterations": trial.suggest_int("iterations", 1000, 2000),
            "loss_function": trial.suggest_categorical(
                "loss_function", ["Logloss", "CrossEntropy"]
            ),

            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["Ordered", "Plain"]
            ),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),

            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),

            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-9, 15.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-9, 15.0, log=True),

            "border_count": trial.suggest_int("border_count", 32, 255),
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0, 20
            )
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        if param["loss_function"] == "Logloss":
            param["auto_class_weights"] = trial.suggest_categorical(
                "auto_class_weights", ["None", "Balanced", "SqrtBalanced"]
            )

        model = CatBoostClassifier(**param)

        model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=0,
            early_stopping_rounds=100,
        )

        return model.get_best_score()["validation"]["AUC"]
    return (objective,)


@app.cell
def _(SEED, objective, optuna):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    study.optimize(objective, n_trials=50, timeout=600)

    print(f"Best trial found: {study.best_value}")
    print(f"Params: {study.best_params}")
    return (study,)


@app.cell
def _(CatBoostClassifier, SEED, full_train_pool, study, train_pool, val_pool):
    best_params = study.best_params.copy()

    best_params.update({"eval_metric": "AUC", "task_type": "GPU", "random_seed": SEED})

    print("Training validation model with params:", best_params)

    validation_model = CatBoostClassifier(**best_params)

    validation_model.fit(train_pool, eval_set=val_pool, verbose=100, plot=False)

    print("Retraining on the full dataset for inference")

    final_model = CatBoostClassifier(**best_params)

    final_model.fit(full_train_pool, verbose=100, plot=False)
    return (final_model,)


@app.cell
def _(CatBoostClassifier, FOLD_TRAIN_VAL_SPLIT, MAX_TIME_SPLITS, Pool, SEED, defaultdict, np, pd, roc_auc_score, train):
    """Monthly stability analysis using time-based cross-validation"""
    
    def analyze_monthly_stability(train_data, best_params):
        """
        Analyze model stability across different time periods using monthly splits.
        This helps identify if the model performance degrades over time or has
        high variance across different months.
        """
        # Get unique months sorted
        train_sorted = train_data.sort_values('month_dt')
        unique_months = sorted(train_sorted['month_dt'].unique())
        
        print(f"Found {len(unique_months)} unique months in training data")
        print(f"Month range: {unique_months[0]} to {unique_months[-1]}")
        
        # Results storage
        monthly_results = {
            'month': [],
            'train_auc': [],
            'val_auc': [],
            'train_val_gap': [],
            'n_train': [],
            'n_val': []
        }
        
        # Use TimeSeriesSplit-like approach: train on earlier months, validate on later
        n_splits = min(MAX_TIME_SPLITS, len(unique_months) - 1)  # At least 2 months needed
        
        if len(unique_months) < 2:
            print("Not enough months for stability analysis")
            return None
            
        # Calculate split points
        for split_idx in range(n_splits):
            # Progressive training: use more data as we progress
            train_months_count = len(unique_months) - n_splits + split_idx
            train_months = unique_months[:train_months_count]
            val_month = unique_months[train_months_count]
            
            print(f"\nSplit {split_idx + 1}/{n_splits}:")
            print(f"  Training months: {train_months[0]} to {train_months[-1]} ({len(train_months)} months)")
            print(f"  Validation month: {val_month}")
            
            # Split data
            train_mask = train_sorted['month_dt'].isin(train_months)
            val_mask = train_sorted['month_dt'] == val_month
            
            fold_train = train_sorted[train_mask]
            fold_val = train_sorted[val_mask]
            
            if len(fold_val) == 0:
                continue
                
            # Create pools
            fold_train_pool = Pool(
                data=fold_train.drop(['a6_flg'], axis=1),
                label=fold_train['a6_flg'],
                cat_features=['product'],
                timestamp=fold_train['month_dt']
            )
            
            fold_val_pool = Pool(
                data=fold_val.drop(['a6_flg'], axis=1),
                label=fold_val['a6_flg'],
                cat_features=['product'],
                timestamp=fold_val['month_dt']
            )
            
            # Train model
            fold_params = best_params.copy()
            fold_params.update({
                'eval_metric': 'AUC',
                'task_type': 'GPU',
                'random_seed': SEED,
                'verbose': False
            })
            
            fold_model = CatBoostClassifier(**fold_params)
            fold_model.fit(fold_train_pool, eval_set=fold_val_pool, 
                          verbose=0, early_stopping_rounds=100)
            
            # Calculate metrics
            train_preds = fold_model.predict_proba(fold_train_pool)[:, 1]
            val_preds = fold_model.predict_proba(fold_val_pool)[:, 1]
            
            train_auc = roc_auc_score(fold_train['a6_flg'], train_preds)
            val_auc = roc_auc_score(fold_val['a6_flg'], val_preds)
            
            monthly_results['month'].append(val_month)
            monthly_results['train_auc'].append(train_auc)
            monthly_results['val_auc'].append(val_auc)
            monthly_results['train_val_gap'].append(train_auc - val_auc)
            monthly_results['n_train'].append(len(fold_train))
            monthly_results['n_val'].append(len(fold_val))
            
            print(f"  Train AUC: {train_auc:.4f}")
            print(f"  Val AUC: {val_auc:.4f}")
            print(f"  Train-Val Gap: {train_auc - val_auc:.4f}")
        
        results_df = pd.DataFrame(monthly_results)
        
        # Calculate stability metrics
        print("\n" + "="*60)
        print("STABILITY ANALYSIS SUMMARY")
        print("="*60)
        print(f"Mean Validation AUC: {results_df['val_auc'].mean():.4f}")
        print(f"Std Validation AUC: {results_df['val_auc'].std():.4f}")
        print(f"Min Validation AUC: {results_df['val_auc'].min():.4f}")
        print(f"Max Validation AUC: {results_df['val_auc'].max():.4f}")
        print(f"AUC Range: {results_df['val_auc'].max() - results_df['val_auc'].min():.4f}")
        print(f"\nMean Train-Val Gap (Overfitting): {results_df['train_val_gap'].mean():.4f}")
        print(f"Std Train-Val Gap: {results_df['train_val_gap'].std():.4f}")
        print(f"Max Train-Val Gap: {results_df['train_val_gap'].max():.4f}")
        
        return results_df
    
    # Run analysis if we have the study results
    monthly_stability_results = None
    return (analyze_monthly_stability, monthly_stability_results)


@app.cell
def _(analyze_monthly_stability, study, train):
    """Execute monthly stability analysis"""
    if study is not None:
        monthly_stability_results = analyze_monthly_stability(train, study.best_params)
    else:
        monthly_stability_results = None
    monthly_stability_results
    return (monthly_stability_results,)


@app.cell
def _(monthly_stability_results, plt):
    """Visualize monthly stability results"""
    if monthly_stability_results is not None and len(monthly_stability_results) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Train vs Val AUC over time
        ax1 = axes[0, 0]
        ax1.plot(monthly_stability_results['month'], monthly_stability_results['train_auc'], 
                marker='o', label='Train AUC', linewidth=2)
        ax1.plot(monthly_stability_results['month'], monthly_stability_results['val_auc'], 
                marker='s', label='Validation AUC', linewidth=2)
        ax1.set_xlabel('Month')
        ax1.set_ylabel('AUC Score')
        ax1.set_title('Model Performance by Month')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Train-Val Gap (Overfitting indicator)
        ax2 = axes[0, 1]
        ax2.bar(monthly_stability_results['month'], monthly_stability_results['train_val_gap'], 
               color='coral', alpha=0.7)
        ax2.axhline(y=0.05, color='r', linestyle='--', label='0.05 threshold')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Train-Val AUC Gap')
        ax2.set_title('Overfitting by Month (Train-Val Gap)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Validation AUC distribution
        ax3 = axes[1, 0]
        ax3.hist(monthly_stability_results['val_auc'], bins=10, color='skyblue', 
                alpha=0.7, edgecolor='black')
        ax3.axvline(monthly_stability_results['val_auc'].mean(), color='r', 
                   linestyle='--', linewidth=2, label=f"Mean: {monthly_stability_results['val_auc'].mean():.3f}")
        ax3.set_xlabel('Validation AUC')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Validation AUC Across Months')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Sample size by month
        ax4 = axes[1, 1]
        ax4.bar(monthly_stability_results['month'], monthly_stability_results['n_val'], 
               color='lightgreen', alpha=0.7)
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Number of Samples')
        ax4.set_title('Validation Set Size by Month')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('monthly_stability_analysis.png', dpi=150, bbox_inches='tight')
        print("Saved stability plots to: monthly_stability_analysis.png")
        fig
    return


@app.cell
def _(CatBoostClassifier, CV_MIN_MEAN_THRESHOLD, FOLD_TRAIN_VAL_SPLIT, MAX_TIME_SPLITS, Pool, SEED, defaultdict, np, pd, train):
    """Feature importance stability analysis"""
    
    def analyze_feature_stability(train_data, best_params):
        """
        Analyze feature importance stability across different time periods.
        This helps identify which features are consistently important vs. those
        that vary in importance over time.
        """
        train_sorted = train_data.sort_values('month_dt')
        unique_months = sorted(train_sorted['month_dt'].unique())
        
        if len(unique_months) < 2:
            print("Not enough months for feature stability analysis")
            return None
        
        # Store feature importances for each time period
        feature_importances = defaultdict(list)
        month_list = []
        
        print("\nAnalyzing feature importance across time periods...")
        
        # Split into time windows
        n_splits = min(MAX_TIME_SPLITS, len(unique_months) - 1)
        
        for split_idx in range(n_splits):
            train_months_count = len(unique_months) - n_splits + split_idx
            train_months = unique_months[:train_months_count]
            
            # Get data for this time period
            train_mask = train_sorted['month_dt'].isin(train_months)
            fold_train = train_sorted[train_mask]
            
            # Create pool
            fold_train_pool = Pool(
                data=fold_train.drop(['a6_flg'], axis=1),
                label=fold_train['a6_flg'],
                cat_features=['product'],
                timestamp=fold_train['month_dt']
            )
            
            # Create a small validation set for early stopping (configurable split ratio)
            val_split_idx = int(len(fold_train) * FOLD_TRAIN_VAL_SPLIT)
            fold_train_subset = fold_train.iloc[:val_split_idx]
            fold_val_subset = fold_train.iloc[val_split_idx:]
            
            fold_train_subset_pool = Pool(
                data=fold_train_subset.drop(['a6_flg'], axis=1),
                label=fold_train_subset['a6_flg'],
                cat_features=['product'],
                timestamp=fold_train_subset['month_dt']
            )
            
            fold_val_subset_pool = Pool(
                data=fold_val_subset.drop(['a6_flg'], axis=1),
                label=fold_val_subset['a6_flg'],
                cat_features=['product'],
                timestamp=fold_val_subset['month_dt']
            )
            
            # Train model with early stopping for consistency
            fold_params = best_params.copy()
            fold_params.update({
                'eval_metric': 'AUC',
                'task_type': 'GPU',
                'random_seed': SEED,
                'verbose': False
            })
            
            fold_model = CatBoostClassifier(**fold_params)
            fold_model.fit(fold_train_subset_pool, eval_set=fold_val_subset_pool, 
                          verbose=0, early_stopping_rounds=100)
            
            # Get feature importances from the trained model
            importances = fold_model.get_feature_importance()
            feature_names = fold_train.drop(['a6_flg'], axis=1).columns
            
            month_list.append(f"Period_{split_idx+1}")
            for feat_name, importance in zip(feature_names, importances):
                feature_importances[feat_name].append(importance)
        
        # Create DataFrame
        importance_df = pd.DataFrame(feature_importances, index=month_list)
        
        # Calculate stability metrics for each feature
        stability_metrics = pd.DataFrame({
            'feature': importance_df.columns,
            'mean_importance': importance_df.mean(),
            'std_importance': importance_df.std(),
            'cv_importance': importance_df.apply(
                lambda col: col.std() / col.mean() if col.mean() > CV_MIN_MEAN_THRESHOLD else 0.0, axis=0
            ),  # Coefficient of variation with safe division
            'min_importance': importance_df.min(),
            'max_importance': importance_df.max(),
            'range_importance': importance_df.max() - importance_df.min()
        }).sort_values('mean_importance', ascending=False)
        
        print("\n" + "="*60)
        print("FEATURE STABILITY ANALYSIS")
        print("="*60)
        print("\nTop 10 Features by Mean Importance:")
        print(stability_metrics.head(10).to_string())
        
        print("\n\nTop 10 Most Stable Features (lowest CV):")
        print(stability_metrics.nsmallest(10, 'cv_importance')[['feature', 'mean_importance', 'cv_importance']].to_string())
        
        print("\n\nTop 10 Least Stable Features (highest CV):")
        print(stability_metrics.nlargest(10, 'cv_importance')[['feature', 'mean_importance', 'cv_importance']].to_string())
        
        return importance_df, stability_metrics
    
    feature_importance_results = None
    feature_stability_metrics = None
    return (analyze_feature_stability, feature_importance_results, feature_stability_metrics)


@app.cell
def _(analyze_feature_stability, study, train):
    """Execute feature stability analysis"""
    if study is not None:
        feature_importance_results, feature_stability_metrics = analyze_feature_stability(train, study.best_params)
    else:
        feature_importance_results = None
        feature_stability_metrics = None
    feature_stability_metrics
    return (feature_importance_results, feature_stability_metrics)


@app.cell
def _(feature_importance_results, feature_stability_metrics, plt):
    """Visualize feature importance stability"""
    if feature_importance_results is not None and feature_stability_metrics is not None:
        # Top 10 most important features
        top_features = feature_stability_metrics.nlargest(10, 'mean_importance')['feature'].tolist()
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Feature importance over time for top features
        ax1 = axes[0]
        for feature in top_features:
            if feature in feature_importance_results.columns:
                ax1.plot(feature_importance_results.index, 
                        feature_importance_results[feature], 
                        marker='o', label=feature, linewidth=2)
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Feature Importance')
        ax1.set_title('Top 10 Feature Importances Over Time')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Feature stability (mean importance vs. coefficient of variation)
        ax2 = axes[1]
        scatter = ax2.scatter(feature_stability_metrics['mean_importance'], 
                            feature_stability_metrics['cv_importance'],
                            alpha=0.6, s=100)
        
        # Annotate top features
        for idx, row in feature_stability_metrics.head(10).iterrows():
            ax2.annotate(row['feature'], 
                        (row['mean_importance'], row['cv_importance']),
                        fontsize=8, alpha=0.7)
        
        ax2.set_xlabel('Mean Feature Importance')
        ax2.set_ylabel('Coefficient of Variation (instability)')
        ax2.set_title('Feature Importance vs. Stability')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_stability_analysis.png', dpi=150, bbox_inches='tight')
        print("Saved feature stability plots to: feature_stability_analysis.png")
        return fig
    return


@app.cell
def _(json, monthly_stability_results, feature_stability_metrics):
    """Export stability analysis results"""
    
    if monthly_stability_results is not None:
        # Save monthly stability results
        monthly_stability_results.to_csv('monthly_stability_results.csv', index=False)
        print("Saved monthly stability results to: monthly_stability_results.csv")
        
        # Calculate and save summary statistics
        summary_stats = {
            'monthly_stability': {
                'mean_val_auc': float(monthly_stability_results['val_auc'].mean()),
                'std_val_auc': float(monthly_stability_results['val_auc'].std()),
                'min_val_auc': float(monthly_stability_results['val_auc'].min()),
                'max_val_auc': float(monthly_stability_results['val_auc'].max()),
                'auc_range': float(monthly_stability_results['val_auc'].max() - monthly_stability_results['val_auc'].min()),
                'mean_overfitting_gap': float(monthly_stability_results['train_val_gap'].mean()),
                'max_overfitting_gap': float(monthly_stability_results['train_val_gap'].max())
            }
        }
        
        if feature_stability_metrics is not None:
            # Save feature stability results
            feature_stability_metrics.to_csv('feature_stability_metrics.csv', index=False)
            print("Saved feature stability metrics to: feature_stability_metrics.csv")
            
            # Add feature stability summary
            summary_stats['feature_stability'] = {
                'n_features': len(feature_stability_metrics),
                'top_10_features': feature_stability_metrics.head(10)['feature'].tolist(),
                'most_stable_features': feature_stability_metrics.nsmallest(5, 'cv_importance')['feature'].tolist(),
                'least_stable_features': feature_stability_metrics.nlargest(5, 'cv_importance')['feature'].tolist()
            }
        
        # Save summary as JSON
        with open('stability_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        print("Saved stability summary to: stability_summary.json")
        
        print("\n" + "="*60)
        print("STABILITY ANALYSIS COMPLETE")
        print("="*60)
        print("\nFiles generated:")
        print("  - monthly_stability_results.csv")
        print("  - monthly_stability_analysis.png")
        print("  - feature_stability_metrics.csv")
        print("  - feature_stability_analysis.png")
        print("  - stability_summary.json")
    return


@app.cell
def _(args, pd):
    ss_sub = pd.read_csv(args.ssub_path)
    return (ss_sub,)


@app.cell
def _(final_model, ss_sub, test_pool):
    ss_sub["a6_flg"] = final_model.predict_proba(test_pool)[:, 1]
    return


@app.cell
def _(ss_sub):
    ss_sub.to_csv("submission.csv", index=False)
    return


if __name__ == "__main__":
    app.run()
