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

    @dataclass
    class Args:
        train_path: str = "data/train.parquet"
        test_path: str = "data/test.parquet"
        ssub_path: str = "data/sample_submission.csv"

    args = parse(Args)
    return CatBoostClassifier, Pool, args, optuna, pd


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
def _(Pool, test, train_df, val_df):
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
    test_pool = Pool(data=test, cat_features=["product"], timestamp=test["month_dt"])
    return test_pool, train_pool, val_pool


@app.cell
def _(CatBoostClassifier, train_pool, val_pool):
    def objective(trial):
        param = {
            "eval_metric": "AUC",
            "task_type": "GPU",
            "verbose": False,
        
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "iterations": trial.suggest_int("iterations", 500, 3000),
        
            "depth": trial.suggest_int("depth", 3, 12),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
            ),
        
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-9, 15.0, log=True),
            "model_size_reg": trial.suggest_float("model_size_reg", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 1e-9, 15.0, log=True),
        
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
        
            "border_count": trial.suggest_int("border_count", 32, 255),
        
            "leaf_estimation_method": trial.suggest_categorical(
                "leaf_estimation_method", ["Newton", "Gradient"]
            ),
            "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 15),
        
            "score_function": trial.suggest_categorical(
                "score_function", ["Cosine", "L2", "NewtonCosine", "NewtonL2"]
            ),
        }
    
        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 15.0)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.4, 1.0)
    
        if param["grow_policy"] == "Lossguide":
            param["max_leaves"] = trial.suggest_int("max_leaves", 16, 128)
    
        if param["grow_policy"] in ["Lossguide", "Depthwise"]:
            param["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 100)
    
        param["leaf_estimation_backtracking"] = trial.suggest_categorical(
            "leaf_estimation_backtracking", ["AnyImprovement", "Armijo"]
        )
    
        if param["grow_policy"] == "Lossguide":
            param["boosting_type"] = "Plain"
        else:
            param["boosting_type"] = trial.suggest_categorical(
                "boosting_type", ["Ordered", "Plain"]
            )
    
        param["od_type"] = trial.suggest_categorical("od_type", ["IncToDec", "Iter"])
        param["od_wait"] = trial.suggest_int("od_wait", 10, 50)
    
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
def _(objective, optuna):
    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=50, timeout=600)

    print(f"Best trial found: {study.best_value}")
    print(f"Params: {study.best_params}")
    return (study,)


@app.cell
def _(CatBoostClassifier, study, train_pool, val_pool):
    best_params = study.best_params.copy()

    best_params.update(
        {
            "eval_metric": "AUC",
            "task_type": "GPU",
            "random_seed": 56
        }
    )

    print("Training final model with params:", best_params)

    final_model = CatBoostClassifier(**best_params)

    final_model.fit(train_pool, eval_set=val_pool, verbose=100, plot=False)
    return (final_model,)


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
