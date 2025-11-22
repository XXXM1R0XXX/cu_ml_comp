import marimo

__generated_with = "0.18.0"
app = marimo.App(width="full")


@app.cell
def _():
    from catboost import CatBoostClassifier, Pool
    import pandas as pd
    import optuna
    from optuna.integration import CatBoostPruningCallback
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    from dataclasses import dataclass
    from simple_parsing import parse

    @dataclass
    class Args:
        train_path: str = "data/train.parquet"
        test_path: str = "data/test.parquet"

    args = parse(Args)
    return CatBoostClassifier, CatBoostPruningCallback, Pool, args, optuna, pd


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
def _(CatBoostClassifier, CatBoostPruningCallback, train_pool, val_pool):
    def objective(trial):
        # Определение пространства гиперпараметров
        param = {
            "objective": "Logloss",
            "eval_metric": "AUC",
            "iterations": 2000,  # Ставим больше, сработает early_stopping или pruning
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "random_strength": trial.suggest_float(
                "random_strength", 1e-8, 10.0, log=True
            ),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "od_type": "Iter",
            "od_wait": 50,
            "allow_writing_files": False,
            "verbose": False,
            'task_type': 'GPU'
        }

        # Условные параметры в зависимости от bootstrap_type
        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0, 10
            )
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        # Интеграция прунинга (остановка неперспективных запусков)
        pruning_callback = CatBoostPruningCallback(trial, "AUC")

        model = CatBoostClassifier(**param)

        model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=0,
            early_stopping_rounds=100,
            callbacks=[pruning_callback],
        )

        # Прунинг: проверяем, нужно ли прервать trial
        pruning_callback.check_pruned()

        return model.get_best_score()["validation"]["AUC"]
    return (objective,)


@app.cell
def _(objective, optuna):
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )

    # n_trials - количество попыток, timeout - ограничение по времени в секундах
    study.optimize(objective, n_trials=50, timeout=600)

    print(f"Best trial found: {study.best_value}")
    print(f"Params: {study.best_params}")
    return (study,)


@app.cell
def _(CatBoostClassifier, study, train_pool, val_pool):
    best_params = study.best_params.copy()

    # Добавляем статические параметры, которые не перебирали, но нужны для финального обучения
    best_params.update(
        {
            "iterations": 2000,
            "eval_metric": "AUC",
            "od_type": "Iter",
            "od_wait": 100,
            "allow_writing_files": False,
        }
    )

    print("Training final model with params:", best_params)

    final_model = CatBoostClassifier(**best_params)

    final_model.fit(train_pool, eval_set=val_pool, verbose=100, plot=False)
    return


@app.cell
def _(pd):
    ss_sub = pd.read_csv("data/sample_submission.csv")
    return (ss_sub,)


@app.cell
def _(model, ss_sub, test_pool):
    ss_sub["a6_flg"] = model.predict_proba(test_pool)[:, 1]
    return


@app.cell
def _(ss_sub):
    ss_sub.to_csv("submission.csv", index=False)
    return


if __name__ == "__main__":
    app.run()
