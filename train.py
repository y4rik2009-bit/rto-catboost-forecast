"""Прогноз РТО следующего месяца: линейные базы и CatBoost + Optuna."""

import os
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error


DATA_PATH = Path("data/train_2.csv")
OUTPUT_DIR = Path("outputs")
TASK_TYPE = os.getenv("CATBOOST_TASK_TYPE", "CPU").upper()  # GPU при наличии CUDA


def prepare_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Поместите исходный CSV сюда: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {"new_id", "Год", "Месяц", "РТО"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"В CSV отсутствуют столбцы: {sorted(missing_columns)}")
    if df["new_id"].isna().any():
        raise ValueError("В new_id есть пропуски")
    if not df["Месяц"].between(1, 12).all():
        raise ValueError("Месяц должен быть числом от 1 до 12")

    df["Date"] = (df["Год"] - 2023) * 12 + df["Месяц"]
    df = df.sort_values(["new_id", "Date"], ignore_index=True)

    # Исходная схема задачи: ровно 26 последовательных месяцев на каждый new_id.
    periods = df.groupby("new_id")["Date"].agg(["size", "min", "max", "nunique"])
    if not ((periods["size"] == 26) & (periods["min"] == 1)
            & (periods["max"] == 26) & (periods["nunique"] == 26)).all():
        raise ValueError("Нужны месяцы 1–26 без пропусков и повторов для каждого new_id")

    rto = pd.to_numeric(df["РТО"], errors="coerce")
    if not np.isfinite(rto).all() or (rto <= 0).any():
        raise ValueError("РТО должен быть заполнен, конечен и строго больше нуля")
    df["РТО"] = np.log(rto)

    categorical = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    categorical = [col for col in categorical if col != "new_id"]
    for col in categorical:
        df[col] = df[col].fillna("Неизвестно").astype(str).astype(object)

    numeric_features = [col for col in df.select_dtypes(include=[np.number]).columns
                        if col not in ("new_id", "РТО")]
    train_periods = df["Date"] <= 21
    means = df.loc[train_periods, numeric_features].mean()
    if means.isna().any():
        raise ValueError(f"Числовые признаки полностью пусты на обучении: {means[means.isna()].index.tolist()}")
    df[numeric_features] = df[numeric_features].fillna(means)

    df["target"] = df.groupby("new_id")["РТО"].shift(-1)
    history = df.loc[df["Date"] <= 25].copy()
    future = df.loc[df["Date"] == 26].drop(columns="target").copy()

    def split(start, end):
        part = history.loc[history["Date"].between(start, end)]
        return part.drop(columns="target"), part["target"]

    X_train, y_train = split(1, 21)
    X_valid, y_valid = split(22, 23)
    X_test, y_test = split(24, 25)
    numeric_baseline = [col for col in X_train.select_dtypes(include=[np.number]).columns
                        if col != "new_id"]
    return X_train, y_train, X_valid, y_valid, X_test, y_test, future, categorical, numeric_baseline


def main():
    if TASK_TYPE not in {"CPU", "GPU"}:
        raise ValueError("CATBOOST_TASK_TYPE должен быть CPU или GPU")
    (X_train, y_train, X_valid, y_valid, X_test, y_test,
     X_final, categorical, numeric_baseline) = prepare_data()
    OUTPUT_DIR.mkdir(exist_ok=True)

    device_params = {"task_type": TASK_TYPE}
    if TASK_TYPE == "GPU":
        device_params["devices"] = "0"

    def objective(trial):
        params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "grow_policy": "Lossguide",
            "depth": trial.suggest_int("depth", 5, 11),
            "max_leaves": trial.suggest_int("max_leaves", 16, 64),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 1000),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30),
            "random_strength": trial.suggest_float("random_strength", 0, 5),
            "bootstrap_type": "Bernoulli",
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "cat_features": categorical,
            "random_seed": 42,
            "verbose": 25,
            "allow_writing_files": False,
            **device_params,
        }
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                  early_stopping_rounds=50, use_best_model=True)
        trial.set_user_attr("best_iterations", model.tree_count_)
        return root_mean_squared_error(y_valid, model.predict(X_valid))

    db_path = OUTPUT_DIR / "optuna_logged_rmse.db"
    study = optuna.create_study(
        study_name=f"catboost_optimization_{TASK_TYPE.lower()}",
        direction="minimize",
        storage=f"sqlite:///{db_path.as_posix()}",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=50)

    linear = LinearRegression()
    linear.fit(X_train[numeric_baseline], y_train)
    linear_pred = linear.predict(X_test[numeric_baseline])
    print("Линейная модель, тест MAPE:", mean_absolute_percentage_error(
        np.exp(y_test), np.exp(linear_pred)))
    print("Последнее известное РТО, тест MAPE:", mean_absolute_percentage_error(
        np.exp(y_test), np.exp(X_test["РТО"])))

    model_catboost = CatBoostRegressor(
        **study.best_params,
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2000,
        cat_features=categorical,
        random_seed=42,
        verbose=True,
        allow_writing_files=False,
        grow_policy="Lossguide",
        bootstrap_type="Bernoulli",
        **device_params,
    )
    model_catboost.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                       early_stopping_rounds=100)

    for label, X_part, y_part in (("валидация", X_valid, y_valid),
                                  ("тест", X_test, y_test)):
        prediction_log = model_catboost.predict(X_part)
        print(f"CatBoost, {label} MAPE:", mean_absolute_percentage_error(
            np.exp(y_part), np.exp(prediction_log)))
        print(f"CatBoost, {label} RMSE (log РТО):",
              root_mean_squared_error(y_part, prediction_log))

    submission = pd.DataFrame({
        "new_id": X_final["new_id"].to_numpy(),
        "rto": np.exp(model_catboost.predict(X_final)),
    })
    output_path = OUTPUT_DIR / "test.csv"
    submission.to_csv(output_path, index=False)
    print(f"Прогноз сохранён: {output_path} ({len(submission)} строк)")


if __name__ == "__main__":
    main()
