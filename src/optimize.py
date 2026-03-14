"""
Lab 3: Hyperparameter Optimization with Optuna + Hydra + MLflow Nested Runs
Dataset: Healthcare Stroke Prediction (same as Lab 1)
Models: XGBoost (primary), RandomForest, LogisticRegression
"""

import os
import json
import pickle
import logging
import warnings
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
import optuna
from omegaconf import DictConfig, OmegaConf
import hydra

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_split(train_path: str, test_path: str, seed: int):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Target
    X_train = train_df.drop("stroke", axis=1)
    y_train = train_df["stroke"]
    X_test = test_df.drop("stroke", axis=1)
    y_test = test_df["stroke"]

    # Drop id column if present
    for col in ["id"]:
        if col in X_train.columns:
            X_train = X_train.drop(col, axis=1)
        if col in X_test.columns:
            X_test = X_test.drop(col, axis=1)

    return X_train, y_train, X_test, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

    numerical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def build_xgboost(trial, cfg, y_train) -> XGBClassifier:
    xgb_cfg = cfg.model.xgboost
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    return XGBClassifier(
        n_estimators=trial.suggest_int(
            "n_estimators", xgb_cfg.n_estimators.low, xgb_cfg.n_estimators.high
        ),
        max_depth=trial.suggest_int(
            "max_depth", xgb_cfg.max_depth.low, xgb_cfg.max_depth.high
        ),
        learning_rate=trial.suggest_float(
            "learning_rate",
            xgb_cfg.learning_rate.low,
            xgb_cfg.learning_rate.high,
            log=True,
        ),
        subsample=trial.suggest_float(
            "subsample", xgb_cfg.subsample.low, xgb_cfg.subsample.high
        ),
        colsample_bytree=trial.suggest_float(
            "colsample_bytree",
            xgb_cfg.colsample_bytree.low,
            xgb_cfg.colsample_bytree.high,
        ),
        min_child_weight=trial.suggest_int(
            "min_child_weight",
            xgb_cfg.min_child_weight.low,
            xgb_cfg.min_child_weight.high,
        ),
        gamma=trial.suggest_float("gamma", xgb_cfg.gamma.low, xgb_cfg.gamma.high),
        reg_alpha=trial.suggest_float(
            "reg_alpha", xgb_cfg.reg_alpha.low, xgb_cfg.reg_alpha.high
        ),
        reg_lambda=trial.suggest_float(
            "reg_lambda", xgb_cfg.reg_lambda.low, xgb_cfg.reg_lambda.high
        ),
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=cfg.seed,
        use_label_encoder=False,
    )


def build_random_forest(trial, cfg) -> RandomForestClassifier:
    rf_cfg = cfg.model.random_forest
    return RandomForestClassifier(
        n_estimators=trial.suggest_int(
            "n_estimators", rf_cfg.n_estimators.low, rf_cfg.n_estimators.high
        ),
        max_depth=trial.suggest_int(
            "max_depth", rf_cfg.max_depth.low, rf_cfg.max_depth.high
        ),
        min_samples_split=trial.suggest_int(
            "min_samples_split",
            rf_cfg.min_samples_split.low,
            rf_cfg.min_samples_split.high,
        ),
        min_samples_leaf=trial.suggest_int(
            "min_samples_leaf",
            rf_cfg.min_samples_leaf.low,
            rf_cfg.min_samples_leaf.high,
        ),
        max_features=trial.suggest_categorical(
            "max_features", list(rf_cfg.max_features)
        ),
        class_weight="balanced",
        random_state=cfg.seed,
    )


def build_logistic_regression(trial, cfg) -> LogisticRegression:
    lr_cfg = cfg.model.logistic_regression
    return LogisticRegression(
        C=trial.suggest_float("C", lr_cfg.C.low, lr_cfg.C.high, log=True),
        solver=trial.suggest_categorical("solver", list(lr_cfg.solver)),
        penalty=trial.suggest_categorical("penalty", list(lr_cfg.penalty)),
        class_weight="balanced",
        max_iter=1000,
        random_state=cfg.seed,
    )


def objective(trial, cfg, X_train, y_train, X_val, y_val, sampler_name: str):
    model_type = cfg.model.type
    preprocessor = build_preprocessor(X_train)

    # Build estimator
    if model_type == "xgboost":
        estimator = build_xgboost(trial, cfg, y_train)
    elif model_type == "random_forest":
        estimator = build_random_forest(trial, cfg)
    elif model_type == "logistic_regression":
        estimator = build_logistic_regression(trial, cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    # Train & evaluate
    if cfg.hpo.use_cv:
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cfg.hpo.cv_folds, scoring="f1", n_jobs=-1
        )
        metric_val = float(scores.mean())
        metric_std = float(scores.std())
    else:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        metric_val = float(f1_score(y_val, y_pred, zero_division=0))
        metric_std = 0.0
        roc_auc_val = float(roc_auc_score(y_val, y_proba))

    # Log as nested (child) MLflow run
    with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
        mlflow.set_tag("sampler", sampler_name)
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("trial_number", str(trial.number))
        mlflow.set_tag("seed", str(cfg.seed))

        mlflow.log_params(trial.params)
        mlflow.log_metric(cfg.hpo.metric, metric_val)
        if not cfg.hpo.use_cv:
            mlflow.log_metric("val_roc_auc", roc_auc_val)
        if metric_std > 0:
            mlflow.log_metric(f"{cfg.hpo.metric}_std", metric_std)

    return metric_val


def create_sampler(sampler_name: str, seed: int):
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    elif sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == "grid":
        # Grid sampler needs a search space — fallback to TPE for complex spaces
        logger.warning(
            "Grid sampler selected but requires explicit grid. Falling back to TPE."
        )
        return optuna.samplers.TPESampler(seed=seed)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")


def run_study(cfg: DictConfig):
    sampler_name = cfg.hpo.sampler
    n_trials = cfg.hpo.n_trials
    direction = cfg.hpo.direction
    model_type = cfg.model.type

    logger.info(
        f"Starting HPO | model={model_type} | sampler={sampler_name} | n_trials={n_trials}"
    )

    # Load data
    X_train, y_train, X_test, y_test = load_and_split(
        cfg.data.train_path, cfg.data.test_path, cfg.seed
    )

    # Split off validation set from training
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=cfg.seed, stratify=y_train
    )

    # Setup MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment_name = f"{cfg.mlflow.experiment_name}_{model_type}_{sampler_name}"
    mlflow.set_experiment(experiment_name)

    sampler = create_sampler(sampler_name, cfg.seed)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=f"{model_type}_{sampler_name}",
    )

    run_name = f"Study_{model_type}_{sampler_name}_n{n_trials}"

    with mlflow.start_run(run_name=run_name) as parent_run:
        # Log study-level params
        mlflow.set_tag("run_type", "study_parent")
        mlflow.set_tag("sampler", sampler_name)
        mlflow.set_tag("model_type", model_type)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("direction", direction)
        mlflow.log_param("metric", cfg.hpo.metric)
        mlflow.log_param("seed", cfg.seed)
        mlflow.log_param("use_cv", cfg.hpo.use_cv)

        # Log full config as artifact
        config_str = OmegaConf.to_yaml(cfg)
        config_path = "hpo_config.yaml"
        with open(config_path, "w") as f:
            f.write(config_str)
        mlflow.log_artifact(config_path)
        os.remove(config_path)

        # Run optimization
        study.optimize(
            lambda trial: objective(trial, cfg, X_tr, y_tr, X_val, y_val, sampler_name),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best_value = study.best_value
        best_params = study.best_params
        best_trial_number = study.best_trial.number

        logger.info(
            f"Best trial #{best_trial_number}: {cfg.hpo.metric}={best_value:.4f}"
        )
        logger.info(f"Best params: {best_params}")

        # Log best trial summary to parent run
        mlflow.log_metric(f"best_{cfg.hpo.metric}", best_value)
        mlflow.log_metric("best_trial_number", best_trial_number)

        best_params_path = "best_params.json"
        with open(best_params_path, "w") as f:
            json.dump(
                {
                    "best_trial": best_trial_number,
                    "best_value": best_value,
                    "best_params": best_params,
                    "metric": cfg.hpo.metric,
                    "sampler": sampler_name,
                    "model_type": model_type,
                },
                f,
                indent=2,
            )
        mlflow.log_artifact(best_params_path)
        os.remove(best_params_path)

        # ── Retrain final model with best params on FULL training data ──
        logger.info("Retraining best model on full training data...")
        preprocessor_final = build_preprocessor(X_train)

        if model_type == "xgboost":
            n_pos = int((y_train == 1).sum())
            n_neg = int((y_train == 0).sum())
            best_estimator = XGBClassifier(
                **best_params,
                scale_pos_weight=n_neg / n_pos,
                eval_metric="logloss",
                random_state=cfg.seed,
                use_label_encoder=False,
            )
        elif model_type == "random_forest":
            best_estimator = RandomForestClassifier(
                **best_params,
                class_weight="balanced",
                random_state=cfg.seed,
            )
        elif model_type == "logistic_regression":
            best_estimator = LogisticRegression(
                **best_params,
                class_weight="balanced",
                max_iter=1000,
                random_state=cfg.seed,
            )

        final_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor_final),
                ("model", best_estimator),
            ]
        )
        final_pipeline.fit(X_train, y_train)

        # Evaluate on test set
        y_pred_test = final_pipeline.predict(X_test)
        y_proba_test = final_pipeline.predict_proba(X_test)[:, 1]
        test_f1 = float(f1_score(y_test, y_pred_test, zero_division=0))
        test_roc_auc = float(roc_auc_score(y_test, y_proba_test))

        mlflow.log_metric("final_test_f1", test_f1)
        mlflow.log_metric("final_test_roc_auc", test_roc_auc)

        logger.info(
            f"Final model — Test F1: {test_f1:.4f} | Test ROC-AUC: {test_roc_auc:.4f}"
        )

        # Save model as artifact
        os.makedirs("models", exist_ok=True)
        model_filename = f"models/best_model_{model_type}_{sampler_name}.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(final_pipeline, f)
        mlflow.log_artifact(model_filename)

        # Log model to MLflow model registry if configured
        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(final_pipeline, artifact_path="best_model")

        parent_run_id = parent_run.info.run_id
        logger.info(f"Parent run ID: {parent_run_id}")

    return study


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Lab 3: Hyperparameter Optimization")
    logger.info(OmegaConf.to_yaml(cfg))

    study = run_study(cfg)

    # Print summary
    print(f"HPO COMPLETED")
    print(f"  Model    : {cfg.model.type}")
    print(f"  Sampler  : {cfg.hpo.sampler}")
    print(f"  Trials   : {cfg.hpo.n_trials}")
    print(f"  Best {cfg.hpo.metric:6s}: {study.best_value:.4f}")
    print(f"  Best trial #{study.best_trial.number}")
    print("  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
