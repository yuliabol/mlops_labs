import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import Counter

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    X = df.drop('stroke', axis=1)
    y = df['stroke']

    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['number']).columns

    if 'id' in numerical_cols:
        numerical_cols = numerical_cols.drop('id')
        X = X.drop('id', axis=1)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    return X, y, preprocessor

def train_model(X, y, preprocessor, max_depth, args):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    n_pos = sum(y_train == 1)
    n_neg = sum(y_train == 0)
    scale_pos_weight = n_neg / n_pos

    model = XGBClassifier(
        max_depth=max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )

    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    return pipeline, X_train, X_test, y_train, y_test


def find_optimal_threshold(y_true, y_proba):
    best_f1 = 0
    best_threshold = 0.5

    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1

def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    optimal_threshold, _ = find_optimal_threshold(y_train, train_proba)

    test_pred = (test_proba >= optimal_threshold).astype(int)
    train_pred = (train_proba >= optimal_threshold).astype(int)

    return {
        "optimal_threshold": optimal_threshold,
        "train_acc": accuracy_score(y_train, train_pred),
        "test_acc": accuracy_score(y_test, test_pred),
        "train_f1": f1_score(y_train, train_pred),
        "test_f1": f1_score(y_test, test_pred),
        "train_precision": precision_score(y_train, train_pred),
        "test_precision": precision_score(y_test, test_pred),
        "train_recall": recall_score(y_train, train_pred),
        "test_recall": recall_score(y_test, test_pred),
        "test_pred": test_pred
    }

def main(args):
    mlflow.set_experiment("XGBoost Stroke Prediction 2")

    depths = [2, 3, 4, 5, 6, 10, 20]

    df = load_data(args.data_path)
    X, y, preprocessor = preprocess_data(df)

    for depth in depths:
        with mlflow.start_run(run_name=f"XGB_depth{depth}_balanced"):

            mlflow.set_tag("model_type", "XGBoost")
            mlflow.set_tag("study", "scale_pos_weight tuning")
            mlflow.log_param("max_depth", depth)
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("learning_rate", args.learning_rate)
            mlflow.log_param("subsample", args.subsample)
            mlflow.log_param("colsample_bytree", args.colsample_bytree)

            pipeline, X_train, X_test, y_train, y_test = train_model(
                X, y, preprocessor, depth, args
            )

            metrics = evaluate_model(pipeline, X_train, y_train, X_test, y_test)

            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            from sklearn.metrics import roc_auc_score
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            mlflow.log_metric("test_roc_auc", roc_auc)

            for k, v in metrics.items():
                if k != "test_pred":
                    mlflow.log_metric(k, v)

            print(f"\n{'='*60}")
            print(f"Depth = {depth}")
            print(f"{'='*60}")
            print(f"Train Acc: {metrics['train_acc']:.3f} | Test Acc: {metrics['test_acc']:.3f}")
            print(f"Train F1: {metrics['train_f1']:.3f} | Test F1: {metrics['test_f1']:.3f}")
            print(f"Train Precision: {metrics['train_precision']:.3f} | Test Precision: {metrics['test_precision']:.3f}")
            print(f"Train Recall: {metrics['train_recall']:.3f} | Test Recall: {metrics['test_recall']:.3f}")
            print(f"Test ROC-AUC: {roc_auc:.3f}")

            mlflow.sklearn.log_model(pipeline, f"xgb_model_depth_{depth}")

            # Confusion Matrix
            y_test_pred = metrics['test_pred']
            cm = confusion_matrix(y_test, y_test_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix (depth={depth})')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            cm_fname = f"cm_depth_{depth}.png"
            plt.savefig(cm_fname)
            mlflow.log_artifact(cm_fname)
            plt.close()
            if os.path.exists(cm_fname):
                os.remove(cm_fname)

            fitted_preprocessor = pipeline.named_steps["preprocess"]
            xgb_model = pipeline.named_steps["model"]

            num_features = fitted_preprocessor.transformers_[0][2]
            cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_features = cat_encoder.get_feature_names_out(
                fitted_preprocessor.transformers_[1][2]
            )

            feature_names = np.concatenate([num_features, cat_features])
            importances = xgb_model.feature_importances_

            indices = np.argsort(importances)[::-1][:5]

            plt.figure(figsize=(10, 6))
            plt.title(f"Top 5 Feature Importances (depth={depth})")
            plt.bar(range(len(indices)), importances[indices], color='skyblue')
            plt.xticks(range(len(indices)), feature_names[indices], rotation=45, ha='right')
            plt.tight_layout()

            fi_fname = f"feature_importance_depth_{depth}.png"
            plt.savefig(fi_fname)
            mlflow.log_artifact(fi_fname)
            plt.close()
            if os.path.exists(fi_fname):
                os.remove(fi_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stroke Prediction Model")
    parser.add_argument("--data_path", type=str, default="data/raw/healthcare-dataset-stroke-data.csv")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    args = parser.parse_args()
    main(args)
