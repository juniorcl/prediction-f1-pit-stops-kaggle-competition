import re
import optuna
import pickle

import numpy as np
import pandas as pd

from sklearn import set_config
from category_encoders import CatBoostEncoder

from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import TargetEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

from feature_engine.selection import DropFeatures


set_config(transform_output="pandas")

def dump_pickle(file_obj, file_path):
    with open(file_path, 'bw') as file:
        pickle.dump(file_obj, file)

column_transformer = ColumnTransformer([
    (
        'target_encoder', 
        TargetEncoder(), 
        ['driver', 'compound', 'race']
    ),
    (
        'catboost_encoder', 
        CatBoostEncoder(), 
        ['driver', 'compound', 'race']
    ),
    (
        'standard_scaler', 
        StandardScaler(), 
        ['lapnumber', 'position', 'raceprogress', 'year', 'position_norm', 'race_progress_sin', 'position_vs_mean']
    ),
    (
        'robust_scaler', 
        RobustScaler(), 
        [
            'position_change', 'cumulative_degradation', 'laptime_delta', 'laptime_s', 'stint', 'driver_mean_lap', 'tyrelife', 'delta_x_tyre_life', 
            'compound_tyre_life', 'stint_progress', 'tyre_life_ratio', 'degradation_per_lap', 'position_change_cum', 'laps_since_pit', 'lap_time_inv',  
            'lap_time_vs_race_mean', 'lap_time_x_tyre', 'position_x_progress', 'degradation_x_progress', 'race_progress_squared', 'driver_avg_position' 
        ]
    ),
], remainder="passthrough")


X_train = pd.read_parquet('../data/X_train.parquet')
y_train = pd.read_parquet('../data/y_train.parquet')


pipe_model = make_pipeline(
    column_transformer,
    KNeighborsClassifier(n_jobs=1)
).fit(X_train, y_train.PitNextLap)


perm_result = permutation_importance(
    estimator=model, 
    X=X_train, 
    y=y_train.PitNextLap, 
    n_jobs=3, 
    scoring='roc_auc'
)

importance_df = pd.DataFrame({
    "feature": X_train.columns.tolist(),
    "importance_mean": perm_result.importances_mean,
    "importance_std": perm_result.importances_std
}).sort_values(by="importance_mean", ascending=False)

features_to_drop = importance_df.query("importance_mean <= 0").feature.tolist()\


def objective(trial, X, y):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    aucs = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

        X_train_fold = X.iloc[train_idx, :]
        X_valid_fold = X.iloc[valid_idx, :]

        y_train_fold = y.iloc[train_idx, 0]
        y_valid_fold = y.iloc[valid_idx, 0]

        model = make_pipeline(
            column_transformer,
            DropFeatures(features_to_drop)
            PCA(
                n_components=trial.suggest_categorical("use_pca", [True, False]),
                random_state=42
            ),
            KNeighborsClassifier(
                n_neighbors=trial.suggest_int("n_neighbors", 3, 10),
                weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                metric=trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
                p=trial.suggest_int("p", 1, 2),
                leaf_size=trial.suggest_int("leaf_size", 10, 100),
                n_jobs=-1
            )
        ).fit(X_train_fold, y_train_fold)

        proba = model.predict_proba(X_valid_fold)[:, 1]

        auc = roc_auc_score(y_valid_fold, proba)
        aucs.append(auc)

        trial.report(np.mean(aucs), step=fold)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(aucs)

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30, n_jobs=-1, show_progress_bar=True)


print("\nBest trial:")
print("AUC:", study.best_trial.value)

print("\nBest params:")
print(study.best_trial.params)


model_tuned = make_pipeline(
    column_transformer,
    DropFeatures(features_to_drop)
    PCA(
        n_components=trial.suggest_categorical("use_pca", [True, False]),
        random_state=42
    ),
    KNeighborsClassifier(
        n_neighbors=trial.suggest_int("n_neighbors", 3, 10),
        weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
        metric=trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
        p=trial.suggest_int("p", 1, 2),
                leaf_size=trial.suggest_int("leaf_size", 10, 100),
                n_jobs=-1
            )
        )
).fit(X_train, y_train.PitNextLap)