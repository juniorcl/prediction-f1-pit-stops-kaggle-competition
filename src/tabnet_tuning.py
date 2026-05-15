import re
import torch
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import TargetEncoder, StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold

from pytorch_tabnet.tab_model import TabNetClassifier


set_config(enable_metadata_routing=True)

def dump_pickle(file_obj, file_path):
    with open(file_path, 'bw') as file:
        pickle.dump(file_obj, file)

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

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


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

aucs = []

for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):

    print(f"\n======== Fold {fold + 1} ========")

    X_tr = X_train.iloc[train_idx, :]
    X_val = X_train.iloc[valid_idx, :]

    y_tr = y_train.iloc[train_idx, 0]
    y_val = y_train.iloc[valid_idx, 0]

    X_tr_transformed = column_transformer.fit_transform(X_tr, y_tr)
    X_val_transformed = column_transformer.transform(X_val)

    tab_net = TabNetClassifier(
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-2),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 20, "gamma": 0.9},
        mask_type="entmax",
        seed=42,
        verbose=1
    )

    tab_net.fit(
        X_train=X_tr_transformed,
        y_train=y_tr,
        eval_set=[(X_val_transformed, y_val)],
        eval_name=["valid"],
        eval_metric=["auc"],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=11,
        drop_last=False
    )

    score = tab_net.predict_proba(X_val_transformed)[:, 1]

    auc = roc_auc_score(y_val, score)
    aucs.append(auc)

    print(f"Fold AUC: {auc:.6f}")

print("\n==============================")
print(f"CV AUC: {np.mean(aucs):.6f}")
print(f"STD: {np.std(aucs):.6f}")
print("==============================")


tab_net.save_model("../models/tabnet_model")