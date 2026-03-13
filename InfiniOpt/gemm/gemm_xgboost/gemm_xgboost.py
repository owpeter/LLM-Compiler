import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

XGB_PARAMS = {
    "objective": "rank:pairwise",
    "eval_metric": "ndcg",
    "max_depth": 8,
    "min_child_weight": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "gamma": 0.1,
    "lambda": 1.0,
    "alpha": 0.1,
    "tree_method": "hist",
    "random_state": 42,
}

DATA_PATH = "/root/LLM-Compiler/InfiniCore/test/profile-test/gemm_profile.csv"
MODEL_PATH = "xgboost_model.json"
META_PATH = "xgboost_model_meta.json"
TEST_SIZE = 0.2
N_ESTIMATORS = 500
N_JOBS = 0


def prepare_data(df):
    required_columns = [
        "m",
        "n",
        "k",
        "block_m",
        "block_n",
        "block_k",
        "unroll",
        "num_warps",
        "num_stages",
        "run_time",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    df = df.copy()
    df["total_flops"] = 2 * df["block_m"] * df["block_n"] * df["block_k"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df["run_time"] > 0]
    df["TFLOP"] = df["total_flops"] / df["run_time"] * 1e9
    df = df.dropna(subset=required_columns + ["TFLOP"])

    feature_columns = [
        "m",
        "n",
        "k",
        "block_m",
        "block_n",
        "block_k",
        "unroll",
        "num_warps",
        "num_stages",
    ]
    df["_group_key"] = df[["m", "n", "k"]].astype(str).agg("_".join, axis=1)
    df = df.sort_values(["_group_key", "TFLOP"], kind="mergesort").reset_index(drop=True)

    def assign_relevance(group):
        n = len(group)
        if n <= 1:
            group["_relevance"] = 0
            return group
        ranks = np.arange(n, dtype=np.float32)
        group["_relevance"] = np.rint(ranks / (n - 1) * 31).astype(np.int32)
        return group

    df = df.groupby("_group_key", sort=False, group_keys=False).apply(assign_relevance)
    return df, feature_columns


def build_model():
    return xgb.XGBRanker(
        objective=XGB_PARAMS["objective"],
        eval_metric=XGB_PARAMS["eval_metric"],
        n_estimators=N_ESTIMATORS,
        max_depth=XGB_PARAMS["max_depth"],
        min_child_weight=XGB_PARAMS["min_child_weight"],
        learning_rate=XGB_PARAMS["eta"],
        subsample=XGB_PARAMS["subsample"],
        colsample_bytree=XGB_PARAMS["colsample_bytree"],
        gamma=XGB_PARAMS["gamma"],
        reg_lambda=XGB_PARAMS["lambda"],
        reg_alpha=XGB_PARAMS["alpha"],
        tree_method=XGB_PARAMS["tree_method"],
        n_jobs=N_JOBS,
        random_state=XGB_PARAMS["random_state"],
    )


def save_metadata(path, features, metrics, params):
    payload = {
        "features": features,
        "metrics": metrics,
        "params": params,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_metadata(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_xgboost_model(model_path):
    model = xgb.Booster()
    model.load_model(model_path)
    return model


def predict_xgboost(feature_values, model_path="xgboost_model.json", meta_path="xgboost_model_meta.json"):
    meta = load_metadata(meta_path)
    features = meta["features"]

    if isinstance(feature_values, dict):
        missing = [name for name in features if name not in feature_values]
        if missing:
            raise ValueError(f"Missing features: {', '.join(missing)}")
        row = {name: feature_values[name] for name in features}
    else:
        values = list(feature_values)
        if len(values) != len(features):
            raise ValueError(f"Expected {len(features)} values, got {len(values)}")
        row = dict(zip(features, values))

    df = pd.DataFrame([row], columns=features)
    model = load_xgboost_model(model_path)
    dmatrix = xgb.DMatrix(df, feature_names=features)
    pred = model.predict(dmatrix)[0]
    return float(pred)


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    df, features = prepare_data(df)
    df = df.sort_values("_group_key").reset_index(drop=True)
    unique_keys = df["_group_key"].unique()
    train_keys, valid_keys = train_test_split(
        unique_keys,
        test_size=TEST_SIZE,
        random_state=XGB_PARAMS["random_state"],
        shuffle=True,
    )

    train_df = df[df["_group_key"].isin(train_keys)].sort_values("_group_key").reset_index(drop=True)
    valid_df = df[df["_group_key"].isin(valid_keys)].sort_values("_group_key").reset_index(drop=True)

    X_train = train_df[features]
    y_train = train_df["_relevance"]
    group_train = train_df.groupby("_group_key", sort=False).size().to_list()

    X_valid = valid_df[features]
    y_valid = valid_df["_relevance"]
    group_valid = valid_df.groupby("_group_key", sort=False).size().to_list()

    model = build_model()
    model.fit(
        X_train,
        y_train,
        group=group_train,
        eval_set=[(X_valid, y_valid)],
        eval_group=[group_valid],
        verbose=False,
    )

    evals_result = getattr(model, "evals_result_", None) or {}
    valid_metrics = evals_result.get("validation_0", {})
    ndcg_values = valid_metrics.get("ndcg") or []
    last_ndcg = float(ndcg_values[-1]) if ndcg_values else None
    metrics = {
        "ndcg": last_ndcg,
        "accuracy": last_ndcg,
        "train_samples": int(len(X_train)),
        "valid_samples": int(len(X_valid)),
        "train_groups": int(len(group_train)),
        "valid_groups": int(len(group_valid)),
    }

    model.save_model(MODEL_PATH)
    save_metadata(META_PATH, features, metrics, model.get_params())
    if last_ndcg is not None:
        print(f"Final accuracy (NDCG): {last_ndcg:.6f}")
    else:
        print("Final accuracy (NDCG): N/A")


if __name__ == "__main__":
    main()
