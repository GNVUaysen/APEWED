# -*- coding: utf-8 -*-
"""
Modos de uso:
1) Construcción completa desde archivos fuente:
   python3 0-data-air_make_model.py --build-raw

2) Solo modelado desde dataset_model_clean_raw.csv ya existente:
   python3 0-data-air_make_model.py
"""

from pathlib import Path
import argparse
import warnings
import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

# =========================
# Configuración general
# =========================
DATA_DIR = Path(".")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

FILES = {
    "PM25": "MP2.5.csv",
    "PM10": "MP10.csv",
    "temp": "temp.csv",
    "hum": "hum.csv",
    "pres": "pres.csv",
    "wind": "wind.csv",
    "winddir": "wind_dir.csv",
}

OUT_RAW = OUTPUT_DIR / "dataset_model_clean_raw.csv"
OUT_SCALED = OUTPUT_DIR / "dataset_model_clean.csv"
OUT_SCALER = OUTPUT_DIR / "scaler.npy"
DEFAULT_RAW_INPUT = "dataset_model_clean_raw.csv"

OUTLIER_Q_LOW = 0.01
OUTLIER_Q_HIGH = 0.99
HORIZON_DAYS = 2
CV_SPLITS = 5
RECALL_TARGET = 0.80
CRITICAL_CATS = {"preemergencia", "emergencia"}
BASE_NUMERIC = ["PM25", "PM10", "temp", "hum", "pres", "wind", "winddir"]
LAGS = [1, 2, 3, 5, 7, 14]
ROLL_WINDOWS = [2, 3, 5, 7, 14]

# =========================
# Bloque 1: Limpieza base
# =========================
def read_csv_robust(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", engine="python", dtype=str)
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_date_yymmdd(series: pd.Series) -> pd.Series:
    series = series.astype(str).str.strip()

    def convert(x: str):
        x = "".join(ch for ch in x if ch.isdigit())
        if len(x) != 6:
            return np.nan
        yy = int(x[:2])
        yyyy = 2000 + yy if yy < 70 else 1900 + yy
        mm = x[2:4]
        dd = x[4:6]
        return f"{yyyy}-{mm}-{dd}"

    return pd.to_datetime(series.map(convert), errors="coerce")


def to_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load_variable(path: Path, var_name: str) -> pd.DataFrame:
    df = read_csv_robust(path)

    date_candidates = [c for c in df.columns if "FECHA" in c.upper()]
    if not date_candidates:
        raise ValueError(f"[{path.name}] No se encontró columna FECHA")
    date_col = date_candidates[0]

    hour_candidates = [c for c in df.columns if "HORA" in c.upper()]
    has_hour = len(hour_candidates) > 0
    hour_col = hour_candidates[0] if has_hour else None

    df["date"] = parse_date_yymmdd(df[date_col])

    ignore = set([date_col] + ([hour_col] if hour_col else []))
    value_cols = [c for c in df.columns if c not in ignore and c != "date"]
    if not value_cols:
        raise ValueError(f"[{path.name}] No se encontró columna de valor")

    best_col, best_count, best_series = None, -1, None
    for c in value_cols:
        s = to_numeric(df[c])
        cnt = int(s.notna().sum())
        if cnt > best_count:
            best_count, best_col, best_series = cnt, c, s

    if best_col is None or best_count == 0:
        raise ValueError(f"[{path.name}] No se encontró serie numérica válida")

    out = pd.DataFrame({"date": df["date"], var_name: best_series}).dropna(subset=["date"])
    out = out.set_index("date").sort_index()
    out = out.resample("D").mean()
    return out


def remove_outliers(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        ql = out[c].quantile(OUTLIER_Q_LOW)
        qh = out[c].quantile(OUTLIER_Q_HIGH)
        out = out[(out[c] >= ql) & (out[c] <= qh)]
    return out


def air_quality_category(pm25: float) -> str:
    if pm25 > 169:
        return "emergencia"
    elif pm25 > 109:
        return "preemergencia"
    elif pm25 > 79:
        return "alerta"
    elif pm25 > 50:
        return "regular"
    else:
        return "buena"


def add_future_categories_by_calendar_day(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["CAT"] = out["PM25"].apply(air_quality_category)
    cat_map = out["CAT"].copy()
    for days_ahead, col in [(1, "CAT_24h"), (2, "CAT_48h"), (3, "CAT_72h")]:
        target_dates = pd.Series(out.index + pd.Timedelta(days=days_ahead), index=out.index)
        out[col] = target_dates.map(cat_map)
    return out


def build_raw_from_source_files() -> pd.DataFrame:
    frames = []
    for var, fname in FILES.items():
        fpath = (DATA_DIR / fname).resolve()
        if not fpath.exists():
            raise FileNotFoundError(f"No encontrado: {fpath}")
        frames.append(load_variable(fpath, var))

    data = pd.concat(frames, axis=1, join="inner").sort_index()
    data = data.interpolate(method="time", limit_direction="both")
    data = data.dropna()
    data = remove_outliers(data, list(FILES.keys()))
    data = add_future_categories_by_calendar_day(data)
    data = data.dropna(subset=["CAT_24h", "CAT_48h", "CAT_72h"])

    raw = data.reset_index().rename(columns={"index": "date"})
    raw.to_csv(OUT_RAW, index=False)

    scaler = StandardScaler()
    X = scaler.fit_transform(data[list(FILES.keys())].astype(float).values)
    scaled = pd.DataFrame(X, columns=list(FILES.keys()), index=data.index)
    final = pd.concat([scaled, data[["CAT", "CAT_24h", "CAT_48h", "CAT_72h"]]], axis=1)
    final = final.reset_index().rename(columns={"index": "date"})
    final.to_csv(OUT_SCALED, index=False)

    np.save(
        OUT_SCALER,
        {"mean": scaler.mean_, "std": scaler.scale_, "features": list(FILES.keys())},
        allow_pickle=True,
    )
    return raw


# =========================
# Bloque 2: Carga robusta raw
# =========================
def load_dataset(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=None, engine="python")

    if raw.shape[1] == 1:
        header = raw.columns[0]
        s = pd.Series([header] + raw.iloc[:, 0].astype(str).tolist())
        parts = s.str.extract(
            r'^(\d{4}-\d{2}-\d{2})\s*,?\s*([\-\\d\\.]+)\s*,?\s*([\-\\d\\.]+)\s*,?\s*([\-\\d\\.]+)\s*,?\s*([\-\\d\\.]+)\s*,?\s*([\-\\d\\.]+)\s*,?\s*([\-\\d\\.]+)\s*,?\s*([\-\\d\\.]+)\s*,?\s*([A-Za-záéíóúñÁÉÍÓÚ]+)\s*$'
        )
        parts.columns = ["date", "PM25", "PM10", "temp", "hum", "pres", "wind", "winddir", "CAT"]
        df = parts.dropna(subset=["date"]).copy()
    else:
        df = raw.copy()
        df.columns = [c.strip() for c in df.columns]

    required = ["date", "PM25", "PM10", "temp", "hum", "pres", "wind", "winddir", "CAT"]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)

    for c in BASE_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["CAT"] = df["CAT"].astype(str).str.strip().str.lower()
    return df


# =========================
# Bloque 3: Feature engineering
# =========================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["dayofyear"] = out["date"].dt.dayofyear
    out["dayofweek"] = out["date"].dt.dayofweek
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    out["winter"] = out["month"].isin([6, 7, 8]).astype(int)
    out["autumn"] = out["month"].isin([3, 4, 5]).astype(int)
    out["spring"] = out["month"].isin([9, 10, 11]).astype(int)
    out["summer"] = out["month"].isin([12, 1, 2]).astype(int)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    return out


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["critical_now"] = out["CAT"].isin(CRITICAL_CATS).astype(int)
    out[f"critical_t_plus_{HORIZON_DAYS}"] = out["critical_now"].shift(-HORIZON_DAYS)
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    blocks = []

    for lag in LAGS:
        temp = out[BASE_NUMERIC].shift(lag)
        temp.columns = [f"{c}_lag{lag}" for c in BASE_NUMERIC]
        blocks.append(temp)

    for w in ROLL_WINDOWS:
        mean_b = out[BASE_NUMERIC].rolling(w).mean().shift(1)
        mean_b.columns = [f"{c}_mean{w}" for c in BASE_NUMERIC]
        max_b = out[BASE_NUMERIC].rolling(w).max().shift(1)
        max_b.columns = [f"{c}_max{w}" for c in BASE_NUMERIC]
        min_b = out[BASE_NUMERIC].rolling(w).min().shift(1)
        min_b.columns = [f"{c}_min{w}" for c in BASE_NUMERIC]
        std_b = out[BASE_NUMERIC].rolling(w).std().shift(1)
        std_b.columns = [f"{c}_std{w}" for c in BASE_NUMERIC]
        blocks.extend([mean_b, max_b, min_b, std_b])

    diff1 = out[BASE_NUMERIC].diff(1)
    diff1.columns = [f"{c}_diff1" for c in BASE_NUMERIC]
    diff2 = out[BASE_NUMERIC].diff(2)
    diff2.columns = [f"{c}_diff2" for c in BASE_NUMERIC]
    blocks.extend([diff1, diff2])

    prev_cat = pd.DataFrame(index=out.index)
    for lag in [1, 2, 3]:
        prev_cat[f"CAT_lag{lag}"] = out["CAT"].shift(lag)
        prev_cat[f"critical_lag{lag}"] = out["critical_now"].shift(lag)
    blocks.append(prev_cat)

    out = pd.concat([out] + blocks, axis=1)
    return out


def make_preprocessor(X: pd.DataFrame):
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols),
    ])
    return preprocessor, numeric_cols, categorical_cols


def select_threshold(y_true, y_prob, recall_target=0.80):
    grid = np.linspace(0.01, 0.99, 199)
    rows = []
    for thr in grid:
        pred = (y_prob >= thr).astype(int)
        rows.append({
            "threshold": thr,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "balanced_acc": balanced_accuracy_score(y_true, pred),
        })
    cand = pd.DataFrame(rows)
    feasible = cand[cand["recall"] >= recall_target].copy()
    if len(feasible) > 0:
        best = feasible.sort_values(["precision", "f1", "balanced_acc"], ascending=False).iloc[0]
    else:
        best = cand.sort_values(["f1", "balanced_acc", "recall"], ascending=False).iloc[0]
    return best.to_dict(), cand


def evaluate_models(X: pd.DataFrame, y: pd.Series, dates: pd.Series):
    models = {
        "logit_balanced": LogisticRegression(
            max_iter=800,
            class_weight="balanced",
            solver="liblinear"
        ),
        "hgb_calibrated": HistGradientBoostingClassifier(
            max_depth=5,
            learning_rate=0.05,
            max_iter=350,
            min_samples_leaf=20,
            random_state=42
        ),
    }

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    fold_rows, threshold_rows, oof_rows = [], [], []

    for model_name, base_model in models.items():
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            d_test = dates.iloc[test_idx]

            preprocessor, _, _ = make_preprocessor(X_train)
            pipe = Pipeline([("prep", preprocessor), ("model", clone(base_model))])

            estimator = (
                CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
                if model_name == "hgb_calibrated"
                else pipe
            )

            estimator.fit(X_train, y_train)
            prob = estimator.predict_proba(X_test)[:, 1]
            best_thr, _ = select_threshold(y_test, prob, recall_target=RECALL_TARGET)
            pred = (prob >= best_thr["threshold"]).astype(int)

            fold_rows.append({
                "model": model_name,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "event_rate_test": float(y_test.mean()),
                "roc_auc": roc_auc_score(y_test, prob) if y_test.nunique() > 1 else np.nan,
                "pr_auc": average_precision_score(y_test, prob) if y_test.nunique() > 1 else np.nan,
                "brier": brier_score_loss(y_test, prob),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "balanced_acc": balanced_accuracy_score(y_test, pred),
                "threshold": best_thr["threshold"],
            })

            threshold_rows.append({"model": model_name, "fold": fold, **best_thr})

            oof_rows.append(pd.DataFrame({
                "date": d_test.values,
                "model": model_name,
                "y_true": y_test.values,
                "y_prob": prob,
                "y_pred": pred,
                "fold": fold
            }))

    fold_metrics = pd.DataFrame(fold_rows)
    thresholds = pd.DataFrame(threshold_rows)
    oof_pred = pd.concat(oof_rows, ignore_index=True)

    summary = fold_metrics.groupby("model", as_index=False).agg({
        "roc_auc": ["mean", "std"],
        "pr_auc": ["mean", "std"],
        "brier": ["mean", "std"],
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
        "f1": ["mean", "std"],
        "balanced_acc": ["mean", "std"],
        "threshold": ["mean", "std"],
        "event_rate_test": "mean",
    })

    summary.columns = [
        "_".join([c for c in col if c]).strip("_")
        for col in summary.columns
    ]
    summary = summary.rename(columns={"model_": "model"})
    summary = summary.sort_values(
        ["recall_mean", "pr_auc_mean", "precision_mean"],
        ascending=False
    )
    return summary, fold_metrics, thresholds, oof_pred


def fit_final_model(X: pd.DataFrame, y: pd.Series, chosen_model: str):
    preprocessor, _, _ = make_preprocessor(X)

    if chosen_model == "hgb_calibrated":
        base = HistGradientBoostingClassifier(
            max_depth=5,
            learning_rate=0.05,
            max_iter=350,
            min_samples_leaf=20,
            random_state=42
        )
        pipe = Pipeline([("prep", preprocessor), ("model", base)])
        estimator = CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
        estimator.fit(X, y)
        prob = estimator.predict_proba(X)[:, 1]
        best_thr, _ = select_threshold(y, prob, recall_target=RECALL_TARGET)
        feature_importance = None
    else:
        base = LogisticRegression(
            max_iter=800,
            class_weight="balanced",
            solver="liblinear"
        )
        estimator = Pipeline([("prep", preprocessor), ("model", base)])
        estimator.fit(X, y)
        prob = estimator.predict_proba(X)[:, 1]
        best_thr, _ = select_threshold(y, prob, recall_target=RECALL_TARGET)
        feature_names = estimator.named_steps["prep"].get_feature_names_out()
        coefs = estimator.named_steps["model"].coef_.ravel()
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "coef": coefs,
            "abs_coef": np.abs(coefs)
        }).sort_values("abs_coef", ascending=False)

    pred = (prob >= best_thr["threshold"]).astype(int)
    final_predictions = pd.DataFrame({
        "date": X.index,
        "y_true": y.values,
        "y_prob": prob,
        "y_pred": pred
    })
    return estimator, best_thr, final_predictions, feature_importance


def save_operational_artifacts(
    estimator,
    best_model: str,
    final_thr: dict,
    feature_cols: list,
    X: pd.DataFrame,
    y: pd.Series,
    coverage_ratio: float
):
    dump(estimator, OUTPUT_DIR / "final_model_48h.joblib")

    with open(OUTPUT_DIR / "final_model_threshold.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": best_model,
            "threshold": float(final_thr["threshold"]),
            "precision": float(final_thr["precision"]),
            "recall": float(final_thr["recall"]),
            "f1": float(final_thr["f1"]),
            "balanced_acc": float(final_thr["balanced_acc"]),
            "horizon_days": HORIZON_DAYS,
            "recall_target": RECALL_TARGET
        }, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "final_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "selected_model": best_model,
            "horizon_days": HORIZON_DAYS,
            "recall_target": RECALL_TARGET,
            "n_rows_model": int(len(X)),
            "n_features": int(len(feature_cols)),
            "feature_columns": feature_cols,
            "event_rate_target": float(y.mean()),
            "date_start": str(X.index.min().date()),
            "date_end": str(X.index.max().date()),
            "coverage_ratio_observed_vs_full_range": float(coverage_ratio)
        }, f, ensure_ascii=False, indent=2)


def run_modeling(df: pd.DataFrame):
    df = add_time_features(df)
    df = add_target(df)
    df = add_lag_features(df)

    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    coverage_ratio = len(df) / len(full_range)

    target_col = f"critical_t_plus_{HORIZON_DAYS}"
    model_df = df.dropna(subset=[target_col]).copy()

    feature_cols = [c for c in model_df.columns if c not in ["date", "CAT", "critical_now", target_col]]
    X = model_df[feature_cols].copy()
    X.index = model_df["date"]
    y = model_df[target_col].astype(int)
    dates = model_df["date"].copy()

    summary, fold_metrics, thresholds, oof_pred = evaluate_models(X, y, dates)
    best_model = summary.iloc[0]["model"]
    estimator, final_thr, final_predictions, feat_imp = fit_final_model(X, y, best_model)

    summary.to_csv(OUTPUT_DIR / "cv_model_summary.csv", index=False)
    fold_metrics.to_csv(OUTPUT_DIR / "cv_fold_metrics.csv", index=False)
    thresholds.to_csv(OUTPUT_DIR / "cv_thresholds.csv", index=False)
    oof_pred.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    final_predictions.to_csv(OUTPUT_DIR / "daily_predictions_48h.csv", index=False)
    model_df.to_csv(OUTPUT_DIR / "model_ready_dataset.csv", index=False)

    if feat_imp is not None:
        feat_imp.to_csv(OUTPUT_DIR / "feature_importance_logit.csv", index=False)

    save_operational_artifacts(
        estimator=estimator,
        best_model=best_model,
        final_thr=final_thr,
        feature_cols=feature_cols,
        X=X,
        y=y,
        coverage_ratio=coverage_ratio
    )

    with open(OUTPUT_DIR / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write("Resumen de ejecución\n")
        f.write("====================\n")
        f.write(f"Filas observadas: {len(df)}\n")
        f.write(f"Cobertura temporal: {coverage_ratio:.4f}\n")
        f.write(f"Evento crítico actual: {df['critical_now'].mean():.4f}\n")
        f.write(f"Evento crítico a 48h: {y.mean():.4f}\n")
        f.write(f"Modelo seleccionado: {best_model}\n")
        f.write(f"Umbral final sugerido: {final_thr['threshold']:.3f}\n")
        f.write(f"Recall objetivo: {RECALL_TARGET:.2f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-raw",
        action="store_true",
        help="Construye dataset_model_clean_raw.csv desde archivos fuente"
    )
    parser.add_argument(
        "--input-raw",
        default=DEFAULT_RAW_INPUT,
        help="Ruta a dataset_model_clean_raw.csv si no se usa --build-raw"
    )
    args = parser.parse_args()

    if args.build_raw:
        df_raw = build_raw_from_source_files()
    else:
        input_path = Path(args.input_raw)
        if not input_path.exists():
            alt = OUT_RAW
            if alt.exists():
                input_path = alt
            else:
                raise FileNotFoundError(f"No se encontró raw dataset: {args.input_raw}")
        df_raw = load_dataset(str(input_path))

    run_modeling(df_raw)
    print("Proceso completo finalizado.")


if __name__ == "__main__":
    main()
