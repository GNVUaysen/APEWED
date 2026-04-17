# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import json
import warnings
import pandas as pd
from joblib import load

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

REQUIRED_COLUMNS = ["PM25", "temp", "hum", "wind"]


def load_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) != 1:
        raise ValueError("Input file must contain exactly one row.")

    for c in REQUIRED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Input file contains non-numeric or missing values in required columns.")

    return df[REQUIRED_COLUMNS].copy()


def find_threshold_file(model_path: Path) -> Path:
    candidates = [
        model_path.with_name("final_model_operational_48h_threshold.json"),
        model_path.with_name(f"{model_path.stem}_threshold.json"),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Threshold file not found. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def load_threshold(model_path: Path) -> float:
    threshold_path = find_threshold_file(model_path)

    with open(threshold_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    return float(info["threshold"])


def main():
    parser = argparse.ArgumentParser(
        description="48-hour prediction of critical air-pollution events"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV file with one-row environmental snapshot"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to final_model_operational_48h.joblib"
    )
    parser.add_argument(
        "--output",
        default="prediction_output.csv",
        help="Output CSV path"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)
    output_path = Path(args.output)

    df = load_input(input_path)
    model = load(model_path)
    threshold = load_threshold(model_path)

    prob = float(model.predict_proba(df)[0, 1])
    alert = int(prob >= threshold)

    out = df.copy()
    out["predicted_probability_48h"] = prob
    out["decision_threshold"] = threshold
    out["critical_alert_48h"] = alert

    out.to_csv(output_path, index=False)

    print(f"Predicted probability (48h): {prob:.4f}")
    print(f"Decision threshold: {threshold:.4f}")
    print(f"Critical alert (0/1): {alert}")
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
