# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import json
import pandas as pd
from joblib import load


REQUIRED_COLUMNS = ["PM25", "PM10", "temp", "hum", "pres", "wind", "winddir"]


def load_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) != 1:
        raise ValueError("Input file must contain exactly one row.")

    return df.copy()


def main():
    parser = argparse.ArgumentParser(description="48-hour critical air-pollution prediction")
    parser.add_argument("--input", required=True, help="CSV file with one-row environmental snapshot")
    parser.add_argument("--model", required=True, help="Path to final_model_48h.joblib")
    parser.add_argument("--threshold", required=True, help="Path to final_model_threshold.json")
    parser.add_argument("--output", default="prediction_output.csv", help="Output CSV path")
    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)
    threshold_path = Path(args.threshold)
    output_path = Path(args.output)

    df = load_input(input_path)
    model = load(model_path)

    with open(threshold_path, "r", encoding="utf-8") as f:
        thr_info = json.load(f)

    threshold = float(thr_info["threshold"])
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
