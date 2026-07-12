import argparse
import json

import joblib
import pandas as pd
from tensorflow import keras


def load_input_data(input_path: str) -> pd.DataFrame:
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        return pd.DataFrame([payload])

    raise ValueError("Input JSON must be an object or a list of objects.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run churn prediction using a saved ANN model.")
    parser.add_argument("--model", required=True, help="Path to saved Keras model (.keras or .h5)")
    parser.add_argument("--input", required=True, help="Path to JSON input file")
    parser.add_argument(
        "--preprocessor",
        default=None,
        help="Optional path to a fitted preprocessing object (.pkl/.joblib)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for churn (default: 0.5)",
    )
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    data = load_input_data(args.input)

    model_input = data
    if args.preprocessor:
        preprocessor = joblib.load(args.preprocessor)
        model_input = preprocessor.transform(data)

    probabilities = model.predict(model_input, verbose=0).flatten()

    results = []
    for prob in probabilities:
        results.append(
            {
                "churn_probability": float(prob),
                "predicted_label": int(prob >= args.threshold),
            }
        )

    print(json.dumps(results if len(results) > 1 else results[0], indent=2))


if __name__ == "__main__":
    main()
