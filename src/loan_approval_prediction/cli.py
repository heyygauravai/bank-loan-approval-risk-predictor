"""Small command-line entry point for training and local inference."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from loan_approval_prediction.inference import load_bundle, predict_one
from loan_approval_prediction.training import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    train,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="loan-approval")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train and evaluate from the Kaggle CSV")
    training.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    training.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    training.add_argument("--no-mlflow", action="store_true")
    predicting = commands.add_parser("predict", help="Predict from a JSON feature object")
    predicting.add_argument("--input", type=Path, required=True)
    predicting.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        if args.command == "train":
            report = train(args.data, args.model, track_mlflow=not args.no_mlflow)
            print(json.dumps({"model": report["selected_model"], "test_metrics": report["test_metrics"]}, indent=2))
        else:
            features = json.loads(args.input.read_text(encoding="utf-8"))
            print(json.dumps(predict_one(load_bundle(args.model), features), indent=2))
    except (FileNotFoundError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
