"""AXON Training Module

Model training and validation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import load_config, ensure_dir
from .models import ModelRegistry, train_model, save_model, get_feature_importance


logger = logging.getLogger(__name__)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    """Main training pipeline."""
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    logger.info(f"Training models for {config.get('project', 'AXON')}...")

    # Ensure output directories exist
    ensure_dir("outputs/artifacts")
    ensure_dir("outputs/metrics")

    # Load processed features
    train_path = Path("data/processed/train_features.parquet")
    val_path = Path("data/processed/validation_features.parquet")
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Processed features not found. Run feature engineering first.")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    target_col = config.get('target', 'y')
    feature_cols = [c for c in train_df.columns if c not in ['timestamp', target_col]]
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    logger.info(f"Samples: train={len(X_train):,}, val={len(X_val):,}, features={len(feature_cols)}")

    # Initialize registry and iterate models
    registry = ModelRegistry(config)
    models_to_train = config.get('models', {}).get('train', ['lightgbm', 'randomforest'])

    for model_name in models_to_train:
        model_name = model_name.lower()
        logger.info(f"Training {model_name} ...")
        try:
            model = registry.get_model(model_name)
            trained_model, metrics = train_model(
                model, X_train, y_train, X_val, y_val, model_name, config
            )

            # Save model + metadata
            model_path = save_model(trained_model, model_name, metrics, feature_cols, config)

            # Save validation metrics file
            ts = _timestamp()
            metrics_payload = {
                'type': 'validation',
                'timestamp': ts,
                'model': model_name,
                'metrics': metrics,
                'features': feature_cols,
                'artifact': model_path,
            }
            metrics_file = f"outputs/metrics/VAL_{model_name}_{ts}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics_payload, f, indent=2)
            logger.info(f"Validation metrics saved: {metrics_file}")

            # Save feature importance if available
            importance_df = get_feature_importance(trained_model, feature_cols, model_name)
            if not importance_df.empty:
                imp_path = f"outputs/artifacts/{model_name}_{ts}_feature_importance.csv"
                importance_df.to_csv(imp_path, index=False)
                logger.info(f"Feature importance saved: {imp_path}")

        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            continue

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
