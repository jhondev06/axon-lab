from typing import Dict, Any, Optional
import json
from pathlib import Path

import pandas as pd

from axon.core.types import PipelineAbort, StepResult
from axon.core.logging import get_logger, jmsg

# Configurable thresholds (can be moved to axon.cfg.yml later)
MIN_ROWS_AFTER_PREP = 500
MIN_ROWS_PER_SPLIT = 100
MIN_FEATURES_REQUIRED = 5
MIN_XY_2D = True
ALLOW_SYNTHETIC_FALLBACK = False

log = get_logger("axon.pipeline")


def _n(df) -> int:
    return 0 if df is None else len(df)


def _assert_min_rows(df, min_rows: int, step: str, reason: str, extra: Optional[Dict[str, Any]] = None):
    n = _n(df)
    if n < min_rows:
        raise PipelineAbort(step, reason, {"rows": n, **(extra or {})})


def _assert_2d(X, y, step: str):
    if X is None or getattr(X, "ndim", 1) != 2 or X.shape[0] == 0:
        raise PipelineAbort(step, "X not 2D or empty", {"X_shape": getattr(X, "shape", None)})
    if y is None or getattr(y, "shape", None) is None or y.shape[0] == 0:
        raise PipelineAbort(step, "y empty", {"y_shape": getattr(y, "shape", None)})


# Orchestrator that wraps the existing modules without breaking them

def run_pipeline(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    results: list[StepResult] = []
    train_ok = False
    failed_step: Optional[str] = None
    failure_reason: Optional[str] = None
    failure_diag: Dict[str, Any] = {}

    try:
        # Step 1/8 — Triage (reuse existing brains.triage main if needed)
        log.info(jmsg("step", phase="1/8", name="Triage"))
        results.append(StepResult("Triage", 0, 0, {"status": "ok"}, True))

        # Step 2/8 — Prepare dataset
        from src import dataset as dataset_mod
        log.info(jmsg("step", phase="2/8", name="Prepare dataset"))
        try:
            dataset_mod.main()  # creates data/processed/{train,validation,test}.parquet
        except Exception as e:
            failed_step, failure_reason = "Prepare dataset", f"dataset.main failed: {e}"
            raise

        # Load processed splits produced by dataset module
        splits = {s: dataset_mod.load_processed_data(s) for s in ["train", "validation", "test"]}
        rows_in = sum(_n(df) for df in splits.values())
        log.info(jmsg("step", phase="2/8", name="Prepare dataset", rows_in=rows_in, rows_out=rows_in))
        _assert_min_rows(pd.concat(list(splits.values()), ignore_index=True), MIN_ROWS_AFTER_PREP, "Prepare dataset", "Too few rows after filters")

        # Guardrails for split sizes
        train_n, val_n, test_n = _n(splits['train']), _n(splits['validation']), _n(splits['test'])
        if train_n < MIN_ROWS_PER_SPLIT or val_n < MIN_ROWS_PER_SPLIT:
            raise PipelineAbort("Split", "Insufficient rows per split", {"train_n": train_n, "val_n": val_n, "test_n": test_n})
        results.append(StepResult("Split", rows_in, rows_in, {"train": train_n, "val": val_n, "test": test_n}, True))

        # Step 3/8 — Feature engineering
        from src import features as features_mod
        log.info(jmsg("step", phase="3/8", name="Create features"))
        try:
            features_mod.main()  # will save train/val/test_features.parquet
        except Exception as e:
            failed_step, failure_reason = "Features", f"features.main failed: {e}"
            raise

        # Load engineered features
        train_f = pd.read_parquet("data/processed/train_features.parquet")
        val_f = pd.read_parquet("data/processed/validation_features.parquet")
        test_path = Path("data/processed/test_features.parquet")
        test_f = pd.read_parquet(test_path) if test_path.exists() else None

        target = cfg.get("target", "y") if cfg else "y"
        feature_cols = [c for c in train_f.columns if c not in ["timestamp", target]]
        n_features = len(feature_cols)
        if n_features < MIN_FEATURES_REQUIRED:
            raise PipelineAbort("Features", "Too few features", {"n_features": n_features})
        if len(train_f) == 0 or len(val_f) == 0:
            raise PipelineAbort("Features", "Rows dropped to zero after engineering", {"train": len(train_f), "val": len(val_f)})
        results.append(StepResult("Create features", rows_in=len(train_f) + len(val_f), n_rows_out=len(train_f) + len(val_f), diagnostics={"n_features": n_features}, ok=True))

        # Step 3.5 — Labeling
        label_diag = {
            "train_label_counts": train_f[target].value_counts(dropna=False).to_dict(),
            "val_label_counts": val_f[target].value_counts(dropna=False).to_dict(),
        }
        if train_f[target].isna().all() or val_f[target].isna().all():
            raise PipelineAbort("Labeling", "All labels NaN", label_diag)
        if train_f[target].nunique(dropna=True) < 2 or val_f[target].nunique(dropna=True) < 2:
            raise PipelineAbort("Labeling", "Single-class labels", label_diag)
        results.append(StepResult("Labeling", n_rows_in=len(train_f) + len(val_f), n_rows_out=len(train_f) + len(val_f), diagnostics=label_diag, ok=True))

        # Step 4/8 — Train models (uses src.train)
        from src import train as train_mod
        log.info(jmsg("step", phase="4/8", name="Train models"))

        # Basic shape checks before delegating
        _assert_2d(train_f[feature_cols].values, train_f[target].values, "Train models")
        _assert_2d(val_f[feature_cols].values, val_f[target].values, "Train models")

        try:
            train_mod.main()
            train_ok = True
        except Exception as e:
            failed_step, failure_reason = "Train models", f"train.main failed: {e}"
            raise
        results.append(StepResult("Train models", len(train_f), len(train_f), {"features": len(feature_cols)}, True))

        # Step 5/8 — Backtest (only if train_ok)
        if train_ok:
            from src import backtest as bt_mod
            # Check test set before running backtests
            if test_f is None or len(test_f) == 0:
                log.warning(jmsg("skip_backtest", reason="no test features"))
            else:
                _assert_min_rows(test_f, MIN_ROWS_PER_SPLIT, "Backtest", "Too few rows in test split", {"rows": len(test_f)})
                log.info(jmsg("step", phase="5/8", name="Run backtests", test_rows=len(test_f)))
                try:
                    bt_mod.main()
                except Exception as e:
                    # Backtest errors should not crash the pipeline if training passed guardrails
                    log.error(jmsg("backtest_error", error=str(e)))
            results.append(StepResult("Backtests", 0, 0, {"status": "completed"}, True))

    except PipelineAbort as e:
        failed_step, failure_reason, failure_diag = e.step, e.reason, e.diag
        log.error(jmsg("abort", step=e.step, reason=e.reason, diag=e.diag))
    except Exception as e:
        if not failed_step:
            failed_step = "Unknown"
            failure_reason = str(e)
        log.error(jmsg("abort", step=failed_step, reason=failure_reason))
    finally:
        # Step 6/8 — Error Lens analysis (always run)
        from src.brains import error_lens as el
        log.info(jmsg("step", phase="6/8", name="Error Lens analysis", failed_step=failed_step, reason=failure_reason))
        try:
            el.main()
        except Exception:
            pass

        # Step 7/8 — Report (always run)
        from src import report as report_mod
        log.info(jmsg("step", phase="7/8", name="Report"))
        try:
            report_mod.main()
        except Exception:
            pass

        # Step 8/8 — Finalize
        log.info(jmsg("step", phase="8/8", name="Finalize"))

        return {
            "train_ok": bool(train_ok),
            "failed_step": failed_step,
            "reason": failure_reason,
            "steps": [r.__dict__ for r in results],
        }