"""AXON Orchestrator

Runs the full AXON pipeline end-to-end using centralized config.
"""

import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from src.utils import load_config, ensure_dir
from axon.pipeline import run_pipeline


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger("axon.main")


def set_seeds(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)


def main():
    """AXON Orchestrator

    Runs the full AXON pipeline end-to-end using centralized config.
    """

    logger = setup_logging()
    logger.info("Starting guarded AXON pipeline")

    try:
        # Delegate to new guarded pipeline which wraps existing steps
        from src.utils import load_config
        cfg = load_config()
        result = run_pipeline(cfg)
        if not result.get("train_ok", False):
            logger.warning(f"Pipeline finished with abort at {result.get('failed_step')}: {result.get('reason')}")
        else:
            logger.info("✅ AXON pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

    config = load_config()
    logger.info(f"Starting AXON pipeline: {config.get('project', 'AXON')}")

    # Telegram notifier
    from src.brains.notifier import TelegramNotifier
    notifier = TelegramNotifier()
    
    # Pipeline start notification
    notifier.send(f"🚀 AXON pipeline iniciado - {config.get('project', 'AXON')}", "pipeline_start")

    # Ensure output dirs
    for p in [
        "outputs/artifacts",
        "outputs/metrics",
        "outputs/ledgers",
        "outputs/reports",
        "outputs/figures",
        "knowledge",
    ]:
        ensure_dir(p)

    try:
        # 1) Triage (queue processing)
        logger.info("Step 1/8: Triage queue")
        from src.brains import triage as triage_mod
        triage_mod.main()

        # 2) Dataset (raw -> processed splits)
        logger.info("Step 2/8: Prepare dataset")
        from src import dataset as dataset_mod
        dataset_mod.main()

        # 3) Feature engineering (processed splits -> features/labels)
        logger.info("Step 3/8: Create features")
        from src import features as features_mod
        features_mod.main()

        # 4) Train models (artifacts + validation metrics)
        logger.info("Step 4/8: Train models")
        from src import train as train_mod
        train_mod.main()

        # 5) Backtest (use artifacts on test set)
        logger.info("Step 5/8: Run backtests")
        from src import backtest as backtest_mod
        backtest_mod.main()

        # 6) Error Lens (regime analysis)
        logger.info("Step 6/8: Error Lens analysis")
        from src.brains import error_lens as error_lens_mod
        error_lens_mod.main()

        # 7) Decision (promotion gate)
        logger.info("Step 7/8: Promotion decision")
        from src.brains import decision as decision_mod
        decision_mod.main()

        # Pós-gate: exportar model bundle se aprovado
        try:
            from src.export import export_model_bundle
            export_model_bundle(config)
        except Exception as ex:
            logger.warning(f"Export bundle skipped/failed: {ex}")

        # 8) Report + Memory
        logger.info("Step 8/8: Reports and memory")
        from src import report as report_mod
        report_mod.main()

        from src.brains import memory as memory_mod
        memory_mod.main()

        logger.info("✅ AXON pipeline completed successfully")
        # Pipeline end notification
        try:
            notifier.send("✅ AXON pipeline finalizado com sucesso", "pipeline_end")
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        try:
            notifier.send(f"❌ AXON pipeline falhou: {e}", "errors")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()

