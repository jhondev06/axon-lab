"""AXON Decision Module

Model promotion gating and stability checks.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from ..utils import load_config, ensure_dir


def _collect_bt_metrics() -> Dict[str, Dict[str, Any]]:
    metrics_dir = Path("outputs/metrics")
    results: Dict[str, Dict[str, Any]] = {}
    if not metrics_dir.exists():
        return results
    for f in metrics_dir.glob("BT_*.json"):
        try:
            data = json.loads(f.read_text())
            model = data.get('model')
            if model:
                results[model] = data.get('metrics', {})
        except Exception:
            continue
    return results


def _deploy_to_battle_arena(decision: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Deploy approved model to Battle Arena for live trading.

    Args:
        decision: Decision result from gating
        config: AXON configuration
    """
    try:
        # Import Battle Arena components
        from battle_arena.core.model_loader import ModelLoader
        from battle_arena.core.battle_controller import BattleController
        from battle_arena.config.battle_config import load_battle_config

        # Load Battle Arena configuration
        battle_config = load_battle_config()

        # Override with AXON config where appropriate
        battle_config.update({
            'symbols': config.get('data', {}).get('symbols', ['BTCUSDT']),
            'data_interval': config.get('data', {}).get('interval', '1m'),
        })

        # Load approved model
        model_loader = ModelLoader()
        model_id = decision.get('candidate_id') or decision.get('candidate')
        model_info = model_loader.load_model(model_id)

        if not model_info:
            raise ValueError(f"Could not load model {model_id}")

        # Validate model compatibility
        validation = model_loader.validate_model_compatibility(model_info)
        if not validation['compatible']:
            raise ValueError(f"Model not compatible with Battle Arena: {validation['issues']}")

        print(f"[+] Model {model_id} validated for Battle Arena deployment")

        # Create and start Battle Controller
        controller = BattleController(battle_config, model_info)

        # Add logging callback
        def log_event(data):
            print(f"[BATTLE] {data}")

        controller.add_event_callback('signal_generated', log_event)
        controller.add_event_callback('order_executed', log_event)
        controller.add_event_callback('error', lambda e: print(f"[BATTLE ERROR] {e}"))

        # Start trading (in background)
        if controller.start_trading():
            print(f"[+] Battle Arena started successfully with model {model_id}")
            print("[+] Live trading is now active!")

            # Save deployment info
            deployment_info = {
                'model_id': model_id,
                'model_name': model_info.model_name,
                'deployed_at': decision['timestamp'],
                'battle_config': battle_config,
                'validation': validation
            }

            deploy_path = Path("outputs/battle_deployments")
            deploy_path.mkdir(exist_ok=True)
            deploy_file = deploy_path / f"deployment_{model_id}_{decision['timestamp']}.json"

            import json
            deploy_file.write_text(json.dumps(deployment_info, indent=2, default=str))
            print(f"[+] Deployment info saved to {deploy_file}")

        else:
            raise RuntimeError("Failed to start Battle Controller")

    except Exception as e:
        print(f"[!] Battle Arena deployment failed: {e}")
        raise


def main():
    """Main decision pipeline."""
    config = load_config()
    print(f"[*] Making promotion decision for {config['project']}...")

    # Ensure output directory exists
    ensure_dir("outputs/metrics")

    # Thresholds
    decision_cfg = config.get('intelligence', {}).get('decision', {})
    min_sharpe = float(decision_cfg.get('min_sharpe', 1.0))
    min_win = float(decision_cfg.get('min_win_rate', 0.55))
    max_dd = float(decision_cfg.get('max_drawdown', 0.2))

    # Load backtest metrics
    bt = _collect_bt_metrics()
    if not bt:
        print("No backtest metrics found. Skipping decision.")
        return

    # Rank by total_return (primary) and Sharpe ratio (tiebreaker)
    ranked = sorted(bt.items(), 
                   key=lambda kv: (kv[1].get('total_return', 0.0), kv[1].get('sharpe_ratio', 0.0)), 
                   reverse=True)
    best_model, best_metrics = ranked[0]

    # Resolve artifact info when available
    model_path = best_metrics.get('model_path')
    model_id = None
    if model_path:
        try:
            model_id = Path(model_path).stem
        except Exception:
            model_id = None

    # Apply gates
    decision = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'project': config.get('project', 'AXON'),
        'candidate': best_model,
        'metrics': best_metrics,
        'thresholds': {
            'min_sharpe': min_sharpe,
            'min_win_rate': min_win,
            'max_drawdown': max_dd,
        },
    }
    # Optional richer identifiers (non-breaking additions)
    if model_id:
        decision['candidate_id'] = model_id
    if model_path:
        decision['artifact'] = model_path

    sharpe_ok = best_metrics.get('sharpe_ratio', 0.0) >= min_sharpe
    win_ok = best_metrics.get('win_rate', 0.0) >= min_win
    dd_ok = abs(best_metrics.get('max_drawdown', 0.0)) <= max_dd
    decision['pass'] = bool(sharpe_ok and win_ok and dd_ok)
    decision['reason'] = {
        'sharpe_ok': sharpe_ok,
        'win_rate_ok': win_ok,
        'max_drawdown_ok': dd_ok,
    }

    # Persist decision
    out_path = Path("outputs/metrics/DECISION.json")
    out_path.write_text(json.dumps(decision, indent=2))
    print(f"Decision written to {out_path}")

    # Auto-deploy to Battle Arena if model passes
    if decision['pass']:
        try:
            print(f"[*] Model {best_model} passed gates - deploying to Battle Arena...")
            _deploy_to_battle_arena(decision, config)
        except Exception as e:
            print(f"[!] Failed to deploy to Battle Arena: {e}")

    # Telegram notify
    try:
        from .notifier import TelegramNotifier
        notifier = TelegramNotifier()
        # Build richer message with metrics
        total_ret = best_metrics.get('total_return')
        sharpe = best_metrics.get('sharpe_ratio')
        win = best_metrics.get('win_rate')
        mdd = best_metrics.get('max_drawdown')
        pf = best_metrics.get('profit_factor')
        trades = best_metrics.get('num_trades')
        final_cap = best_metrics.get('final_capital')
        tp = best_metrics.get('test_period') or {}
        tp_start = tp.get('start')
        tp_end = tp.get('end')
        tp_bars = tp.get('num_bars')
        model_label = model_id or best_model
        artifact_name = os.path.basename(model_path) if model_path else None

        # Safe string formats
        sharpe_s = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else "n/a"
        win_s = f"{win:.2%}" if isinstance(win, (int, float)) else "n/a"
        mdd_s = f"{mdd:.2%}" if isinstance(mdd, (int, float)) else "n/a"
        ret_s = f"{total_ret:.2%}" if isinstance(total_ret, (int, float)) else "n/a"
        pf_s = f"{pf:.2f}" if isinstance(pf, (int, float)) else "n/a"
        trades_s = f"{trades}" if trades is not None else "n/a"

        if decision['pass']:
            lines = [
                "🟢 Promotion PASS",
                f"Modelo: {model_label}",
                f"Sharpe: {sharpe_s} • Win: {win_s} • MDD: {mdd_s}",
                f"Retorno: {ret_s} • PF: {pf_s} • Trades: {trades_s}",
            ]
            if final_cap is not None:
                try:
                    lines.append(f"Capital Final: {final_cap:,.2f}")
                except Exception:
                    lines.append(f"Capital Final: {final_cap}")
            if tp_start and tp_end and tp_bars is not None:
                lines.append(f"Teste: {tp_start} → {tp_end} ({tp_bars} barras)")
            if artifact_name:
                lines.append(f"Artifact: {artifact_name}")
            msg = "\n".join(lines)
            notifier.send(msg, "model_promotion")
        else:
            msg = (f"🔴 Promotion FAIL: {model_label}\n"
                   f"Sharpe: {sharpe_s} • Win: {win_s} • MDD: {mdd_s}\n"
                   f"Thresholds → Sharpe≥{min_sharpe:.2f}, Win≥{min_win:.2%}, MDD≤{max_dd:.2%}")
            notifier.send(msg, "gate_failures")
    except Exception:
        pass


if __name__ == "__main__":
    main()
