"""AXON Memory Module

Knowledge accumulation and playbook updates.
"""

import json
from datetime import datetime
from pathlib import Path

from ..utils import load_config, ensure_dir
from .tiny_llm import TinyLLM


def main():
    """Main memory update pipeline."""
    config = load_config()
    print(f"[*] Updating memory for {config['project']}...")

    # Enable local LLM if configured
    llm_enabled = config.get('llm_local', False)
    llm = TinyLLM(enabled=llm_enabled) if llm_enabled else None

    # Ensure knowledge directory exists
    ensure_dir("knowledge")

    findings_path = Path("knowledge/findings.jsonl")
    triage_report = Path("knowledge/triage_report.json")
    decision_path = Path("outputs/metrics/DECISION.json")
    error_lens_data = Path("outputs/reports/error_lens_data.json")

    record = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'project': config.get('project', 'AXON'),
    }

    # Attach decision
    if decision_path.exists():
        try:
            record['decision'] = json.loads(decision_path.read_text())
        except Exception:
            record['decision'] = {}

    # Attach error lens insights
    if error_lens_data.exists():
        try:
            el = json.loads(error_lens_data.read_text())
            record['insights'] = el.get('insights', [])
            record['patterns'] = el.get('patterns', {})
        except Exception:
            pass

    # Attach triage summary if exists
    if triage_report.exists():
        try:
            record['triage'] = json.loads(triage_report.read_text())
        except Exception:
            pass

    # Optional: LLM summarization and micro-patch suggestion
    if llm_enabled and llm:
        try:
            if record.get('insights'):
                record['llm_summary'] = llm.summarize_error_lens(record['insights'])
        except Exception:
            pass
        try:
            analysis_parts = []
            triage = record.get('triage')
            if isinstance(triage, dict):
                summary = triage.get('summary') or triage.get('notes') or ""
                if summary:
                    analysis_parts.append(str(summary))
            patterns = record.get('patterns')
            if isinstance(patterns, dict) and patterns:
                analysis_parts.append(" ".join(map(str, patterns.keys())))
            analysis_text = " ".join(analysis_parts).strip()
            if analysis_text:
                record['llm_suggestion'] = llm.suggest_micro_patch(analysis_text)
        except Exception:
            pass

    # Append line-delimited JSON
    with findings_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Knowledge updated: {findings_path}")


if __name__ == "__main__":
    main()
