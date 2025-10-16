"""AXON Tiny LLM Module

Local-only LLM interface stub (safe to disable).
"""

from ..utils import load_config


class TinyLLM:
    """Stub interface for local LLM integration."""
    
    def __init__(self, enabled=False):
        self.enabled = enabled
        if not enabled:
            print("🤖 TinyLLM disabled (llm_local: false)")
    
    def summarize_error_lens(self, bullets):
        """Summarize error lens bullets into up to 5 lines (deterministic)."""
        if not self.enabled:
            return "LLM disabled - no summary available"
        
        if not bullets:
            return "No insights available"
        
        # Normalize and select top items
        lines = [str(b).strip() for b in bullets if str(b).strip()]
        top = lines[:5]
        return "\n".join(f"- {line}" for line in top) if top else "No insights available"
    
    def suggest_micro_patch(self, analysis):
        """Suggest one micro-patch comment via simple heuristics."""
        if not self.enabled:
            return "LLM disabled - no suggestions available"
        
        text = (analysis or "").lower()
        if "leak" in text or "leakage" in text:
            return "Guard against data leakage in feature pipeline (correct train/validation split)."
        if "overfit" in text or "overfitting" in text:
            return "Add stronger regularization and CV; review feature selection and data splits."
        if "latency" in text or "timeout" in text:
            return "Add retry/backoff, instrument timings, and optimize data fetch paths."
        if "drift" in text:
            return "Implement drift detection (PSI/KS) and schedule periodic retraining."
        if "missing" in text or "null" in text:
            return "Harden data validation; add imputers and pre-checks before training."
        
        return "Add structured logging around failing path and capture inputs for reproducibility."


def main():
    """Main tiny LLM module."""
    config = load_config()
    enabled = config.get('llm_local', False)
    
    llm = TinyLLM(enabled=enabled)
    print(f"🤖 TinyLLM module for {config['project']} (enabled: {enabled})")
    
    if enabled:
        print("⚠️ Local LLM integration ready for implementation")
    else:
        print("✅ TinyLLM safely disabled")


if __name__ == "__main__":
    main()