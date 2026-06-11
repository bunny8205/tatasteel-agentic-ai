"""Small local LLM wrapper for explanatory text."""

from __future__ import annotations

from dataclasses import dataclass

from .config import LOCAL_LLM_MODEL_ID, USE_LOCAL_LLM


@dataclass
class LocalLLM:
    model_id: str = LOCAL_LLM_MODEL_ID
    enabled: bool = USE_LOCAL_LLM
    tokenizer: object | None = None
    model: object | None = None
    load_error: str = ""

    def load(self) -> "LocalLLM":
        if not self.enabled:
            return self
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                device_map=None,
            )
            self.model.to("cpu")
            self.model.eval()
        except Exception as exc:
            self.load_error = str(exc)
            self.tokenizer = None
            self.model = None
        return self

    @property
    def available(self) -> bool:
        return self.tokenizer is not None and self.model is not None

    def explain(self, facts: dict, max_new_tokens: int = 110) -> str:
        fallback = self._fallback_explanation(facts)
        if not self.available:
            return fallback

        prompt = (
            "You are a steel plant maintenance engineer. Write 3 concise numbered points. "
            "Explain the locked risk decision. Do not change asset ID, risk, priority, or RUL.\n\n"
            f"Facts: {facts}\n\nExplanation:"
        )

        try:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900).to("cpu")
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            explanation = text.split("Explanation:")[-1].strip()
            return explanation or fallback
        except Exception:
            return fallback

    def _fallback_explanation(self, facts: dict) -> str:
        asset = facts.get("asset_id", facts.get("top_asset", "the asset"))
        risk = facts.get("risk_level", "UNKNOWN")
        priority = facts.get("priority", "UNKNOWN")
        rul = facts.get("rul_days", "unknown")
        hybrid = facts.get("hybrid_failure_risk", "unknown")
        rule_score = facts.get("operational_rule_score", "unknown")
        return (
            f"1. {asset} is classified as {risk} with priority {priority} based on locked hybrid scoring.\n"
            f"2. The hybrid failure risk is {hybrid} and the operational rule score is {rule_score}, so the decision is traceable.\n"
            f"3. Estimated RUL is {rul} days, so the maintenance plan should prioritize safe intervention and spare readiness."
        )
