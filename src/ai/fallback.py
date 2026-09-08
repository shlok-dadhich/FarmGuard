from __future__ import annotations
from src.ai.provider import AIProvider
DISCLAIMER = "Assistive explanation only; not agronomic certainty. Classifier is source of truth."
class FallbackProvider(AIProvider):
    def explain(self, ctx: dict) -> str:
        top = ", ".join(f"{l} ({p:.1%})" for l, p in ctx.get("top_k", [])[:3])
        return (f"Model ({ctx.get('model','?')}) predicts **{ctx.get('label','?')}** at {ctx.get('confidence',0):.1%} for {ctx.get('crop','?')} crop.\n" f"Top candidates: {top}.\n" f"Visual evidence: heatmap focuses on {ctx.get('xai_note','highlighted regions')}.\n" f"Uncertainty: {'low confidence - verify with expert' if ctx.get('confidence',1) < 0.6 else 'moderate-high confidence, still verify'}.\n" f"Next step: {ctx.get('next_step','inspect lesion close-up and compare with reference images')}.\n\n_{DISCLAIMER}_")
class MockProvider(FallbackProvider): pass
