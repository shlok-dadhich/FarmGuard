from __future__ import annotations
import os
from src.ai.provider import AIProvider
from src.ai.fallback import FallbackProvider
class OpenAICompatibleProvider(AIProvider):
    def __init__(self, base_url="", api_key="", model="gpt-4o-mini"):
        self.base_url, self.api_key, self.model = base_url, api_key, model
    def explain(self, ctx: dict) -> str:
        if not self.api_key:
            raise RuntimeError("WHAT: AI API key missing\nWHY: AI_API_KEY env not set\nHOW: export AI_API_KEY or use fallback provider")
        try:
            from openai import OpenAI
            cl = OpenAI(base_url=self.base_url or None, api_key=self.api_key)
            from src.ai.prompts import SYSTEM, build_prompt
            r = cl.chat.completions.create(model=self.model, messages=[{"role":"system","content":SYSTEM},{"role":"user","content":build_prompt(ctx)}], max_tokens=600)
            return r.choices[0].message.content
        except Exception as e:
            return FallbackProvider().explain(ctx) + f"\n\n(fallback used: {e})"
def get_provider(cfg: dict) -> AIProvider:
    name = cfg.get("provider", "fallback")
    if name == "openai_compatible":
        o = cfg.get("openai_compatible", {})
        return OpenAICompatibleProvider(o.get("base_url",""), os.environ.get(o.get("api_key_env","AI_API_KEY"),""), o.get("model","gpt-4o-mini"))
    if name == "mock":
        from src.ai.fallback import MockProvider
        return MockProvider()
    return FallbackProvider()
