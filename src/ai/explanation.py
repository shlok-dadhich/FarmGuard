from src.ai.client import get_provider
def explain_prediction(ctx: dict, cfg: dict) -> str:
    return get_provider(cfg).explain(ctx)
