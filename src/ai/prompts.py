SYSTEM = "You explain plant-disease classifier outputs. Never diagnose directly; only explain the model's prediction, confidence, XAI evidence, uncertainty, limits, next steps." 
def build_prompt(ctx: dict) -> str:
    return f"Crop={ctx.get('crop')} Prediction={ctx.get('label')} conf={ctx.get('confidence')} topk={ctx.get('top_k')} xai={ctx.get('xai_note')}"
