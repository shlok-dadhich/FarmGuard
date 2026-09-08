# Training protocol
Screening: short epochs, 1 seed. Full: 3 seeds, mean+-std. AdamW + cosine default, CE loss, shared augmentation, deterministic stratified 75/10/15 splits in data/splits/<crop>/*.csv. Primary metric macro F1. Track via outputs/metrics + optional MLflow (`mlflow ui --backend-store-uri ./outputs/mlruns`).
