# Model contract (external team)
Place code: models/custom/<arch>.py ; weights: models/checkpoints/<crop>_<arch>.pt as {"state_dict":...}.
Implement/adjust adapter in src/models/adapters/<arch>.py: build(), target_layers(), parameter_count().
Never commit large checkpoints. App detects missing checkpoints with actionable errors.
