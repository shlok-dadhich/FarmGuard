# AgroVision
Multi-crop plant disease classification research prototype.
## Setup
pip install -r requirements.txt
## Run
streamlit run app.py
python scripts/prepare_data.py --crop tomato
python scripts/train.py --crop tomato --model mock_demo --seed 42
python scripts/evaluate.py --crop tomato --model mock_demo
python scripts/generate_xai.py --image outputs/demo_leaf.jpg --model mock_demo
python scripts/smoke_test.py
## Tracking (MLflow alternative)
Default file backend (outputs/metrics). Optional MLflow: set tracking.backend=both, `mlflow ui --backend-store-uri ./outputs/mlruns`. Optional TensorBoard: `tensorboard --logdir outputs/tb`. See docs/tracking.md.
## Models
models/ is externally owned. See docs/model_contract.md. DEMO mock_demo model is clearly labelled and never a research result.
