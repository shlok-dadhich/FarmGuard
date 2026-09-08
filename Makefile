setup:
	pip install -r requirements.txt
smoke:
	python scripts/smoke_test.py
train:
	python scripts/train.py --crop tomato --model mock_demo
app:
	streamlit run app.py
mlflow-ui:
	mlflow ui --backend-store-uri ./outputs/mlruns
tb:
	tensorboard --logdir outputs/tb
