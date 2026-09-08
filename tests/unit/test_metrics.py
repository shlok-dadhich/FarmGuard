def test_macro_f1():
    from src.evaluation.metrics import compute_metrics
    m = compute_metrics([0,1,1,0],[0,1,0,0])
    assert 0 <= m["macro_f1"] <= 1 and m["accuracy"]==0.75
