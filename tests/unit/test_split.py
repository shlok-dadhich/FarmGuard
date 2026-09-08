def test_split_ratios():
    import pandas as pd
    from src.data.splitting import stratified_split, check_leakage
    df = pd.DataFrame({"path":[f"p{i}" for i in range(100)],"label":[f"c{i%4}" for i in range(100)],"label_id":[i%4 for i in range(100)]})
    s = stratified_split(df, seed=42)
    assert len(s["train"])==75 and len(s["val"])==10 and len(s["test"])==15
    assert check_leakage(s)==[]
