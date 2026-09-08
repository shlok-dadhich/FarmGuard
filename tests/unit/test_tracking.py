def test_tracker_file(tmp_path):
    from src.tracking.tracker import ExperimentTracker
    from src.tracking.run_schema import RunRecord
    t = ExperimentTracker({"backend":"file","file_store":str(tmp_path)})
    rec = RunRecord(run_id="r1", crop="tomato", architecture="mock_demo", accuracy=0.5, macro_f1=0.4)
    t.finish_run(rec)
    assert (tmp_path/"r1.json").exists() and (tmp_path/"runs.csv").exists()


def test_append_never_overwrites(tmp_path):
    from src.tracking.tracker import ExperimentTracker
    from src.tracking.run_schema import RunRecord
    import pandas as pd
    t = ExperimentTracker({"backend":"file","file_store":str(tmp_path)})
    t.finish_run(RunRecord(run_id="a", crop="tomato", architecture="mock_demo", accuracy=0.1))
    t.finish_run(RunRecord(run_id="b", crop="tomato", architecture="mock_demo", accuracy=0.2))
    df = pd.read_csv(tmp_path / "runs.csv")
    assert set(df["run_id"]) == {"a", "b"}  # history is appended, never overwritten


def test_schema_migration(tmp_path):
    """A CSV written by an older schema version gains the new columns on append."""
    from src.tracking.run_schema import RunRecord, append_run
    import pandas as pd
    csv = tmp_path / "runs.csv"
    csv.write_text("timestamp,run_id,crop,architecture,accuracy,macro_f1\n"
                   "2026-01-01,old,tomato,mock_demo,0.5,0.4\n")
    rec = RunRecord(run_id="new", crop="tomato", architecture="mock_demo", accuracy=0.6,
                    macro_f1=0.5, kind="eval", split="test", weighted_f1=0.5)
    append_run(rec, tmp_path)
    df = pd.read_csv(csv)
    assert len(df) == 2
    for col in ("accuracy", "macro_f1", "kind", "split", "weighted_f1"):
        assert col in df.columns
    assert df[df["run_id"] == "new"].iloc[0]["kind"] == "eval"


def test_append_uses_file_column_order(tmp_path):
    """Appended rows follow the existing file's column order (regression: dataclass
    field order may differ from a migrated/rebuilt header)."""
    from src.tracking.run_schema import RunRecord, append_run
    import pandas as pd
    csv = tmp_path / "runs.csv"
    csv.write_text("timestamp,run_id,crop,architecture,accuracy,macro_f1,kind,split,weighted_f1,evaluation_samples\n"
                   "2026-01-01,old,tomato,mock_demo,0.5,0.4,train,val,0.4,100\n")
    rec = RunRecord(run_id="new", crop="tomato", architecture="mock_demo", accuracy=0.6,
                    macro_f1=0.5, kind="eval", split="test", weighted_f1=0.5, evaluation_samples=200)
    append_run(rec, tmp_path)
    df = pd.read_csv(csv)
    new_row = df[df["run_id"] == "new"].iloc[0]
    assert len(df) == 2
    assert new_row["accuracy"] == 0.6 and new_row["macro_f1"] == 0.5
    assert new_row["evaluation_samples"] == 200 and new_row["weighted_f1"] == 0.5
    assert new_row["kind"] == "eval" and new_row["split"] == "test"


def test_corrupt_csv_rebuilt_from_json(tmp_path):
    """Inconsistent rows (schema drift/corruption) trigger a rebuild from JSON history."""
    from src.tracking.run_schema import RunRecord, append_run
    import pandas as pd
    (tmp_path / "runs.csv").write_text("a,b\n1,2\n3,4,5,6,7\n")
    rec = RunRecord(run_id="run-r9", crop="tomato", architecture="mock_demo", accuracy=0.7)
    append_run(rec, tmp_path)
    df = pd.read_csv(tmp_path / "runs.csv")
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "run-r9"  # rebuilt from the JSON (authoritative), old junk dropped
