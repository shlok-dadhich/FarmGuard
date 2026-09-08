def test_tracker_file(tmp_path):
    from src.tracking.tracker import ExperimentTracker
    from src.tracking.run_schema import RunRecord
    t = ExperimentTracker({"backend":"file","file_store":str(tmp_path)})
    rec = RunRecord(run_id="r1", crop="tomato", architecture="mock_demo", accuracy=0.5, macro_f1=0.4)
    t.finish_run(rec)
    assert (tmp_path/"r1.json").exists() and (tmp_path/"runs.csv").exists()
