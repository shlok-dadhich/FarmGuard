def test_ensemble_math():
    from src.inference.ensemble import majority_vote, soft_average, agreement_rate
    assert majority_vote([[0,1],[0,0]])==[0,0]
    assert agreement_rate([[0,0],[0,0]])==1.0
    assert len(soft_average([[[0.7,0.3]],[[0.6,0.4]]])[0])==2
