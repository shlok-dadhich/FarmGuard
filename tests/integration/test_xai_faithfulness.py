def test_faithfulness():
    import numpy as np
    from src.explainability.faithfulness import pointing_hit, faithfulness_score
    h = np.zeros((8,8)); h[2,3]=1.0
    mask = np.zeros((8,8)); mask[2,3]=1
    assert pointing_hit(h, mask) is True
    assert faithfulness_score([True,False])["score"]==0.5
    assert faithfulness_score([])["score"] is None
