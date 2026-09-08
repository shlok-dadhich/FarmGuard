def test_seed_deterministic():
    from src.core.device import set_seed
    import random
    set_seed(7); a = random.random(); set_seed(7); assert random.random()==a
