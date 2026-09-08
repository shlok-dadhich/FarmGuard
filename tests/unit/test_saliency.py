"""Saliency XAI method tests."""

from PIL import Image

from src.explainability.saliency import Saliency
from src.inference.preprocessing import preprocess_pil
from src.models.factory import build_model, get_adapter
from src.utils.image import make_demo_image


def _setup():
    ad = get_adapter("mock_demo")
    m = build_model("mock_demo", 4).eval()
    p = make_demo_image("outputs/_sal_test.jpg")
    x = preprocess_pil(Image.open(p).convert("RGB"), ad.input_size())
    return m, x


def test_saliency_dims_and_range():
    m, x = _setup()
    sal, c = Saliency(m)(x)
    assert sal.shape == (x.shape[2], x.shape[3])
    assert 0.0 <= sal.max() <= 1.0 + 1e-6
    assert c == int(m(x).argmax(1)[0])


def test_saliency_class_idx():
    m, x = _setup()
    _, c = Saliency(m)(x, class_idx=1)
    assert c == 1


def test_saliency_via_pipeline():
    m, x = _setup()
    from src.explainability.xai_pipeline import run_xai

    r = run_xai(m, get_adapter("mock_demo"), Image.open("outputs/_sal_test.jpg").convert("RGB"),
                x, methods=("saliency",))
    assert "saliency" in r and "file" in r["saliency"]