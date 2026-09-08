def test_xai_dims():
    import torch
    from PIL import Image
    from src.models.factory import build_model, get_adapter
    from src.inference.preprocessing import preprocess_pil
    from src.utils.image import make_demo_image
    from src.explainability.gradcam import GradCAMpp
    from src.explainability.target_layers import resolve_target_layers
    ad = get_adapter("mock_demo"); m = build_model("mock_demo", 4).eval()
    p = make_demo_image("outputs/_t.jpg")
    x = preprocess_pil(Image.open(p).convert("RGB"), ad.input_size())
    cam, c = GradCAMpp(m, resolve_target_layers(ad, m)[-1])(x)
    assert cam.shape[0] > 0 and cam.max() <= 1.0 + 1e-6
