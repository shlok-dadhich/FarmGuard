# Explainability (XAI)

## Methods (`src/explainability/`)

| Method | File | Notes |
|---|---|---|
| **Grad-CAM++** | `gradcam.py` | default; class-discriminative localization on the adapter's target layer |
| **Score-CAM** | `scorecam.py` | perturbation-based cross-check; subsampled masks for CPU speed |
| **Saliency** | `saliency.py` | input-gradient map (channel-max abs gradient); works for any differentiable model, no target layer needed |

All methods share the interface `(heatmap_in_0_1, class_idx)`.

## Target layers

`src/explainability/target_layers.py` delegates to each adapter's `target_layers()`
so every architecture resolves its own feature/conv layer — nothing assumes
`model.layer4[-1]` works everywhere. If resolution fails, the error explains why and
the rest of the app keeps working.

## Supported / unsupported models

- CNNs (torchvision + custom) → all three methods.
- Models without a resolvable target layer → Grad-CAM++/Score-CAM unavailable for
  that model; Saliency may still work (gradient-based).
- Models that don't propagate gradients to the input (e.g. inference-only wrappers)
  → Saliency reports *"XAI method unavailable for this model"* with the reason.

The UI never crashes on an unsupported method — it shows the error in that method's
tab and keeps the others usable.

## Pipeline

`xai_pipeline.run_xai(model, adapter, pil_img, input_tensor, methods, class_idx, meta)`
writes, per method, a PNG with a meaningful filename plus one meta JSON sidecar:

```
outputs/xai/
├── xai_<ts>_gradcampp_c<cls>.jpg
├── xai_<ts>_scorecam_c<cls>.jpg
├── xai_<ts>_saliency_c<cls>.jpg
└── xai_<ts>_meta.json       # prediction, confidence, target class, model, method
```

## Faithfulness (pointing game)

`src/explainability/faithfulness.py`:

- `pointing_hit(heatmap, mask)` — is the heatmap maximum inside the lesion mask?
- `faithfulness_score(hits)` — hits / evaluated samples.

Requires binary lesion masks (`data/raw/<crop>/masks/` or uploaded per image in the
UI). Without masks the platform reports:

> Quantitative faithfulness evaluation unavailable because lesion masks are not provided.

Qualitative XAI always works.

## CLI

```bash
python scripts/generate_xai.py --image img.jpg --model <id> --method both
python scripts/generate_xai.py --image img.jpg --model <id> --method saliency
```

## UI

The Diagnosis and Explainability pages expose: method selector, overlay opacity,
target class (predicted or selected), original / heatmap / overlay / side-by-side
views, target-layer metadata, and artifact download.