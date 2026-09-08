# Model Integration

AgroVision never trains or rewrites your models. It wraps externally supplied
checkpoints behind a clean interface (`src/models/interface.py`) so the rest of the
platform (inference, evaluation, XAI, tracking, UI) never depends on model internals.

## 1. The two concepts

1. **Architecture registry** — `configs/models.yaml -> models:` maps a key
   (e.g. `efficientnet_v2_s`) to an *adapter* that knows how to build the module,
   resolve XAI target layers, and report input size / parameter count.
2. **Model instances** — `configs/models.yaml -> instances:` binds a concrete
   checkpoint + crop + class mapping to an architecture. This is what a user edits
   to register a trained model.

## 2. Checkpoint placement

```
models/
├── checkpoints/
│   └── tomato/
│       └── my_model.pt
└── custom/
    └── my_model.py
```

Accepted formats:
- `{"state_dict": {...}, ...}` (dict with a `state_dict` key), or
- a bare `state_dict`.

Loading is strict about structure but tolerant of key names: `strict=False` is used
and mismatches surface as `CheckpointError` messages listing the configured path,
expected classes and the model head size.

Two discovery modes:
- **Explicit** — `instances:` entry with a `checkpoint:` path (relative to repo root).
- **Auto-discovery** — when no `checkpoint:` is set, the newest
  `models/checkpoints/<crop>_<arch>*.pt` (also `outputs/checkpoints/`) wins.

## 3. Instance configuration

```yaml
# configs/models.yaml
instances:
  - id: efficientnet_crop1
    name: EfficientNetV2-S (crop1)
    architecture: efficientnet_v2_s   # key from models.yaml -> models:
    crop: tomato                      # or crops: [tomato, pepper]
    checkpoint: models/checkpoints/tomato/efficientnet_v2_s.pt
    classes_file: configs/classes/tomato.yaml
    input_size: 224                   # optional; defaults to the adapter's size
    target_layer: auto                # optional; adapters resolve their own layers
    notes: trained on v1 data
```

Validation (`src/models/metadata.py`) raises `ConfigError` with actionable messages
for: missing `id`, duplicate `id`, unknown `architecture`, missing `crop`, bad
`input_size`. A `classes_file` that does not parse raises too — nothing is silently
defaulted.

## 4. Class mapping

Priority when resolving class names for a crop (`ui/common.get_class_names`):

1. `data/splits/<crop>/*.csv` manifest labels — the source of truth for evaluation
   ground truth.
2. `classes_file` declared on an instance for that crop.
3. `datasets.yaml -> crops.<crop>.classes`.
4. Generic `class0..class3` fallback — never used during real evaluation.

The classes file format:

```yaml
# configs/classes/tomato.yaml
classes:
  - Bacterial_spot
  - Early_blight
  - Healthy
  - Late_blight
```

## 5. Standard-library architectures (no code needed)

| Key | torchvision name | input |
|---|---|---|
| `efficientnet_v2_s` | efficientnet_v2_s | 224 |
| `resnet50` | resnet50 | 224 |
| `convnext_tiny` | convnext_tiny | 224 |
| `regnet_y_4gf` | regnet_y_4gf | 224 |
| `densenet121` | densenet121 | 224 |

These adapters (`src/models/adapters/`) build the torchvision model, replace the
head with `num_classes` outputs, resolve a sensible XAI target layer, and report
parameter count. `mock_demo` is the deterministic demo/test model (NEVER research).

## 6. Custom externally supplied models

Only needed when your architecture is not one of the built-ins.

1. **Place the model code** in `models/custom/<arch>.py` (a plain `nn.Module`).
2. **Write an adapter** under `src/models/adapters/<arch>.py`:

```python
from src.models.interface import ModelAdapter

class MyModelAdapter(ModelAdapter):
    architecture = "my_model"

    def build(self, num_classes, pretrained=False):
        from models.custom.my_model import MyModel
        return MyModel(num_classes=num_classes)

    def target_layers(self, module):
        return [module.features[-1]]   # last conv block — model-specific!

    def parameter_count(self, module):
        return sum(p.numel() for p in module.parameters())

    def input_size(self):
        return 224
```

3. **Register** in `src/models/registry.py` and add the architecture key to
   `configs/models.yaml -> models:` (adapter dotted path).
4. Reference it from an `instances:` entry as usual.

The adapter contract (`src/models/interface.py`) requires: `build`,
`target_layers`, `parameter_count`, `input_size`, plus optional `flops`/`metadata`.
Target layers are per-architecture — nothing assumes `model.layer4[-1]` works everywhere.

## 7. Error reporting

Checkpoint/class/target-layer failures raise exceptions with WHAT / WHY / HOW
structure, e.g.:

```
WHAT: Checkpoint not found: models/checkpoints/tomato/x.pt
WHY: file missing
HOW: place externally supplied weights under models/checkpoints/ or train via scripts/train.py
```

The UI surfaces these verbatim so a person adding a model never has to read source.