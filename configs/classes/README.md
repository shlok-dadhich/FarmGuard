# configs/classes/

Per-crop class mapping files used by model *instances* (configs/models.yaml ->
`instances:` -> `classes_file`).

## Format

```yaml
# configs/classes/<crop>.yaml
classes:
  - Bacterial_spot
  - Early_blight
  - Healthy
  - Late_blight
```

The list order defines the model's output index order (index i = classes[i]).
It must match the order the model was trained with.

## How it is used

1. `src/models/metadata.py` loads the file when an instance for that crop declares
   a `classes_file`.
2. Class-name resolution priority (see `ui/common.get_class_names`):
   1. split-manifest labels from `data/splits/<crop>/*.csv` (source of truth for
      evaluation ground truth)
   2. `classes_file` declared on an instance for the crop
   3. `datasets.yaml -> crops.<crop>.classes`
   4. generic `class0..class3` fallback (never used during real evaluation)

If the class list does not match the checkpoint's head size, loading fails with a
clear `CheckpointError` explaining the mismatch — nothing is silently resized.