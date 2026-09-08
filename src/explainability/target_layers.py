from __future__ import annotations
def resolve_target_layers(adapter, module):
    try:
        layers = adapter.target_layers(module)
        assert layers, "adapter returned no layers"
        return layers
    except Exception as e:
        raise RuntimeError(f"XAI target layer failure: {e}. HOW: implement target_layers() in the adapter.") from e
