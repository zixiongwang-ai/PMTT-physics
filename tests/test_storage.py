from pathlib import Path

import numpy as np

from pmtt_physics.storage import FeatureStore, create_store, write_rows


def test_safe_feature_store(tmp_path: Path) -> None:
    arrays = create_store(
        tmp_path,
        kind="synthetic",
        scale="S",
        rows=2,
        width=4,
        future_branches=4,
        rgb_width=6,
        geometry_width=3,
    )
    arrays["center"][:] = 1
    arrays["states"][:] = 2
    arrays["future"][:] = 3
    arrays["rgb"][:] = 4
    arrays["projected_geometry"][:] = 5
    for array in arrays.values():
        array.flush()
    write_rows(
        tmp_path / "metadata.csv",
        [
            {"sample_id": "a", "source_seed": 1, "unit": "u1"},
            {"sample_id": "b", "source_seed": 2, "unit": "u2"},
        ],
    )
    store = FeatureStore.open(tmp_path)
    assert store.kind == "synthetic"
    assert store.scale == "S"
    assert store.arrays["center"].dtype == np.float32
    assert store.arrays["states"].shape == (2, 5, 16, 4)
    assert store.index == {"a": 0, "b": 1}
