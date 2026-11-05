from pathlib import Path
import hashlib

import numpy as np

from pmtt_physics.synthetic import load_conditions, physical_pose, render


ROOT = Path(__file__).resolve().parents[1]


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _condition(base_id: str, block: str, state_index: int):
    conditions = load_conditions(ROOT / "assets" / "manifests" / "synthetic_conditions.csv")
    return next(
        condition
        for condition in conditions
        if condition.base.base_id == base_id
        and condition.block == block
        and condition.state_index == state_index
    )


def test_existing16_renderer_identity() -> None:
    condition = _condition("existing-00", "configuration", 0)
    rotations, translations, half_extents = physical_pose(
        condition.base, condition.block, condition.value
    )
    image, depth, surface = render(condition.base, rotations, translations, half_extents)
    assert _sha(np.asarray(image)) == "b4d4b5d2ed8c5839f0e072855a7de7423ccb2973294401f5edf3898335e0d8bc"
    assert _sha(depth) == "72fbcf0b93363e532915abc5670fd9c7148cdb508ac1d05a5da2ce7d1cb2c3b5"
    assert _sha(surface) == "4ab680c2b64171bbc0fb1528150beefd54200d64d85e49cd193f1a18dc478762"


def test_new16_renderer_identity() -> None:
    condition = _condition("holdout-00", "configuration", 0)
    rotations, translations, half_extents = physical_pose(
        condition.base, condition.block, condition.value
    )
    image, depth, surface = render(condition.base, rotations, translations, half_extents)
    assert _sha(np.asarray(image)) == "9a4bc2fd89d1c506da361952323e56f8732ac1a64e72d40bf4151426e1a92358"
    assert _sha(depth) == "dd70b16cf2ca044ad5285163488e8d97c983ee7f6614681d04482b6903c69a54"
    assert _sha(surface) == "ce62b978cf258df10c1417fe0d70fa306242dd0f461996eafb0fa401d0b01fe2"
