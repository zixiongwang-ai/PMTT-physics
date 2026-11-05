import numpy as np
import torch

from pmtt_physics.observations import acquisition_pools, canonicalize_sampling, pair_indices
from pmtt_physics.randomness import source_observation_seed, stable_seed


def test_acquisition_membership() -> None:
    pools = acquisition_pools()
    assert len(pools["Lawful32"]) == 32
    assert len(pools["Lawful64"]) == 64
    assert len(pools["L8"]) == 8
    assert len(pools["L16"]) == 16
    assert len(pools["L32"]) == 32
    assert set(pools["L8"]) < set(pools["L16"]) < set(pools["L32"]) < set(pools["L64"])
    assert set(pools["L_A32"]).isdisjoint(pools["L_B32"])
    assert set(pools["L_A32"]) | set(pools["L_B32"]) == set(range(64))
    assert len(pair_indices("Orbital", 0)) == 8
    assert len(pair_indices("Geometric", 0)) == 8
    assert len(pair_indices("Context", 0)) == 8


def test_sampling_canonicalization() -> None:
    states = torch.arange(5 * 16, dtype=torch.float32).reshape(1, 5, 16, 1)
    seed = 12345
    permutation = np.random.default_rng(seed).permutation(16)
    scrambled = states.clone()
    scrambled[0, 1] = states[0, 1, permutation]
    restored = canonicalize_sampling(scrambled, [seed])
    assert torch.equal(restored, states)


def test_frozen_seeds() -> None:
    assert source_observation_seed(1000, 0) == 20261102 + 1000 * 32 * 7919
    assert source_observation_seed(10000, 31) == 20261102 + (10000 * 32 + 31) * 7919
    assert stable_seed("sequence", 7, 4) == stable_seed("sequence", 7, 4)
    assert stable_seed("sequence", 7, 4) != stable_seed("sequence", 7, 5)
