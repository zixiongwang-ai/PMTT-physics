# Reference Results

The tables under `reference_results/` are compact references for the paper protocol. They preserve exact saved means, confidence intervals, independent-unit counts, comparison directions, and the primary per-unit sufficient statistics. They are not a complete archive of every intermediate value.

## Metric semantics

### Temporal Explained Energy (TEE)

TEE measures the fraction of squared future tangent-feature change explained by a static representation subspace. It is not physical energy.

### Coverage

Coverage compares the squared distance from a held-out state to a representation with its squared distance to the Canonical State. `bounded_coverage` uses the bounded continuous PMTT organization; `coverage` or `unbounded_coverage` uses the corresponding directional geometry.

### Subspace overlap

Overlap is basis-invariant and measures alignment between two local subspaces. Real continuity compares the same persistent track against other identities.

### Physical-state gaps

`DINO_gap`, `RGB_gap`, and `Geometry_gap` compare the matched physical state with source-only closest wrong states selected by the named criterion. `matched_minus_wrong` is the state-specific residual after removing object-shared geometry.

### Reader errors

`normalized_error` is total prediction error divided by future tangent energy. `support_miss`, `flow_error_inside`, and `outside_prediction_error` separate missing support from signed-flow selection errors.

## Central reference values

### Static-to-future support

New16 Lawful32 Top-4 TEE:

| Backbone | TEE |
|---|---:|
| DINOv3-S | 35.50% |
| DINOv3-B | 32.54% |
| DINOv3-L | 29.02% |

Fixed natural-video `+1` Lawful32 Top-4 TEE:

| Backbone | TEE |
|---|---:|
| DINOv3-S | 24.10% |
| DINOv3-B | 23.50% |
| DINOv3-L | 27.58% |

DINOv3-B full natural-video substrate:

| Horizon | Top-1 | Top-2 | Top-4 | Top-8 |
|---|---:|---:|---:|---:|
| `+1` | 9.78% | 16.80% | 23.03% | 29.40% |
| `+4` | 9.92% | 16.02% | 21.82% | 27.70% |

### Analytic organization

New16 Observed to bounded PMTT coverage:

| Backbone | Observed | bounded PMTT |
|---|---:|---:|
| S | 13.35% | 44.90% |
| B | 10.61% | 41.36% |
| L | 8.01% | 37.39% |

DINOv3-B full natural video:

| Horizon | Observed | bounded PMTT |
|---|---:|---:|
| `+1` | 9.15% | 28.47% |
| `+4` | 13.48% | 26.18% |

### Observation budget

DINOv3-B full `+1`, relative to the current L64 reference:

| Budget | Recovery | Subspace overlap with L64 |
|---|---:|---:|
| L8 | 66.67% | 0.4660 |
| L16 | 88.31% | 0.7013 |
| L32 | 96.30% | 0.8638 |
| L64 | 100% | 1.0000 |

L64 is the richest observation set evaluated here. It is not asserted to be the complete Patch-Level Local Feature Space.

### Physical-state organization

New16 matched state minus DINO-closest wrong, Top-4 TEE gap:

| Comparison | S Original | B Original | L Original |
|---|---:|---:|---:|
| Configuration | 15.10 pp | 15.07 pp | 14.90 pp |
| Constraint aggregate | 18.29 pp | 20.91 pp | 18.58 pp |

The constraint aggregate combines axis, joint-location, and motion-family blocks. Full per-block, construction, scale, and selector results are in `physical_controls.csv`, `state_specific_results.csv`, `physical_matrices.csv`, and `paired_effects.csv`.

### Persistent physical identity

Same-track minus same-video DINO-similar donor overlap at `+1`:

| Substrate | Gap |
|---|---:|
| S fixed | +0.2743 |
| B fixed | +0.2913 |
| L fixed | +0.2969 |
| B full | +0.2583 |

### Reader diagnostic

DINOv3-B fixed-subset `+1` normalized error:

| Method | Error |
|---|---:|
| Zero | 1.000000 |
| DampedPast | 0.944521 |
| Orbital16 | 0.924765 |
| SamplingMatched16 | 0.928125 |
| Lawful32 | 0.924382 |
| Lawful64 | 0.919030 |
| Sensitivity16 | 0.966331 |
| GenericGeometry | 0.928159 |

Lower is better. This is a minor functional diagnostic, not solved future prediction.

## Reference-file map

| File | Purpose |
|---|---|
| `primary_results.csv` | compact central values used across the main scientific story |
| `cross_scale_results.csv` | primary S/B/L confirmations |
| `acquisition_results.csv` | cross-family prediction and subspace overlap |
| `observation_budget.csv` | L8/L16/L32/L64 recovery and complementarity |
| `physical_controls.csv` | matched-state and closest-wrong controls |
| `state_specific_results.csv` | object-shared removal results |
| `continuity_results.csv` | persistent-identity continuity |
| `prediction_diagnostic.csv` | frozen reader error decomposition |
| `retrieval_results.csv` | physical-state retrieval accuracy |
| `physical_matrices.csv` | complete synthetic target-by-source matrices |
| `paired_effects.csv` | saved paired comparisons and confidence intervals |
| `primary_per_unit.csv` | independent-unit values for the primary effects |
| `expected_metrics.toml` | compact machine-readable headline values |

## Statistical unit

The independent unit is a synthetic base identity or a natural-video sequence. Rows corresponding to points, anchors, frames, future branches, or backbone scales are not independent replicates.

## Scope

The released evidence establishes static future support, low-dimensional organization, acquisition sharing and complementarity, physical-state conditionality, and persistent-identity continuity within the tested DINOv3 S/B/L and synthetic/TAP-Vid-DAVIS substrates. It does not establish force, mass, friction, hidden mechanics, a Euclidean copy of physical parameter distance, a complete reachable boundary, or exact signed long-horizon dynamics.
