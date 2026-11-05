# Reproducing the PMTT Physics Results

Run all commands from the repository root. Replace the example asset paths with locations on your machine.

## 1. Validate the released identities

```bash
pmtt-physics validate-assets --root .
pytest
```

This checks the exact counts, schemas, folds, source identities, parameter tables, and compact reference files without loading DINOv3 or TAP-Vid.

## 2. Define external asset paths

```bash
export TAPVID_DAVIS=/path/to/tapvid_davis.pkl
export DINOV3_S=/path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
export DINOV3_B=/path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
export DINOV3_L=/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
mkdir -p outputs/features outputs/results outputs/comparisons
```

## 3. Extract synthetic features

```bash
pmtt-physics extract-synthetic \
  --scale S \
  --checkpoint "$DINOV3_S" \
  --output outputs/features/synthetic-S
pmtt-physics extract-synthetic \
  --scale B \
  --checkpoint "$DINOV3_B" \
  --output outputs/features/synthetic-B
pmtt-physics extract-synthetic \
  --scale L \
  --checkpoint "$DINOV3_L" \
  --output outputs/features/synthetic-L
```

Each output contains all Existing16 and New16 source states, five observation families, and four held-out future branches.

## 4. Extract TAP-Vid features

DINOv3-B uses the full natural-video substrate:

```bash
pmtt-physics extract-tapvid \
  --scale B \
  --checkpoint "$DINOV3_B" \
  --dataset "$TAPVID_DAVIS" \
  --interfaces assets/manifests/tapvid_interfaces.csv \
  --output outputs/features/tapvid-B-full
```

DINOv3-S and DINOv3-L use the fixed cross-scale interface set:

```bash
pmtt-physics extract-tapvid \
  --scale S \
  --checkpoint "$DINOV3_S" \
  --dataset "$TAPVID_DAVIS" \
  --interfaces assets/manifests/tapvid_fixed_interfaces.csv \
  --output outputs/features/tapvid-S-fixed
pmtt-physics extract-tapvid \
  --scale L \
  --checkpoint "$DINOV3_L" \
  --dataset "$TAPVID_DAVIS" \
  --interfaces assets/manifests/tapvid_fixed_interfaces.csv \
  --output outputs/features/tapvid-L-fixed
```

## 5. Analyze synthetic support and physical-state structure

```bash
for scale in S B L; do
  pmtt-physics analyze-synthetic \
    --store "outputs/features/synthetic-${scale}" \
    --output "outputs/results/synthetic-${scale}"
done
```

Each directory contains `summary.csv`, `per_unit.csv`, physical matrices, wrong-state selectors, retrieval results, and state-specific residual results.

## 6. Analyze natural-video support and continuity

Full DINOv3-B:

```bash
pmtt-physics analyze-tapvid \
  --store outputs/features/tapvid-B-full \
  --scope full \
  --output outputs/results/tapvid-B-full
```

Fixed cross-scale comparisons:

```bash
pmtt-physics analyze-tapvid \
  --store outputs/features/tapvid-B-full \
  --scope fixed \
  --output outputs/results/tapvid-B-fixed
pmtt-physics analyze-tapvid \
  --store outputs/features/tapvid-S-fixed \
  --scope fixed \
  --output outputs/results/tapvid-S-fixed
pmtt-physics analyze-tapvid \
  --store outputs/features/tapvid-L-fixed \
  --scope fixed \
  --output outputs/results/tapvid-L-fixed
```

The fixed analysis uses the released donor identities. The full analysis reconstructs its source-only donor comparisons from the extracted source features.

## 7. Reproduce the reader diagnostic

Extract previous-frame features for the fixed B subset:

```bash
pmtt-physics extract-reader-past \
  --scale B \
  --checkpoint "$DINOV3_B" \
  --dataset "$TAPVID_DAVIS" \
  --output outputs/features/reader-past-B
```

Run the exact-paper scalar mode:

```bash
pmtt-physics analyze-reader \
  --store outputs/features/tapvid-B-full \
  --past-root outputs/features/reader-past-B \
  --mode published_scalars \
  --output outputs/results/reader-published-scalars
```

To refit the scalar support channels, first materialize the Context scale-1 variant used by two inner folds:

```bash
pmtt-physics extract-reader-context \
  --scale B \
  --checkpoint "$DINOV3_B" \
  --dataset "$TAPVID_DAVIS" \
  --output outputs/features/reader-context-scale1-B
pmtt-physics analyze-reader \
  --store outputs/features/tapvid-B-full \
  --past-root outputs/features/reader-past-B \
  --context-variant-root outputs/features/reader-context-scale1-B \
  --mode refit_scalars \
  --output outputs/results/reader-refit-scalars
```

The reader evaluates modest functional use of the discovered support geometry. It does not establish exact signed direction, magnitude, or rollout prediction.

## 8. Compare reproduced summaries with references

Example:

```bash
pmtt-physics compare-reference \
  --generated outputs/results/synthetic-B/summary.csv \
  --reference reference_results/cross_scale_results.csv \
  --output outputs/comparisons/synthetic-B.csv
```

Without `--atol`, the command reports matched rows and absolute differences without assigning pass/fail. A tolerance is accepted only when the user has an independently justified numerical bound:

```bash
pmtt-physics compare-reference \
  --generated outputs/results/tapvid-B-full/summary.csv \
  --reference reference_results/primary_results.csv \
  --output outputs/comparisons/tapvid-B-full.csv \
  --atol <justified-absolute-tolerance>
```

## 9. Statistical semantics

- Synthetic estimates first pool within each base identity, then average the 16 independent bases.
- Natural-video estimates first pool within each sequence, then average the 30 independent sequences.
- Confidence intervals use 2,000 deterministic paired bootstrap draws with seed `20261103`.
- Combined32 is a descriptive pooled estimate of Existing16 and New16, not 32 new holdout identities.
- A paired confidence interval must be computed from paired independent-unit values, never by subtracting two marginal confidence intervals.
- TEE is a ratio of squared feature-vector change, not physical energy.

## 10. Expected outputs and scale

Full extraction is computationally and storage intensive. Feature stores are intentionally excluded from this repository. The compact files in `reference_results/` provide the expected scientific values and primary independent-unit records needed for comparison.
