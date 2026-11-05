# PMTT Physics

**Paper:** *A Patch in a Static Image Can Explain Part of the Future*

PMTT physics studies a simple but unexpected fact: without observing the future, a low-dimensional Patch-Level Local Feature Space extracted from one patch in one static image explains part of the latent change produced by future physical evolution.

A patch is represented by a Canonical State together with identity-preserving observation-conditioned states and their local organization. The resulting static geometry is evaluated against held-out future states. Its alignment with physical-time evolution opens a new empirical problem: why can observation-induced static representation geometry explain structured components of future physical change, and why is that geometry organized by the current physical state and constraints?

PMTT physics is the name of this physical-representation research space. It is not a claim of a new fundamental physical law.

## What this repository provides

- a public implementation of the PMTT representation geometry;
- the five observation families used in the study;
- the frozen DINOv3 S/B/L feature interface;
- deterministic synthetic-state generation and exact experiment identities;
- TAP-Vid-DAVIS source, transition, fold, and donor identities;
- static-to-future, acquisition, physical-state, continuity, and reader evaluations;
- small reference-result tables and independent-unit values for the primary comparisons;
- the released scalar reader coefficients and nested sequence splits.

The repository does not redistribute DINOv3 source or checkpoints, TAP-Vid videos, DAVIS frames, or third-party annotations. Obtain those assets from their official sources and pass their locations through the command line.

## Repository layout

```text
src/pmtt_physics/        scientific implementation
config/paper.toml        paper protocol
assets/manifests/        exact experiment identities
assets/parameters/       model identities and frozen small parameters
reference_results/       compact numerical references
tests/                   CPU tests for scientific invariants
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
pmtt-physics validate-assets --root .
pytest
```

DINOv3 must also be installed from its official repository before feature extraction. See [INSTALL.md](INSTALL.md).

## Reproduction entry points

```bash
pmtt-physics --help
```

The main commands are:

- `validate-assets`
- `extract-synthetic`
- `extract-tapvid`
- `extract-reader-past`
- `extract-reader-context`
- `analyze-synthetic`
- `analyze-tapvid`
- `analyze-reader`
- `compare-reference`

A complete command sequence is provided in [REPRODUCE.md](REPRODUCE.md). Dataset and checkpoint requirements are in [DATA.md](DATA.md). Metric definitions and reference files are in [RESULTS.md](RESULTS.md).

## Reproduction modes

The reader diagnostic has two modes:

- `published_scalars` loads the released scalar channel coefficients;
- `refit_scalars` refits those coefficients from the released nested sequence splits.

Both modes reconstruct the Center proposal from the public training folds. The signed-flow reader is a minor diagnostic; it does not turn static future support into solved direction, magnitude, or rollout prediction.

## Numerical scope

Reference values use synthetic base identities and natural-video sequences as the independent statistical units. Patch points, anchors, frames, future branches, and backbone scales do not create additional independent samples. Confidence intervals in the released tables are the saved paper intervals; package construction does not recompute them.

Cross-machine floating-point values may vary with the DINOv3 checkout, accelerator, CUDA, BLAS, and SVD implementation. The repository therefore provides reference tables and exact experiment identities, but it does not assert bit-identical end-to-end results across every environment.

## License

First-party code and documentation are released under the [MIT License](LICENSE). Third-party source, checkpoints, datasets, and annotations retain their own terms. See [THIRD_PARTY.md](THIRD_PARTY.md).

## Citation

Citation metadata is available in [CITATION.cff](CITATION.cff). The paper title is:

> *A Patch in a Static Image Can Explain Part of the Future*
