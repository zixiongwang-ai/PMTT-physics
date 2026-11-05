
# PMTT Physics

**Paper:** [A Patch in a Static Image Can Explain Part of the Future](https://doi.org/10.5281/zenodo.23984040)

**Foundational paper:** [A Patch is More Than a Token](https://arxiv.org/abs/2610.41814)

PMTT Physics builds on the patch-level representation framework introduced in *A Patch is More Than a Token*.

PMTT Physics studies a simple but unexpected finding: without observing the future, a low-dimensional Patch-Level Local Feature Space extracted from one patch in one static image explains part of the latent change produced by future physical evolution.

A patch is represented by a Canonical State together with identity-preserving observation-conditioned states and their local organization. A frozen, image-pretrained DINOv3 backbone supplies the representations. The static geometry is constructed analytically from the current image before future states are introduced for independent evaluation.

The study establishes two connected scientific contributions:

1. **A new representation foundation for deep-learning physical modeling.** The Patch-Level Local Feature Space and its continuous low-dimensional Manifold organization provide structured support for future latent variation, before learning a temporal future-prediction mapping.
2. **A New Physics Question Space.** Geometry formed along the static Observation axis systematically aligns with subsequent change along the Physical-time axis. This alignment is organized by current physical configuration, constraints, joint location, and motion family. The resulting question is why static observation-induced representation geometry contains directions aligned with future physical evolution, and why that alignment depends on the current physical state and constraints.

Together, these findings connect static visual representation geometry with physical-time dynamics and open new opportunities for representation learning and physical modeling.

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

## Relationship to PMTT

The foundational paper, [A Patch is More Than a Token](https://arxiv.org/abs/2610.41814), introduces the Representation State Family and Patch-Level Local Feature Space associated with one persistent image patch. It establishes the structured, low-dimensional geometry of this representation and demonstrates its value for intrinsic representation analysis and downstream computer vision tasks.

PMTT Physics takes this representation foundation into a distinct scientific problem: whether geometry constructed from one static image aligns with future physical evolution.

The study investigates both the representation value of this static geometry for future latent dynamics and the physical structure of the alignment itself. Controlled experiments and natural-video observations characterize its dependence on physical state, constraints, and persistent identity.

The two papers therefore form a connected research line:

**PMTT:** Patch representation, observation-conditioned states, and local feature geometry.

**PMTT Physics:** Static patch geometry, future physical dynamics, and the New Physics Question Space.

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

Both modes reconstruct the Center proposal from the public training folds. The signed-flow reader is a separate functional diagnostic operating on fixed source geometry. It investigates the use of static support for temporal selection rather than replacing the primary static-to-future representation measurements.

## Numerical scope

Reference values use synthetic base identities and natural-video sequences as the independent statistical units. Patch points, anchors, frames, future branches, and backbone scales do not create additional independent samples. Confidence intervals in the released tables are the saved paper intervals; package construction does not recompute them.

Cross-machine floating-point values may vary with the DINOv3 checkout, accelerator, CUDA, BLAS, and SVD implementation. The repository therefore provides reference tables and exact experiment identities, but it does not assert bit-identical end-to-end results across every environment.

## Papers and citation

**Current paper**

[A Patch in a Static Image Can Explain Part of the Future](https://doi.org/10.5281/zenodo.23984040)  
Zixiong Wang and Haiping Ma, 2026.  
DOI: 10.5281/zenodo.23984040

**Foundational paper**

[A Patch is More Than a Token](https://arxiv.org/abs/2610.41814)  
Zixiong Wang and Haiping Ma, 2026.  
arXiv:2610.41814

Citation metadata for PMTT Physics is available in [CITATION.cff](CITATION.cff).

For inquiries: Zixiong Wang, `zixiongwang.ai@outlook.com`.

## License

First-party code and documentation are released under the [MIT License](LICENSE). Third-party source, checkpoints, datasets, and annotations retain their own terms. See [THIRD_PARTY.md](THIRD_PARTY.md).
