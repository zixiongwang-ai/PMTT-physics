# Installation

## Supported environment

The reference environment used Python 3.11 with:

- PyTorch 2.7.1
- torchvision 0.22.1
- NumPy 2.2.6
- SciPy 1.15.3
- Pillow 12.3.0

A CUDA-capable accelerator is strongly recommended for DINOv3 feature extraction. CPU execution is suitable for the included tests and small mathematical checks, but not for the full feature workload.

## Create the Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

For a specific CUDA build, install the matching PyTorch and torchvision wheels using the official PyTorch instructions, then install this project with the same versions.

## Install DINOv3

Obtain the official DINOv3 source outside this repository and install it in the active environment:

```bash
git clone https://github.com/facebookresearch/dinov3.git external/dinov3
python -m pip install -e external/dinov3
```

The implementation imports `dinov3.hub.backbones` and uses the public patch-token interface. The release does not specify an immutable DINOv3 source commit, so source-level bit identity is not claimed. Record the source revision used for every reproduction. DINOv3 requires PyTorch 2.7.1 or newer according to its official installation documentation. DINOv3 code and model weights are governed by the DINOv3 License.

## Verify the installation

```bash
pmtt-physics --help
pmtt-physics validate-assets --root .
pytest
```

`validate-assets` reads only the released CSV and TOML files. It does not load a dataset, checkpoint, or CUDA context.

## Checkpoint validation

Feature extraction checks the expected checkpoint filename and byte size before loading it. The expected identities are stored in:

```text
assets/parameters/model_registry.toml
```

The release does not include checkpoint SHA-256 values because the supplied scientific record contains filenames and byte sizes, not frozen hashes. Do not substitute a checkpoint with a different filename or size without treating it as a different model identity.

## Platform notes

- The main numerical workload is designed for Linux and CUDA.
- DINOv3 source compatibility matters because the implementation accesses its public backbone construction and token interfaces.
- Small differences can arise from CUDA, BLAS, SVD, or interpolation implementations.
- All input and output locations are supplied through command-line arguments.
