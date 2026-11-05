# Third-Party Software and Data

The root MIT License applies only to the first-party PMTT physics code and documentation in this repository. External source, pretrained checkpoints, datasets, annotations, and videos retain their own licenses and terms.

## DINOv3

- Source: https://github.com/facebookresearch/dinov3
- Use: frozen S/B/L visual backbones and patch-token features
- Redistribution: not included in this repository
- License: DINOv3 code and model weights are released under the DINOv3 License. Review the license in the official repository before use.

The checkpoint filenames and byte sizes expected by this implementation are listed in `assets/parameters/model_registry.toml`.

## TAPNet and TAP-Vid

- Source: https://github.com/google-deepmind/tapnet
- Use: official TAP-Vid-DAVIS data format and benchmark identities
- Redistribution: no software, checkpoints, videos, frames, or annotations are included here

The official TAPNet repository states that its general software is licensed under Apache License 2.0. It also states that TAP-Vid annotations and specified non-software materials are released under Creative Commons Attribution 4.0, while the original DAVIS source videos retain the Creative Commons terms of their creators. Consult the official TAPNet and DAVIS pages for the terms that apply to the files you obtain.

## Python dependencies

NumPy, SciPy, Pillow, PyTorch, torchvision, setuptools, wheel, and pytest are external projects distributed under their own licenses. They are dependencies, not copied source.

## Dataset and checkpoint boundary

This repository contains only first-party code, experiment identities, frozen small parameter tables, and compact reference results. Users must obtain third-party source, checkpoints, and datasets themselves and are responsible for complying with their terms.
