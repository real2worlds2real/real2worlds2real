# Third-party software

Most dependencies are installed separately and are not redistributed in this repository: Blender, Isaac Lab / Isaac Sim / PhysX / Isaac RTX, Newton, Warp, Codex, Python packages, COLMAP and optional VGGT-Omega. Their upstream licenses and service terms apply. Model weights and simulator container images are not included.

Optional Gaussian training/transcoding invokes a separately installed [nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut) checkout through its documented command interfaces. No upstream training/export source is vendored. The native CPU PLY/OpenUSD preparation module is a local implementation using the installed plyfile/OpenUSD APIs. Record upstream revisions and obtain licenses/data permissions for the assets used in each deployment.

Optional photo fitting uses SciPy, automatic UV unwrapping uses [xatlas](https://github.com/mworchel/xatlas-python), and image restoration uses a separately installed official [Fixer](https://github.com/nv-tlabs/Fixer) with its Cosmos dependencies and model weights.

## Local changes to external libraries

- [SAM 3D Objects configuration](setup/configure_sam.py). Setup adjusts CUDA environment detection and renderer defaults in the separately fetched revision `f91db411c50efee93d8db7aeb323885650f6f722`. The external Gaussian renderer also carries Inria/GRAPHDECO non-commercial research and evaluation terms. Review the [SAM License](patches/sam-3d-objects.LICENSE) and the [renderer notice at the pinned revision](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/model/backbone/tdfy_dit/renderers/gaussian_render.py).
- [Fixer compatibility patch](requirements/fixer-text-encoder-lazy.patch). The upstream file is `imaginaire/auxiliary/text_encoder.py` in [NVIDIA Cosmos Predict2 1.0.9](https://pypi.org/project/cosmos-predict2/1.0.9/), copyright 2025 NVIDIA CORPORATION & AFFILIATES. The local change defers two text-encoder imports. The [Apache License 2.0](requirements/cosmos-predict2.LICENSE) is copied from that release.

## Blender Cell Fracture

`vendor/cell-fracture/object_fracture_cell/` contains the official `object_fracture_cell` add-on from [blender/blender-addons](https://github.com/blender/blender-addons/tree/b42d68627734cb18af0e6f41537063984313a284/object_fracture_cell), with local integration modifications. Original copyright and **GPL-2.0-or-later** headers are retained. The [GPL text](vendor/cell-fracture/COPYING) and [change record](vendor/cell-fracture/PROVENANCE.json) accompany its source.

The integration uses Blender 4.5's MANIFOLD Boolean solver, scale-relative merge tolerance, coincident adjacent Voronoi boundaries, and disables destructive post-Boolean cleanup. The [fracture exporter](scripts/export_blend_fracture.py) imports the add-on. When redistributing the combined Blender integration, preserve the GPL terms and provide the corresponding source, including the integration changes. The source checkout contains the exporter and the modified add-on.

Model weights and source photographs are supplied separately.

Optional solid control meshing uses separately installed [MeshPy](https://github.com/inducer/meshpy) and its TetGen backend. Check both upstream license terms for the intended deployment; TetGen has separate terms from MeshPy. No mesher source is vendored. Dimensionality-specific deformable geometry and behavioral testing draw research inspiration from [CoDimRecon](https://arxiv.org/html/2609.36024v3); the native Newton adapter and harness here are local implementations, not a copy of its unpublished code or its IPC-family solver.

The optional concave rigid contact diagnostics follow the supported SDF mesh contact path demonstrated in the official [Newton nut/bolt example](https://github.com/newton-physics/newton/blob/main/newton/examples/contacts/example_nut_bolt_sdf.py). Newton is installed separately under its upstream license. The request contract, measured motion checks, agent tool and replay here are local harness code; no example mesh assets are redistributed.

## Scene pipeline submodules

Git stores a commit reference for each submodule. Initializing submodules downloads the corresponding sources. Check the separate model card and terms when obtaining weights. See the [robot inventory](assets/robots/README.md) for model-specific files, notices and supported configurations.

| Submodule | Terms at the pinned commit |
|---|---|
| SAM 3D Objects | [SAM License](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/LICENSE), including the research publication acknowledgement in §1(b)(ii). Individual files may have additional notices. |
| MoGe | [MIT and Apache terms for the respective source components](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/LICENSE) |
| Depth Anything 3 | [Apache-2.0 for code](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/LICENSE) |
| 3DGRUT | [Apache-2.0](https://github.com/nv-tlabs/3dgrut/blob/e6f4552b408af020a96de09ab5757f9b57430aaa/LICENSE) |
| Diffusion Policy | [MIT](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/LICENSE) |
| OpenPI | [Apache-2.0](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/LICENSE) and [upstream Gemma terms](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/LICENSE_GEMMA.txt) for the covered components |

### External BODex solver

This checkout provides the [tabletop solver interface](docs/robots/grasping.md). Operators supply a compatible implementation and its assets separately. Earlier integrations referenced a port at revision `e2d4dfa`. Its file-by-file provenance could not be verified for this distribution.

The method is BODex; cite it when you use this grasp synthesis:

```bibtex
@article{chen2024bodex,
  title   = {BODex: Scalable and Efficient Robotic Dexterous Grasp Synthesis Using Bilevel Optimization},
  author  = {Chen, Jiayi and Ke, Yubin and Wang, He},
  journal = {arXiv preprint arXiv:2412.16490},
  year    = {2024}
}
```

Published at ICRA 2025. The upstream project identifies the cuRobo-derived code under the NVIDIA License and its other code under CC BY-NC 4.0. Its separate assets also carry conditions. See the [BODex license statement](https://github.com/JYChen18/BODex#license) and [upstream asset terms](https://github.com/JYChen18/BODex/blob/main/LICENSE_ASSETS).

### Installed cuRobo dependency

`setup/setup_grasp_env.sh` installs NVlabs/cuRobo at revision `a35a708ecfbb26eb9ab2d7ef22c65919c4fae4a9`. The [license at that pinned revision](https://github.com/NVlabs/curobo/blob/a35a708ecfbb26eb9ab2d7ef22c65919c4fae4a9/LICENSE) is Apache License 2.0.
