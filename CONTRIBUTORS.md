# Contributors and provenance

## Paper authors

Jinzhong Gu<sup>1,\*</sup>, Jiakai Chen<sup>3,\*</sup>, Xinyi Cao<sup>4,\*</sup>, Yan Shen<sup>1,\*</sup>,
Hangke Sui<sup>3,\*</sup>, Shutian Zheng<sup>1</sup>, Xiang Xiao<sup>2</sup>, Haoran Lu<sup>1</sup>, Fan Xu<sup>1</sup>,
Lu Shi<sup>2</sup>, Daochen Shi<sup>2</sup>, Ruihua Zhang<sup>2</sup>, Wei Zhan<sup>1</sup>,
Masayoshi Tomizuka<sup>1</sup>, Ruihai Wu<sup>1</sup>

<sup>1</sup>University of California, Berkeley · <sup>2</sup>NVIDIA · <sup>3</sup>University of Illinois Urbana-Champaign ·
<sup>4</sup>Carnegie Mellon University · <sup>\*</sup>Equal contribution

## Source history

| Public component | Source before the public release |
|---|---|
| Real2Sim Harness (`real2sim_harness`) | The object-generation and task-workspace harness formerly imported as `ucb_remo`. Its command names included `ucb-remo` and `real2sim`. |
| Calibrated Scene Reconstruction (`calibrated_scene_reconstruction`) | The metric scene pipeline from `r2s2r_datagen`, formerly imported as `r2s2r`. Capture, calibration, reconstruction, native checks and delivery retain the component boundaries in the [architecture guide](real2sim/docs/architecture.md). |
| Policy station (`astra_station`) | The task-independent station controller and policy-development tools. |
| End-to-end workflow (`astra_e2e`) | The oversight, integration and chain code that joins scene delivery to station evaluation and the physical-validation handover. |

The optional tabletop grasp solver interface follows the BODex method by Jiayi Chen, Yubin Ke and He Wang. See the [component notices](real2sim/THIRD_PARTY.md) and [third-party licenses](THIRD_PARTY.md).

## Compatibility for existing integrations

Installed releases keep the previous `ucb_remo` and `r2s2r` imports, including dotted submodules and Python objects saved with those paths. Existing `ucb-remo`, `real2sim`, and `real2sim-auth` invocations continue to work. The public CLI and Python examples use the new names. Older Blender metadata is read in memory without editing delivered assets. Frozen records retain their schema IDs, evidence and hashes.

Reconstruction settings use the `SCENE_RECONSTRUCTION_` prefix. For existing jobs, a setting also accepts its previous `R2S2R_` name when the new variable is unset.
End-to-end workers receive the oversight run and plan hash as `ASTRA_E2E_OVERSIGHT_RUN` and `ASTRA_E2E_OVERSIGHT_PLAN_SHA256`. Existing workers also receive their original variable names. The USD attribute namespaces and registered challenge task IDs used by delivered assets retain their wire names.

## Citation

```bibtex
@misc{real2worlds2real,
  title         = {Real2Worlds2Real: Scaling Self-Improving Agentic Real2Sim2Real for Robot Manipulation},
  author        = {Gu, Jinzhong and Chen, Jiakai and Cao, Xinyi and Shen, Yan and Sui, Hangke and Zheng, Shutian and
                   Xiao, Xiang and Lu, Haoran and Xu, Fan and Shi, Lu and Shi, Daochen and Zhang, Ruihua and Zhan, Wei and
                   Tomizuka, Masayoshi and Wu, Ruihai},
  year          = {2026},
  eprint        = {2611.31098},
  archivePrefix = {arXiv}
}
```
