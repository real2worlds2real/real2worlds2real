# Robot models and external assets

The source archive includes ten robot or end-effector entries. The [catalogue](manifest.json) also lists compatible external entries. Each included entry records its source and available notices. The bundled Apache-licensed models use the [full Apache 2.0 terms](licenses/Apache-2.0.txt) together with their individual notices.

| Included model | Contents | Terms |
|---|---|---|
| Franka Panda | Standard-hand URDF, meshes and MuJoCo Menagerie model | [Apache 2.0](arms/franka_panda/LICENSE), [Franka NOTICE](arms/franka_panda/NOTICE), [Menagerie license](arms/franka_panda/mjcf_menagerie/LICENSE) |
| Universal Robots | UR10e URDF, UR10e and UR5e meshes | [BSD 3-Clause](arms/ur/LICENSE) |
| Unitree G1 | G1 URDFs and manufacturer meshes | [BSD 3-Clause](humanoids/unitree_g1/LICENSE) |
| Dexmate | Manufacturer's Vega bodies and grippers, URDF and meshes | [Apache 2.0](humanoids/dexmate/LICENSE) |
| Sharpa Wave 01 | Left and right HB1 hand URDFs and meshes | [Sharpa license](hands/sharpa_wave_01/LICENSE.txt), [NOTICE](hands/sharpa_wave_01/NOTICE.txt), [mesh path changes](hands/sharpa_wave_01/ATTRIBUTION.md) |
| Vega 1P + Sharpa assembly | The simulation robot with welded wheels, both hands and PhysX drives | [Assembly provenance and component notices](assemblies/vega_1p_sharpa/ATTRIBUTION.md) |
| Wuji Hand | First-generation model, URDF, MJCF and USD | [MIT](hands/wuji_hand/LICENSE) |
| Wuji Hand 2 Beta 2 | URDF, MJCF, USD and accompanying CAD | [MIT](hands/wuji_hand2_beta2/LICENSE), [source revision](hands/wuji_hand2_beta2/provenance.json) |
| Barrett BH8 | Original ROS description with xacro and meshes | [BSD 3-Clause](hands/barrett_bh8_280/LICENSE.vendor) |
| Robotiq 2F-85 | MuJoCo Menagerie MJCF and its meshes | [BSD 2-Clause](grippers/robotiq_2f85/mjcf_menagerie/LICENSE) |

For Wuji Hand 2 Beta 2 outside ROS, use `body/urdf/left.urdf` or `body/urdf/right.urdf` inside the model directory. The vendor's `*-ros.urdf` variants use the ROS package name `wuji_hand2_beta2_description` and require that package in a ROS workspace.

## Use the standard Franka model

From the repository root, activate the installed environment and run `source env.sh`. The command below resolves the packaged URDF and its meshes and writes an assembled URDF. It uses the Python standard library.

```sh
python -m calibrated_scene_reconstruction.robots.urdf \
  real2sim/assets/robots/arms/franka_panda/franka_panda.urdf \
  -o /tmp/astra-franka.urdf
```

## Use the default Vega and Sharpa robot

Open [vega_1p_sharpa.usd](assemblies/vega_1p_sharpa/vega_1p_sharpa.usd) in Isaac Sim 5.1. The Vega simulation profiles use this asset directly from the checkout. To select another compatible assembly, set `SCENE_RECONSTRUCTION_VEGA_USD` to its root USD. Keep its configuration layers together.

## Use a separately obtained model

For a custom robot, provide a robot [spec](../../src/calibrated_scene_reconstruction/robots/spec.py) with an absolute USD path or URDF body path. The robot's owner or supplier can provide compatible model files and their terms. Existing Vega, Dex3, Galaxea R1 and Robotiq URDF presets also accept local inputs. Set `SCENE_RECONSTRUCTION_ROBOT_ASSET_ROOT` to the directory containing the `robots/` and `grasping/` trees, with layout as shown below. Preserve the model's relative mesh and USD layer references.

```text
operator-assets/
  grasping/hands/sharpa_wave/sharpa_wave_right.yml
  grasping/hands/sharpa_wave/{urdf,collision,contact_points,grasp_synthesis}/...
  grasping/hands/unitree_dex3/unitree_dex3_right.yml
  grasping/hands/unitree_dex3/{urdf,collision,grasp_synthesis}/...
  robots/grippers/robotiq_2f85/urdf/robots/robotiq_arg85_description.urdf
  robots/grippers/robotiq_2f85/urdf/{meshes,package.xml}
  robots/mobile/galaxea_r1/r1_source.urdf
```

```sh
export SCENE_RECONSTRUCTION_ROBOT_ASSET_ROOT=/absolute/path/to/operator-assets
```

Only supply the inputs for the profile you use. For the optional grasp-synthesis path, also provide the matching hand descriptor, collision and calibration files, plus the separately installed synthesis backend. A standalone vendor hand is a different model from the calibrated hand assembly in that profile. The standard UR10e/Panda plus Robotiq presets require the additional ROS URDF gripper package. The plain Franka profile uses the included hand.

A matching file in the operator directory takes precedence over its counterpart in the source tree. Repo-relative paths and the historical `R2S2R_VEGA_USD` setting remain supported. Generated USDs from `apps/robot_build.py` go in the checkout's ignored `assets/robots/built/` directory. The [task examples](../../../examples/README.md) are self-contained.
