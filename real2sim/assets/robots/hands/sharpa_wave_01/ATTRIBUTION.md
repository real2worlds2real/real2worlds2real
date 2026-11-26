# SharpaWave hand assets — attribution

The `right/` and `left/` URDFs and meshes here are derived from Sharpa's
official robot description, **© 2025 Sharpa Group, Apache License 2.0**
(see `LICENSE.txt` / `NOTICE.txt`).

The descriptions and meshes correspond to the [Sharpa Wave 01 vendor tree](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/tree/d2db0b8ba4aeb737e6274d4c4386e210eedfa1da/wave_01).

The only modification vs. upstream: mesh references were rewritten from the ROS
package form `package://<side>_sharpa_wave/meshes/...` to repo-relative
`meshes/...` so SAPIEN/pinocchio resolve them without a ROS package path.

The [complete Apache License 2.0](../../licenses/Apache-2.0.txt) and the
vendor's [notice](NOTICE.txt) accompany these files.
