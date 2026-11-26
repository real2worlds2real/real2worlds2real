"""URDF assembly for the robot library: mount an end effector on a body's flange and make every mesh path absolute.

    python -m calibrated_scene_reconstruction.robots.urdf <body.urdf> [--ee <ee.urdf> --flange tool0 --prefix ee_ --xyz .. --rpy ..] -o out.urdf

Pure XML (no ROS): `package://<pkg>/...` resolves to the directory of the package.xml whose <name> is <pkg>, searched
upwards from the URDF and then across assets/robots; relative paths resolve against the URDF's directory. Link and
joint names become valid USD prim names (`link_0.0` -> `link_0_0`); a mimic joint whose multiplier is negative gets
limits below zero (Robotiq 2F-85 upstream has [0, u]). The end effector's links and joints get a prefix
(both descriptions usually have a `base_link`), and its root link is attached to the flange with a fixed joint, so the
Isaac URDF importer builds one articulation. `drop` removes links of the body first (with what they carry): a Panda's
own hand before a Robotiq 2F-85 goes on panda_link8.
"""
import argparse
import os
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

from calibrated_scene_reconstruction.core.environment import setting
from calibrated_scene_reconstruction.core.paths import ROBOTS_DIR, ROOT, robot_asset_path


@lru_cache(maxsize=None)
def _packages(search_root: str):
    pkgs = {}
    for xml in Path(search_root).rglob("package.xml"):
        try:
            name = ET.parse(xml).getroot().findtext("name")
        except ET.ParseError:
            continue
        if name:
            pkgs.setdefault(name.strip(), xml.parent)
    return pkgs


def resolve_mesh(filename: str, urdf_dir: Path) -> str:
    if filename.startswith("package://"):
        pkg, _, rest = filename[len("package://"):].partition("/")
        for d in [urdf_dir, *urdf_dir.parents]:
            xml = d / "package.xml"
            if xml.exists() and (ET.parse(xml).getroot().findtext("name") or "").strip() == pkg:
                return str(d / rest)
            if d == ROBOTS_DIR:
                break
        external = setting("ROBOT_ASSET_ROOT")
        base = _packages(str(Path(external).expanduser() / "robots")).get(pkg) if external else None
        if base is None:
            base = _packages(str(ROBOTS_DIR)).get(pkg)
        if base is None:
            raise FileNotFoundError(f"ROS package {pkg!r} is missing. Supply it under SCENE_RECONSTRUCTION_ROBOT_ASSET_ROOT/robots.")
        return str(base / rest)
    if filename.startswith("file://"):
        filename = filename[len("file://"):]
    p = Path(filename)
    return str(p if p.is_absolute() else (urdf_dir / p).resolve())


def usd_name(name: str) -> str:
    """USD prim names allow [A-Za-z0-9_] only (Allegro's `link_0.0` would make the importer write an empty stage)."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def load(urdf: Path) -> ET.Element:
    """Parse a URDF, make every <mesh filename> absolute and every link / joint name a valid USD prim name."""
    urdf = robot_asset_path(Path(urdf).resolve(), required=True)
    root = ET.parse(urdf).getroot()
    for mesh in root.iter("mesh"):
        mesh.set("filename", resolve_mesh(mesh.get("filename"), urdf.parent))
    for el in root.iter():
        if el.tag in ("link", "joint") and el.get("name"):
            el.set("name", usd_name(el.get("name")))
        elif el.tag in ("parent", "child") and el.get("link"):
            el.set("link", usd_name(el.get("link")))
        elif el.tag == "mimic" and el.get("joint"):
            el.set("joint", usd_name(el.get("joint")))
    for j in root.findall("joint"):  # upstream slips: a mimic with multiplier < 0 but limits [0, u] could never move
        m, lim = j.find("mimic"), j.find("limit")
        if m is not None and lim is not None and float(m.get("multiplier", 1)) < 0 and \
                float(lim.get("lower", 0)) >= 0 < float(lim.get("upper", 0)):
            lim.set("lower", str(-float(lim.get("upper"))))
            lim.set("upper", "0")
    return root


def root_link(robot: ET.Element) -> str:
    children = {j.find("child").get("link") for j in robot.findall("joint")}
    roots = [link.get("name") for link in robot.findall("link") if link.get("name") not in children]
    if len(roots) != 1:
        raise ValueError(f"expected one root link, found {roots}")
    return roots[0]


def _prefix(robot: ET.Element, prefix: str):
    """Prefix every link / joint name of an end effector (and the references to them)."""
    for link in robot.findall("link"):
        link.set("name", prefix + link.get("name"))
    for joint in robot.findall("joint"):
        joint.set("name", prefix + joint.get("name"))
        for tag in ("parent", "child"):
            el = joint.find(tag)
            el.set("link", prefix + el.get("link"))
        mimic = joint.find("mimic")
        if mimic is not None:
            mimic.set("joint", prefix + mimic.get("joint"))
    for tr in robot.findall("transmission"):  # not used by the importer; drop rather than rename
        robot.remove(tr)


def drop_links(robot: ET.Element, names) -> ET.Element:
    """Remove the named links, everything they carry and the joints to them (a body's own hand, before mounting
    another end effector on its flange)."""
    gone, todo = set(), list(names)
    while todo:
        n = todo.pop()
        if n in gone:
            continue
        gone.add(n)
        todo += [j.find("child").get("link") for j in robot.findall("joint") if j.find("parent").get("link") == n]
    for el in list(robot):
        if (el.tag == "link" and el.get("name") in gone) or (el.tag == "joint" and el.find("child").get("link") in gone):
            robot.remove(el)
    return robot


def mount(body: ET.Element, ee: ET.Element, flange: str, prefix="ee_", xyz=(0, 0, 0), rpy=(0, 0, 0)) -> ET.Element:
    if flange not in {link.get("name") for link in body.findall("link")}:
        raise ValueError(f"flange link {flange!r} not in {body.get('name')}")
    ee_root = root_link(ee)
    _prefix(ee, prefix)
    names = {el.get("name") for el in body if el.tag in ("link", "joint")}
    for el in ee:
        if el.tag in ("link", "joint"):
            if el.get("name") in names:
                raise ValueError(f"name clash after prefixing: {el.get('name')}")
            body.append(el)
        elif el.tag == "material" and el.get("name") not in {m.get("name") for m in body.findall("material")}:
            body.append(el)
    j = ET.SubElement(body, "joint", name=f"{prefix}mount", type="fixed")
    ET.SubElement(j, "parent", link=flange)
    ET.SubElement(j, "child", link=prefix + ee_root)
    ET.SubElement(j, "origin", xyz=" ".join(map(str, xyz)), rpy=" ".join(map(str, rpy)))
    body.set("name", f"{body.get('name')}__{ee.get('name')}")
    return body


def write(robot: ET.Element, out: Path):
    """Write a URDF; meshes inside the repository are referenced relative to the URDF's folder (the way URDF loaders
    and the Isaac importer resolve them), so built robots move with the repository."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    here = out.parent.resolve()
    for mesh in robot.iter("mesh"):
        f = mesh.get("filename") or ""
        if here.is_relative_to(ROOT) and Path(f).is_absolute() and Path(f).is_relative_to(ROOT):
            mesh.set("filename", os.path.relpath(f, here))
    ET.indent(robot)
    ET.ElementTree(robot).write(out, encoding="utf-8", xml_declaration=True)
    return out


def assemble(body_urdf, out, ee_urdf=None, flange=None, prefix="ee_", xyz=(0, 0, 0), rpy=(0, 0, 0), drop=()):
    robot = drop_links(load(body_urdf), [usd_name(n) for n in drop])
    if ee_urdf:
        robot = mount(robot, load(ee_urdf), flange, prefix, xyz, rpy)
    missing = [m.get("filename") for m in robot.iter("mesh") if not Path(m.get("filename")).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} meshes missing, e.g. {missing[:3]}")
    return write(robot, out)


def mimics(urdf):
    """{joint: (master, multiplier, offset)} of a (built) URDF."""
    out = {}
    for j in ET.parse(urdf).getroot().findall("joint"):
        m = j.find("mimic")
        if m is not None:
            out[j.get("name")] = (m.get("joint"), float(m.get("multiplier", 1)), float(m.get("offset", 0)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("body")
    ap.add_argument("--ee")
    ap.add_argument("--flange")
    ap.add_argument("--prefix", default="ee_")
    ap.add_argument("--xyz", nargs=3, type=float, default=(0, 0, 0))
    ap.add_argument("--rpy", nargs=3, type=float, default=(0, 0, 0))
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    out = assemble(a.body, a.out, a.ee, a.flange, a.prefix, a.xyz, a.rpy)
    r = ET.parse(out).getroot()
    movable = [j.get("name") for j in r.findall("joint") if j.get("type") not in ("fixed",)]
    print(f"{out}: {len(r.findall('link'))} links, {len(movable)} movable joints")


if __name__ == "__main__":
    main()
