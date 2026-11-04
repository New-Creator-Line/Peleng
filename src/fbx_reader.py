"""
Minimal binary FBX reader (no external dependencies).

Used to extract the truck mesh vertices from ``data/assets/CargoTruck.fbx`` so the
3D box dimensions and the actor pivot offset (cabin vs. body center) come from the
real 3D model instead of being fitted against ground truth.
"""

import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FbxNode:
    name: str
    props: List[Any] = field(default_factory=list)
    children: List["FbxNode"] = field(default_factory=list)

    def find(self, name: str) -> Optional["FbxNode"]:
        for c in self.children:
            if c.name == name:
                return c
        return None

    def find_all(self, name: str) -> List["FbxNode"]:
        return [c for c in self.children if c.name == name]


_ARRAY_TYPES = {"f": ("f", 4), "d": ("d", 8), "l": ("q", 8), "i": ("i", 4), "b": ("?", 1)}


def _read_prop(data: bytes, pos: int) -> Tuple[Any, int]:
    t = chr(data[pos])
    pos += 1
    if t == "Y":
        return struct.unpack_from("<h", data, pos)[0], pos + 2
    if t == "C":
        return bool(data[pos]), pos + 1
    if t == "I":
        return struct.unpack_from("<i", data, pos)[0], pos + 4
    if t == "F":
        return struct.unpack_from("<f", data, pos)[0], pos + 4
    if t == "D":
        return struct.unpack_from("<d", data, pos)[0], pos + 8
    if t == "L":
        return struct.unpack_from("<q", data, pos)[0], pos + 8
    if t in _ARRAY_TYPES:
        length, encoding, comp_len = struct.unpack_from("<III", data, pos)
        pos += 12
        raw = data[pos:pos + comp_len]
        pos += comp_len
        if encoding == 1:
            raw = zlib.decompress(raw)
        fmt, _ = _ARRAY_TYPES[t]
        return np.frombuffer(raw, dtype=np.dtype("<" + fmt), count=length).copy(), pos
    if t in ("S", "R"):
        length = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        raw = data[pos:pos + length]
        pos += length
        if t == "S":
            return raw.decode("utf-8", errors="replace"), pos
        return raw, pos
    raise ValueError(f"Unknown FBX property type '{t}' at {pos - 1}")


def _read_node(data: bytes, pos: int, wide: bool) -> Tuple[Optional[FbxNode], int]:
    if wide:
        end_offset, num_props, _plen = struct.unpack_from("<QQQ", data, pos)
        pos += 24
    else:
        end_offset, num_props, _plen = struct.unpack_from("<III", data, pos)
        pos += 12
    name_len = data[pos]
    pos += 1
    if end_offset == 0:
        return None, pos
    name = data[pos:pos + name_len].decode("ascii", errors="replace")
    pos += name_len
    node = FbxNode(name)
    for _ in range(num_props):
        val, pos = _read_prop(data, pos)
        node.props.append(val)
    while pos < end_offset:
        child, pos = _read_node(data, pos, wide)
        if child is None:
            break
        node.children.append(child)
    return node, end_offset


def read_fbx(path: str) -> FbxNode:
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(b"Kaydara FBX Binary"):
        raise ValueError("Only binary FBX files are supported")
    version = struct.unpack_from("<I", data, 23)[0]
    wide = version >= 7500
    root = FbxNode("root")
    pos = 27
    while pos < len(data):
        node, pos = _read_node(data, pos, wide)
        if node is None:
            break
        root.children.append(node)
    return root


def _properties70(node: Optional[FbxNode]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    if node is None:
        return out
    p70 = node.find("Properties70")
    if p70 is None:
        return out
    for p in p70.find_all("P"):
        if p.props:
            out[str(p.props[0])] = p.props[4:]
    return out


def _euler_xyz_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """FBX default rotation order eEulerXYZ: R = Rz @ Ry @ Rx (degrees)."""
    ax, ay, az = np.radians([rx, ry, rz])
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def load_fbx_scene_vertices(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Loads every mesh of the FBX scene and applies its Model local transform
    (Lcl Scaling -> Lcl Rotation -> Lcl Translation), following parent links.

    Returns (all_vertices (N, 3) in FBX scene space, per-model vertex dict).
    """
    root = read_fbx(path)
    objects = root.find("Objects")
    conns = root.find("Connections")
    if objects is None:
        return np.zeros((0, 3)), {}

    geoms: Dict[int, np.ndarray] = {}
    for geom in objects.find_all("Geometry"):
        v = geom.find("Vertices")
        if v is not None and v.props:
            geoms[int(geom.props[0])] = np.asarray(v.props[0], dtype=np.float64).reshape(-1, 3)

    models: Dict[int, Dict[str, Any]] = {}
    for model in objects.find_all("Model"):
        p = _properties70(model)
        t = np.array(p.get("Lcl Translation", [0, 0, 0])[:3], dtype=np.float64)
        r = np.array(p.get("Lcl Rotation", [0, 0, 0])[:3], dtype=np.float64)
        s = np.array(p.get("Lcl Scaling", [1, 1, 1])[:3], dtype=np.float64)
        M = np.eye(4)
        M[:3, :3] = _euler_xyz_matrix(*r) @ np.diag(s)
        M[:3, 3] = t
        name = str(model.props[1]).split("\x00")[0] if len(model.props) > 1 else str(model.props[0])
        models[int(model.props[0])] = {"name": name, "M": M}

    geom_to_model: Dict[int, int] = {}
    parent: Dict[int, int] = {}
    if conns is not None:
        for c in conns.find_all("C"):
            if len(c.props) >= 3 and c.props[0] == "OO":
                child, par = int(c.props[1]), int(c.props[2])
                if child in geoms and par in models:
                    geom_to_model[child] = par
                elif child in models:
                    parent[child] = par

    def world_matrix(mid: int) -> np.ndarray:
        M = models[mid]["M"]
        p = parent.get(mid)
        while p is not None and p in models:
            M = models[p]["M"] @ M
            p = parent.get(p)
        return M

    per_model: Dict[str, np.ndarray] = {}
    for gid, v in geoms.items():
        mid = geom_to_model.get(gid)
        if mid is None:
            per_model[f"geom_{gid}"] = v
            continue
        M = world_matrix(mid)
        vh = np.hstack([v, np.ones((len(v), 1))]) @ M.T
        per_model[models[mid]["name"]] = vh[:, :3]

    all_v = np.vstack(list(per_model.values())) if per_model else np.zeros((0, 3))
    return all_v, per_model


def load_fbx_vertices(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Returns all mesh vertices (N, 3) in FBX file coordinates and global settings
    (UpAxis, FrontAxis, CoordAxis, UnitScaleFactor). Model local transforms
    (Lcl Translation / Rotation / Scaling) are reported in ``info['models']``.
    """
    root = read_fbx(path)
    settings = _properties70(root.find("GlobalSettings"))
    info: Dict[str, Any] = {k: settings[k][0] for k in (
        "UpAxis", "UpAxisSign", "FrontAxis", "FrontAxisSign",
        "CoordAxis", "CoordAxisSign", "UnitScaleFactor") if k in settings and settings[k]}

    objects = root.find("Objects")
    verts: List[np.ndarray] = []
    models = []
    if objects is not None:
        for geom in objects.find_all("Geometry"):
            v = geom.find("Vertices")
            if v is not None and v.props:
                verts.append(np.asarray(v.props[0], dtype=np.float64).reshape(-1, 3))
        for model in objects.find_all("Model"):
            p = _properties70(model)
            models.append({
                "name": model.props[1] if len(model.props) > 1 else "",
                "translation": p.get("Lcl Translation"),
                "rotation": p.get("Lcl Rotation"),
                "scaling": p.get("Lcl Scaling"),
            })
    info["models"] = models
    all_v = np.vstack(verts) if verts else np.zeros((0, 3))
    return all_v, info
