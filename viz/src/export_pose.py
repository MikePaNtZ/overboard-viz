#!/usr/bin/env python3
"""Read the MuJoCo model and emit the board's rest pose as world-space geom
transforms — the one file that crosses from the controls repo into this one.

This is V1.0's stand-in for the pose track specified in V1 §5: a single frame
instead of N, but already carrying the conventions that matter (SI metres,
+Z up, scalar-first quaternions, body-origin semantics), so that promoting it
to the real `.otrk` at V1.2 is an extension rather than a rewrite.

Why go through MuJoCo at all, rather than hardcoding transforms in the Blender
script? Because then the render is provably of the model the metrics describe.
Every geom placement, every colour, and the tyre's dimensions are read out of
`overboard_onewheel.xml` as committed — if the model moves, the render moves,
and nobody has to remember to update two places.

Run with the controls repo's venv, which has mujoco:

    ~/projects/overboard/.venv/bin/python viz/src/export_pose.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

DEFAULT_MODEL = Path.home() / "projects/overboard/sim/models/overboard_onewheel.xml"
ROOT = Path(__file__).resolve().parents[2]


def _quat(xmat: np.ndarray) -> list[float]:
    """MuJoCo's row-major 3x3 -> unit quaternion, scalar-first (w,x,y,z).

    Blender's Object.rotation_quaternion uses the same order and the same
    right-handed +Z-up frame, so this value is copied across with no
    conversion at all. That 1:1 mapping is most of why this pipeline is cheap,
    and it is asserted in V1 §5.2 precisely so nobody "helpfully" adds an axis
    swap later.
    """
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, xmat.flatten())
    return [float(v) for v in q]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=ROOT / "viz/scenes/board_rest_pose.json")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)  # rest pose: wheel in contact, board upright

    mesh_dir = args.model.parent / "meshes" / "openwheel"
    geoms = []

    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name == "ground":
            continue  # the render supplies its own bench and floor

        gtype = model.geom_type[gid]
        entry = {
            "name": name,
            "pos": [float(v) for v in data.geom_xpos[gid]],
            "quat": _quat(data.geom_xmat[gid]),
        }

        mat_id = model.geom_matid[gid]
        if mat_id >= 0:
            entry["material"] = {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id),
                # MuJoCo rgba is linear, which is also what Blender's Base Color
                # wants. No sRGB conversion — doing one would wash the palette out.
                "rgba": [float(v) for v in model.mat_rgba[mat_id]],
            }

        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)

            # MuJoCo does NOT keep mesh vertices as authored: on load it
            # recentres them on the mesh centre of mass and rotates them onto
            # the inertial axes, recording what it did in mesh_pos/mesh_quat.
            # So geom_xpos/geom_xmat place the *canonical* mesh, not the STL on
            # disk — visible here as a ~90° rotation on parts the MJCF gives no
            # rotation at all. Blender imports the raw STL, so that transform
            # has to be composed back out or every shell lands rotated and
            # offset:
            #     v_world = geom_x ∘ canonical(v_raw),  canonical = Rm⁻¹(v - pm)
            # ⇒   R = Rg·Rmᵀ ,  t = geom_xpos − R·pm
            Rg = data.geom_xmat[gid].reshape(3, 3)
            Rm = np.zeros(9)
            mujoco.mju_quat2Mat(Rm, model.mesh_quat[mid])
            R = Rg @ Rm.reshape(3, 3).T
            t = data.geom_xpos[gid] - R @ model.mesh_pos[mid]

            entry["pos"] = [float(v) for v in t]
            entry["quat"] = _quat(R)
            entry.update(
                kind="mesh",
                mesh=mesh_name,
                # Authored in millimetres in a shared assembly frame centred on
                # the axle; the MJCF scales them 0.001 at load and so must we.
                file=str(mesh_dir / f"{mesh_name}.stl"),
                scale=0.001,
            )
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, half_w = float(model.geom_size[gid][0]), float(model.geom_size[gid][1])
            entry.update(kind="cylinder", radius=r, half_width=half_w)
        else:
            continue

        geoms.append(entry)

    out = {
        "schema_version": "0.1-v1.0-still",
        "note": "Single-frame precursor to the .otrk pose track (V1 §5). Same conventions, one frame.",
        "conventions": {
            "units": "SI metres",
            "world_frame": "right-handed, +Z up",
            "forward": "-X",
            "quaternion": "(w, x, y, z), unit, body->world",
        },
        "source": {
            "model_file": str(args.model),
            "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "mujoco_version": mujoco.__version__,
        },
        "geoms": geoms,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    top = max(g["pos"][2] for g in geoms)
    print(f"wrote {args.out}  ({len(geoms)} geoms, highest at z={top:.3f} m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
