#!/usr/bin/env python3
"""Run the impulse scenario and emit a pose track — the real `.otrk` from ICD §5.

This is the seam the whole V1 epic is built around. `overboard` computes the
physics and writes this file; `overboard-viz` reads it and makes pictures.
Neither imports the other.

Crucially the track is a *replay* of the captured `qpos` history rather than a
re-simulation, so the film is provably of the same run the metrics describe —
and the entire class of "our renderer's physics disagrees with MuJoCo" bugs
cannot exist.

Conventions are frozen in ICD §5.2 and honoured here: SI metres, right-handed
+Z up, forward = −X, scalar-first (w,x,y,z) quaternions, body-origin poses.

Run with the controls repo's venv, which has mujoco and the scenario package:

    ~/projects/overboard/.venv/bin/python viz/src/export_track.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OVERBOARD = Path.home() / "projects/overboard"
MESH_DIR = OVERBOARD / "sim/models/meshes/openwheel"
sys.path.insert(0, str(OVERBOARD))

import mujoco  # noqa: E402
from sim.scenarios.impulse_response import ImpulseParams, load_model, run  # noqa: E402
from sim.scenarios.rust_controller import RustController  # noqa: E402
from sim.scenarios import shuttle_run as shuttle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# Bodies are discovered from the model rather than hardcoded: the shuttle-run
# plant adds a `ballast` body carrying the rider proxy, and a fixed list would
# silently drop it — the failure mode being a film of a board doing something
# it can only do with a mass it is not showing.
def _bodies(model) -> list:
    out = []
    for gid in range(model.ngeom):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) == "ground":
            continue
        b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])
        if b and b not in out:
            out.append(b)
    return out


def _quat_from_mat(xmat) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(xmat).flatten())
    return q


def _bindings(model, data) -> list:
    """Where each visible geom sits *relative to its body*, computed once.

    The track carries bodies; this carries the mesh-to-body binding, which
    belongs on the render side (ICD §5.6). It is computed rather than
    hardcoded because it has to undo MuJoCo's mesh recentring — on load
    MuJoCo moves mesh vertices onto the mesh centre of mass and rotates them
    onto the inertial axes, recording the transform in mesh_pos/mesh_quat. So
    geom_xpos/geom_xmat place the *canonical* mesh, not the STL on disk.
    """
    out = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name == "ground":
            continue

        gtype = model.geom_type[gid]
        bid = model.geom_bodyid[gid]
        Rb = data.xmat[bid].reshape(3, 3)
        pb = data.xpos[bid]

        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
            Rm = np.zeros(9)
            mujoco.mju_quat2Mat(Rm, model.mesh_quat[mid])
            Rw = data.geom_xmat[gid].reshape(3, 3) @ Rm.reshape(3, 3).T
            pw = data.geom_xpos[gid] - Rw @ model.mesh_pos[mid]
            entry = dict(kind="mesh", mesh=mesh,
                         file=str(MESH_DIR / f"{mesh}.stl"), scale=0.001)
        elif gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER,
                       mujoco.mjtGeom.mjGEOM_CAPSULE,
                       mujoco.mjtGeom.mjGEOM_SPHERE):
            Rw = data.geom_xmat[gid].reshape(3, 3)
            pw = data.geom_xpos[gid]
            size = model.geom_size[gid]
            if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                entry = dict(kind="sphere", radius=float(size[0]))
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                # MuJoCo capsules are Z-aligned in their own frame: size[0] is
                # the radius, size[1] the half-length of the straight section.
                entry = dict(kind="capsule", radius=float(size[0]),
                             half_length=float(size[1]))
            else:
                entry = dict(kind="cylinder", radius=float(size[0]),
                             half_width=float(size[1]))
        else:
            continue

        mat_id = model.geom_matid[gid]
        entry.update(
            name=name,
            body=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid),
            pos=[float(v) for v in Rb.T @ (pw - pb)],       # world -> body-local
            quat=[float(v) for v in _quat_from_mat(Rb.T @ Rw)],
            # Fall back to the geom's own rgba when it has no material. The
            # rider proxy sets colour directly on each capsule, so a
            # material-only path renders the whole figure default grey — which
            # loses the one signal that says "abstraction, not a person".
            material=(dict(
                name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id),
                rgba=[float(v) for v in model.mat_rgba[mat_id]],
            ) if mat_id >= 0 else dict(
                name=f"geom_{name}",
                rgba=[float(v) for v in model.geom_rgba[gid]],
            )),
        )
        out.append(entry)
    return out


def _export_shuttle(args) -> int:
    """The commanded-velocity route, on the RIDDEN plant.

    Separate from the impulse path because it is a genuinely different vehicle:
    `plant.build_model()` bolts a 70 kg ballast 0.75 m above the axle, carrying
    the stylised rider proxy. The scenario **refuses** the driverless plant, and
    the reason matters for anything published from it — below the axle the
    pitch->velocity coupling inverts and the outer loop's speed authority
    collapses ~200x. So this footage cannot honestly be shown as an empty
    board: the mass is load-bearing, and the proxy is how it stays visible.
    """
    from sim.scenarios.plant import build_model

    name = "cruise" if getattr(args, "cruise", False) else "shuttle_run"
    if getattr(args, "cruise", False):
        # A single long leg. Same scenario, same cascade, same plant — only the
        # commanded route differs, so nothing about its validity changes. The
        # out-and-back exists to prove reversal and return-to-home; a cruise
        # exists because a trail only reads as a trail when it recedes, and
        # that needs sustained travel in one direction.
        params = shuttle.ShuttleParams(
            route=(shuttle.Leg("settle", hold_s=0.8),
                   shuttle.Leg("cruise", distance_m=+args.cruise_m),
                   shuttle.Leg("stop", hold_s=2.5)),
            cruise_m_s=args.cruise_speed,
        )
    else:
        params = shuttle.ShuttleParams()
    print(f"shuttle_run: {params.ballast_mass_kg} kg ballast at "
          f"{params.ballast_height_m} m, cruise {params.cruise_m_s} m/s …")
    result = shuttle.run(params, capture_state=True)

    m = result.metrics
    for f in ("return_error_m", "max_creep_m", "peak_abs_pitch_deg", "held"):
        if hasattr(m, f):
            print(f"  {f} = {getattr(m, f)}")

    # Rebuilt with the same arguments run() used, so the geometry the poses are
    # computed against is the geometry the physics ran on.
    model = build_model(params.ballast_mass_kg, params.ballast_height_m,
                        params.max_current_a)

    dt = float(model.opt.timestep)
    stride = max(1, int(round((1.0 / args.fps) / dt)))
    last = len(result.qpos) if args.seconds <= 0 else min(
        len(result.qpos), int(args.seconds / dt) + 1)
    idx = np.arange(0, last, stride)
    n = len(idx)

    BODIES = _bodies(model)
    data = mujoco.MjData(model)
    bid = {b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in BODIES}
    pos = {b: np.zeros((n, 3), np.float32) for b in BODIES}
    quat = {b: np.zeros((n, 4), np.float32) for b in BODIES}

    bindings = None
    for k, i in enumerate(idx):
        data.qpos[:] = result.qpos[i]
        mujoco.mj_forward(model, data)
        if bindings is None:
            bindings = _bindings(model, data)
        for b in BODIES:
            pos[b][k] = data.xpos[bid[b]]
            quat[b][k] = data.xquat[bid[b]]

    t = (idx * dt).astype(np.float64)

    def _sample(a):
        a = np.asarray(a, np.float32)
        return a[idx] if a.size > idx[-1] else np.zeros(n, np.float32)

    channels = {
        "pos_m": _sample(result.pos_m),
        "pos_cmd_m": _sample(result.pos_cmd_m),
        "v_m_s": _sample(result.v_m_s),
        "v_ref_m_s": _sample(result.v_ref_m_s),
        "pitch_rad": np.radians(_sample(result.pitch_deg)),
        "motor_current_a": _sample(result.motor_current_a),
    }

    # Leg boundaries drive camera timing and let a caption name what is on
    # screen without anyone re-deriving the route by eye.
    events = []
    for t0, t1, _v0, _v1, label in shuttle.VelocityProfile(params).segments:
        events.append(dict(t=float(t0), kind="leg", note=label))

    manifest = {
        "schema_version": "1.0",
        "source": {
            "kind": "sim",
            "scenario": name,
            "plant": "ridden",
            "ballast_mass_kg": params.ballast_mass_kg,
            "ballast_height_m": params.ballast_height_m,
            "model_file": "built by sim.scenarios.plant.build_model()",
            "mujoco_version": mujoco.__version__,
            "exporter_version": "otrk-export 1.0.0",
        },
        "time": {"fps": args.fps, "n_frames": n, "duration_s": float(t[-1])},
        "bodies": BODIES,
        "channels": list(channels),
        "events": events,
        "hints": {"subject_body": "frame", "ground_z": 0.0},
        "conventions": {
            "units": "SI metres", "world_frame": "right-handed, +Z up",
            "forward": "-X", "quaternion": "(w, x, y, z), unit, body->world",
        },
        "bindings": bindings,
    }

    arrays = {"manifest": json.dumps(manifest), "t": t}
    for b in BODIES:
        arrays[f"pos/{b}"] = pos[b]
        arrays[f"quat/{b}"] = quat[b]
    arrays.update({f"ch/{k}": v for k, v in channels.items()})

    out = args.out or ROOT / f"viz/scenes/{name}.otrk.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"\nwrote {out.name}: {n} frames @ {args.fps}fps ({t[-1]:.2f}s), "
          f"bodies={BODIES}")
    print(f"  size {out.stat().st_size / 1024:.0f} KiB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=6.0,
                    help="clip length; <=0 means the whole run")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--closed-loop", action="store_true",
                    help="attach the real Rust PitchRegulator over the C ABI")
    ap.add_argument("--shuttle", action="store_true",
                    help="commanded-velocity route on the RIDDEN plant")
    ap.add_argument("--cruise", action="store_true",
                    help="one long forward leg instead of the out-and-back route")
    ap.add_argument("--cruise-m", type=float, default=18.0)
    ap.add_argument("--cruise-speed", type=float, default=1.2)
    args = ap.parse_args()

    if args.shuttle or args.cruise:
        return _export_shuttle(args)

    model = load_model()
    params = ImpulseParams()
    kind = "closed_loop" if args.closed_loop else "impulse_response"
    out = args.out or ROOT / f"viz/scenes/{kind}.otrk.npz"

    print(f"{kind}: {params.magnitude_ns} N·s at t={params.t0_s}s, "
          f"{params.sim_seconds}s sim …")

    if args.closed_loop:
        # The same disturbance as the open-loop clip, deliberately: the two
        # films are only comparable if the shove is identical. The controller
        # is the ONLY difference between them.
        with RustController() as controller:
            result = run(params, model=model, controller=controller,
                         capture_state=True)
    else:
        result = run(params, model=model, capture_state=True)

    m = result.metrics
    for f in ("nose_strike", "toppled", "peak_abs_pitch_deg", "peak_pitch_deg",
              "t_strike_s", "settle_time_s", "travel_m"):
        if hasattr(m, f):
            print(f"  {f} = {getattr(m, f)}")

    dt = float(model.opt.timestep)
    stride = max(1, int(round((1.0 / args.fps) / dt)))
    last = min(len(result.qpos), int(args.seconds / dt) + 1)
    idx = np.arange(0, last, stride)
    n = len(idx)

    BODIES = _bodies(model)
    data = mujoco.MjData(model)
    bid = {b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in BODIES}
    pos = {b: np.zeros((n, 3), np.float32) for b in BODIES}
    quat = {b: np.zeros((n, 4), np.float32) for b in BODIES}

    bindings = None
    for k, i in enumerate(idx):
        data.qpos[:] = result.qpos[i]
        mujoco.mj_forward(model, data)
        if bindings is None:  # frame 0 is at rest; bindings are rigid anyway
            bindings = _bindings(model, data)
        for b in BODIES:
            pos[b][k] = data.xpos[bid[b]]
            quat[b][k] = data.xquat[bid[b]]

    t = (idx * dt).astype(np.float64)

    def _sample(a):
        a = np.asarray(a, np.float32)
        return a[idx] if a.size > idx[-1] else np.zeros(n, np.float32)

    # Free to export, expensive to regenerate. Nothing in V1 renders these —
    # overlays are cut (§9) — but exporting them means a future overlay, plot
    # or web viewer never forces a re-export. ICD §5.5.
    channels = {
        "pitch_rad": np.radians(_sample(result.pitch_deg)),
        "wheel_rate_rads": _sample(result.wheel_rate_rads),
        "motor_current_a": _sample(result.motor_current_a),
        "travel_m": _sample(result.travel_m),
    }

    # Events drive camera timing, not on-screen overlays.
    events = [dict(t=float(params.t0_s), kind="impulse_strike",
                   note=f"{params.magnitude_ns} N·s along {params.direction}")]
    if getattr(m, "t_strike_s", None):
        events.append(dict(t=float(m.t_strike_s), kind="bumper_contact",
                           note="nose bumper meets ground"))

    manifest = {
        "schema_version": "1.0",
        "source": {
            "kind": "sim",
            "scenario": kind,
            "model_file": "sim/models/overboard_onewheel.xml",
            "model_sha256": hashlib.sha256(
                (OVERBOARD / "sim/models/overboard_onewheel.xml").read_bytes()).hexdigest(),
            "mujoco_version": mujoco.__version__,
            "exporter_version": "otrk-export 1.0.0",
        },
        "time": {"fps": args.fps, "n_frames": n, "duration_s": float(t[-1])},
        "bodies": BODIES,
        "channels": list(channels),
        "events": events,
        "hints": {"subject_body": "frame", "ground_z": 0.0},
        "conventions": {
            "units": "SI metres", "world_frame": "right-handed, +Z up",
            "forward": "-X", "quaternion": "(w, x, y, z), unit, body->world",
        },
        "bindings": bindings,
    }

    arrays = {"manifest": json.dumps(manifest), "t": t}
    for b in BODIES:
        arrays[f"pos/{b}"] = pos[b]
        arrays[f"quat/{b}"] = quat[b]
    arrays.update({f"ch/{k}": v for k, v in channels.items()})

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)

    travel = float(pos["frame"][-1][0] - pos["frame"][0][0])
    drop = float(pos["frame"][:, 2].min() - pos["frame"][0][2])
    print(f"\nwrote {out.name}: {n} frames @ {args.fps}fps ({t[-1]:.2f}s)")
    print(f"  frame travels {travel:+.2f} m in X, drops {drop:+.3f} m")
    print(f"  size {out.stat().st_size / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
