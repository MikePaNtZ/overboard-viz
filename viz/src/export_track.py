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


def _geom_table(model) -> list:
    """Every geom in the model as primitive parameters, in BODY-LOCAL space.

    The bench rig is the first plant made entirely of MuJoCo primitives rather
    than STL meshes, so there is no mesh file for the renderer to load and the
    scene has to be built from numbers. Those numbers are read out of the
    compiled model here instead of being transcribed into the Blender scene by
    hand — which means a change to `sim/models/bench_rig.xml` propagates into
    the render on the next export, and cannot silently disagree with it.

    That matters more than convenience: a hand-copied desk overhang or disc
    radius that drifts from the MJCF would make the film a picture of a rig
    that was never simulated, while still passing every check we have.

    ⚠️ Colour comes from the geom's MATERIAL when it has one, and only from
    `geom_rgba` when it does not. A geom carrying `material="amber"` leaves
    `geom_rgba` at MuJoCo's default 0.5 0.5 0.5 — so reading `geom_rgba` alone
    silently renders the whole rig in flat grey that looks like a deliberate
    neutral study rather than a bug. Resolve `geom_matid` first.

    ⚠️ These values are sRGB, as everywhere in MJCF. Blender's Base Color is
    linear. The renderer converts; do not convert twice.
    """
    _TYPE = {
        mujoco.mjtGeom.mjGEOM_PLANE: "plane",
        mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
        mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
        mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid",
        mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
        mujoco.mjtGeom.mjGEOM_BOX: "box",
    }
    out = []
    for gid in range(model.ngeom):
        gtype = model.geom_type[gid]
        if gtype not in _TYPE:
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 model.geom_bodyid[gid])
        matid = int(model.geom_matid[gid])
        rgba = (model.mat_rgba[matid] if matid >= 0 else model.geom_rgba[gid])
        out.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid),
            "body": body or "world",
            "type": _TYPE[gtype],
            # Local to the parent body, so the renderer parents each geom to an
            # empty driven by that body's track and never re-derives a pose.
            "pos": [float(v) for v in model.geom_pos[gid]],
            "quat": [float(v) for v in model.geom_quat[gid]],
            "size": [float(v) for v in model.geom_size[gid]],
            "rgba_srgb": [float(v) for v in rgba],
            "material": (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, matid)
                         if matid >= 0 else None),
            "mass_kg": float(model.body_mass[model.geom_bodyid[gid]]),
        })
    return out


def _export_bench(args) -> int:
    """The wave-0 bench rig: one outrunner, one flywheel, one hinge.

    This is NOT the board and NOT the board-in-a-fixture that the Bench
    Test-Stand design doc describes — see that doc's §3, which specifies side
    plates picking up the axle bolts. What Mechanical actually built first is a
    desk-scale rig whose job is to prove the *identification method* on hardware
    we own before it is pointed at a hub motor we do not. Anything published
    from this footage has to say that, or it reads as progress on the board.

    WHY THIS IS SHOT IN SLOW MOTION, AND WHY THAT IS STILL LANE A
    ------------------------------------------------------------
    The identify run is 20 ms long and the loaded disc passes 13 rev/s inside
    150 ms. At 30 fps real time it is either half a frame or an aliased blur;
    there is no honest real-time cut of it.

    So one rendered frame is exactly ONE MuJoCo timestep — no interpolation, no
    duplicated frames, no invented in-betweens. Playing 2000 Hz of simulation
    at 30 fps is uniform 66.7x slow motion, the time scale is recorded in the
    manifest, and the renderer burns it into the frame. Every frame remains a
    measurement, which is the Lane A test: a reader can regenerate this exact
    sequence from the committed .otrk and the committed scene.

    (Uniform, declared, integer-exact retiming is the only retiming that keeps
    that property. Non-uniform retiming, trimming that hides, or interpolated
    in-betweens do not, and are Lane B.)
    """
    from sim.scenarios import bench_spinup as bench
    from sim.scenarios.imperfections import STAGE0_PLACEHOLDER, ImperfectionState

    loaded = args.bench_variant != "bare"
    model = bench.load_model() if loaded else bench.build_bare_model()
    ip = bench.IdentifyParams()
    dt = float(model.opt.timestep)

    # Mechanical's own fit, run first. Its reported ramp slopes are the
    # reference this export is checked against below, so a divergence between
    # what they measure and what this films fails loudly instead of shipping.
    ident = bench.identify(ip)
    ref_alpha = (ident.metrics.alpha_loaded_rad_s2 if loaded
                 else ident.metrics.alpha_bare_rad_s2)

    print(f"bench identify ({'flywheel-loaded' if loaded else 'bare rotor'}): "
          f"{ip.commanded_current_a} A held, dt={dt*1e3:.2f} ms")
    print(f"  kt_fit      = {ident.metrics.kt_fit_nm_per_a:.5f} N·m/A")
    print(f"  J_bare_fit  = {ident.metrics.j_bare_fit_kg_m2:.4e} kg·m²")
    print(f"  J_disc      = {ident.metrics.j_disc_known_kg_m2:.4e} kg·m² (known)")
    print(f"  alpha ref   = {ref_alpha:.1f} rad/s²")

    # Re-run the identical held-current drive, logging true shaft angle. The
    # scenario's own result carries the SENSED rate only (quantised through the
    # imperfection profile), and integrating that would accumulate quantisation
    # error into the disc's visible angle — a small, plausible-looking lie.
    # The profile owns a seeded generator and a fresh ImperfectionState per run,
    # so this reproduces their drive bit-for-bit.
    n_steps = int(round(args.bench_seconds / dt))
    data = mujoco.MjData(model)
    imp = ImperfectionState(profile=STAGE0_PLACEHOLDER, dt_s=dt)
    mujoco.mj_forward(model, data)

    qpos = np.zeros(n_steps, np.float64)
    qvel = np.zeros(n_steps, np.float64)
    w_sensed = np.zeros(n_steps, np.float64)
    tau = np.zeros(n_steps, np.float64)
    t = np.zeros(n_steps, np.float64)
    for k in range(n_steps):
        current = imp.apply_current(ip.commanded_current_a)
        data.ctrl[0] = current * bench.NAMEPLATE_KT_NM_PER_A
        mujoco.mj_step(model, data)
        qpos[k] = data.qpos[0]
        qvel[k] = data.qvel[0]
        # The same degraded channel the identification actually fits — see the
        # cross-check below for why the render logs it as well as truth.
        w_sensed[k] = imp.wheel_rate(float(data.qvel[0]), float(data.time))
        tau[k] = data.ctrl[0]
        t[k] = data.time

    # Cross-check against Mechanical's fit, using THEIR fitter on THEIR
    # quantity. This has to be the sensed rate, not MuJoCo truth: over the 5 ms
    # fit window a 1 ms actuation delay and a 500 Hz update staircase bend the
    # measurable ramp well away from the ideal kt*i/J, and fitting truth here
    # reports ~385 rad/s² against their 269 — a 43% "divergence" that is really
    # just two different signals. Comparing like with like is what makes this a
    # real guard rather than a tripwire that has to be loosened until it passes.
    slope, r2 = bench._fit_line(t, w_sensed, ip.fit_window_s)
    err = abs(slope - ref_alpha) / abs(ref_alpha)
    if err > 0.02:
        raise SystemExit(
            f"bench export diverged from sim.scenarios.bench_spinup.identify(): "
            f"re-run alpha={slope:.1f} rad/s² vs reported {ref_alpha:.1f} "
            f"({err:.1%} > 2%). The film would not be of the run whose numbers "
            f"it carries. Refusing to write the track."
        )
    print(f"  alpha re-run= {slope:.1f} rad/s² (R²={r2:.4f})  "
          f"✓ within {err:.2%} of the fit")

    BODIES = ["rotor"]
    bid = {b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in BODIES}
    pos = {b: np.zeros((n_steps, 3), np.float32) for b in BODIES}
    quat = {b: np.zeros((n_steps, 4), np.float32) for b in BODIES}
    for k in range(n_steps):
        data.qpos[0] = qpos[k]
        mujoco.mj_forward(model, data)
        for b in BODIES:
            pos[b][k] = data.xpos[bid[b]]
            quat[b][k] = data.xquat[bid[b]]

    time_scale = 1.0 / (args.fps * dt)  # sim seconds per played second
    rev = qpos / (2.0 * np.pi)
    name = f"bench_identify_{'loaded' if loaded else 'bare'}"

    manifest = {
        "schema_version": "1.0",
        "lane": "A",
        "source": {
            "kind": "sim",
            "scenario": name,
            "plant": "bench_rig wave-0 (desk-scale)",
            "model_file": "sim/models/bench_rig.xml",
            "model_sha256": hashlib.sha256(
                bench.MODEL_PATH.read_bytes()).hexdigest(),
            "commanded_current_a": ip.commanded_current_a,
            "imperfection_profile": STAGE0_PLACEHOLDER.profile_id,
            "mujoco_version": mujoco.__version__,
            "exporter_version": "otrk-export 1.1.0",
        },
        # One frame is one timestep. `fps` is the PLAYBACK rate; `time_scale`
        # is how much slower than life that is, and the renderer must display
        # it. Anything that reads this file and ignores time_scale will present
        # a 20 ms event as if it took ten seconds.
        "time": {
            "fps": args.fps,
            "n_frames": n_steps,
            "duration_s": float(t[-1]),
            "sim_dt_s": dt,
            "frames_per_timestep": 1,
            "time_scale": time_scale,
            "playback_note": f"{time_scale:.1f}x slow motion",
        },
        "bodies": BODIES,
        "channels": ["shaft_rad", "shaft_rad_s", "shaft_rev", "cmd_torque_nm"],
        "events": [],
        "hints": {"subject_body": "rotor", "ground_z": 0.0,
                  "static_bodies": ["world"]},
        "conventions": {
            "units": "SI metres", "world_frame": "right-handed, +Z up",
            "quaternion": "(w, x, y, z), unit, body->world",
        },
        # The scene itself, read out of the compiled model — see _geom_table.
        "geoms": _geom_table(model),
        "bindings": None,
        # Carried so the render can caption itself without anyone re-typing a
        # number that has since been refitted. Lane A may show these; Lane B
        # may not show any of them.
        "fit": ident.metrics.__dict__ | {"alpha_rerun_rad_s2": slope},
    }

    arrays = {
        "manifest": json.dumps(manifest),
        "t": t,
        "ch/shaft_rad": qpos.astype(np.float32),
        "ch/shaft_rad_s": qvel.astype(np.float32),
        "ch/shaft_rev": rev.astype(np.float32),
        "ch/cmd_torque_nm": tau.astype(np.float32),
    }
    for b in BODIES:
        arrays[f"pos/{b}"] = pos[b]
        arrays[f"quat/{b}"] = quat[b]

    out = args.out or ROOT / f"viz/scenes/lane_a/{name}.otrk.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"\nwrote {out.relative_to(ROOT)}: {n_steps} frames "
          f"({t[-1]*1e3:.0f} ms of sim) → {n_steps/args.fps:.1f}s at "
          f"{args.fps}fps = {time_scale:.1f}x slow motion")
    print(f"  peak {qvel.max():.1f} rad/s ({qvel.max()/(2*np.pi):.1f} rev/s), "
          f"{rev[-1]:.2f} revolutions total")
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
    ap.add_argument("--bench", action="store_true",
                    help="wave-0 desk bench rig: held-current flywheel spin-up")
    ap.add_argument("--bench-variant", default="loaded",
                    choices=["loaded", "bare"],
                    help="flywheel fitted, or the bare rotor it is fitted against")
    ap.add_argument("--bench-seconds", type=float, default=0.15,
                    help="sim seconds to capture; every timestep becomes one "
                         "frame, so 0.15 s is 300 frames = 10 s at 30 fps")
    args = ap.parse_args()

    if args.bench:
        return _export_bench(args)

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
