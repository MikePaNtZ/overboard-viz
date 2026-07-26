#!/usr/bin/env python3
"""Render a pose track as a clip. V1.1.

Defaults are the **G2 grade**, chosen 2026-07-26 for the overboard-web #now
card. The plate has to sit inside the page's dark band (--bg #0F1922) or it
reads as a lightbox glowing off a matte page — the single thing that makes a
render look pasted into a design rather than built into it. The fix is value,
not hue: the board's own colours are already the site's brand tokens.

The camera sits almost on the floor for the same reason. At eye level the
concrete filled most of the frame as a large flat pale mass — the brightest
thing in the shot and, on the page, a lightbox. Dropping to 6 cm above the
board's centre compresses the floor into the lower third and puts the dark
garage behind the subject, which fixes the value problem and the composition
in one move. Measured: floor luminance ~24/255 against the page's --bg at ~25.

    blender --background --factory-startup --python viz/src/render_clip.py -- [args]
    (drop --factory-startup for --engine CYCLES; the Cycles add-on is not
     loaded under factory startup)

Reads `impulse.otrk.npz` — the ICD §5 pose track — and drives the same board
and the same garage as the V1.0 still. Two differences, both forced by the
physics rather than chosen:

  · **The floor, not the benchtop.** The impulse scenario pushes the board
    3.6 m across the ground. A bench is the wrong stage for a vehicle that
    travels; it would roll off in the first half second.
  · **Motion blur on.** V1.0 disabled it because nothing was moving.

Every visible part is parented to one of two empties, `frame` and `wheel`,
which are the only things keyframed. That is the ICD's body/binding split made
literal: the track supplies two rigid-body trajectories, the bindings supply
where each mesh sits on its body, and Blender's parenting does the rest.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_scene as bs  # noqa: E402  — the V1.0 scene, reused wholesale

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACK = ROOT / "viz/scenes/impulse.otrk.npz"
CONCRETE = ROOT / "viz/assets/textures/concrete_floor_worn_001"
OUTDOOR_HDRI = ROOT / "viz/assets/hdri/approaching_storm_4k.hdr"
GRASS = ROOT / "viz/assets/textures/aerial_grass_rock"


def _outdoor(size: float = 400.0, tint: float = 0.55):
    """Open ground for the anonymous-outdoor scene.

    Texture only. The plant is a flat plane, so the render may not add camber,
    gravel, kerbs or slope it does not model — set dressing that invents
    terrain is a claim about where the board can go. Same reason the HDRI was
    chosen for having no skyline, landmarks, benches or people: depicting an
    identifiable place asserts the board was taken there.
    """
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "ground"
    ground.data.materials.append(
        bs._textured("outdoor_ground", GRASS, uv_scale=size / 2.5, tint=tint))
    return ground


def _floor(size: float = 200.0, tint: float = 0.42):
    """Garage floor.

    Deliberately enormous. At 24 m the plane's far edge landed inside the shot
    as a hard horizon line with the HDRI above it — the giveaway that the
    "room" is a postage stamp floating in an environment map. Pushing the edge
    out to 100 m puts it beyond anything the lens resolves, and perspective
    plus depth of field finish the job.

    uv_scale is tied to `size` so the concrete keeps a ~2 m repeat whatever the
    plane does; scaling one without the other is how the texture ends up
    stretched into invisible smears.
    """
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "floor"
    floor.data.materials.append(
        bs._textured("concrete", CONCRETE, uv_scale=size / 2.0, tint=tint))
    return floor


def _capsule(b: dict):
    """A MuJoCo capsule: Z-aligned, `radius` + `half_length` of the straight part.

    Built as one cylinder with its rim circles bevelled by exactly the radius,
    which produces the hemispherical caps — the same trick as the tyre crown.
    Cheaper and less fragile than joining three meshes, and it stays a single
    object so material assignment and parenting work like every other geom.
    """
    r, hl = b["radius"], b["half_length"]
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=r, depth=2 * (hl + r))
    obj = bpy.context.object
    bev = obj.modifiers.new("cap", "BEVEL")
    bev.width, bev.segments = r * 0.999, 8
    bev.limit_method = "ANGLE"
    bev.angle_limit = math.radians(30)
    return obj


def _sphere(b: dict):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=b["radius"])
    return bpy.context.object


def _rig(track, bindings):
    """Build the two driven empties and hang every mesh off them."""
    empties = {}
    for body in track["bodies"]:
        e = bpy.data.objects.new(f"body_{body}", None)
        bpy.context.collection.objects.link(e)
        e.rotation_mode = "QUATERNION"
        empties[body] = e

    for b in bindings:
        if b["kind"] == "mesh":
            before = set(bpy.data.objects)
            bpy.ops.wm.stl_import(filepath=b["file"], global_scale=b["scale"])
            obj = list(set(bpy.data.objects) - before)[0]
        elif b["kind"] == "cylinder":
            obj = bs._tyre({"radius": b["radius"], "half_width": b["half_width"],
                            "crown": 0.18})
        elif b["kind"] == "capsule":
            obj = _capsule(b)
        elif b["kind"] == "sphere":
            obj = _sphere(b)
        else:
            continue
        obj.name = b["name"]

        # Parent first, then set the local transform. Assigning .parent in
        # Python leaves matrix_parent_inverse as identity, so location and
        # rotation are interpreted directly in the parent's frame — which is
        # exactly what the binding is expressed in.
        obj.parent = empties[b["body"]]
        obj.location = Vector(b["pos"])
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = Quaternion(b["quat"])

        mat = b.get("material") or {}
        obj.data.materials.append(bs._principled(
            f"{mat.get('name', 'shell_mat')}_{obj.name}",
            mat.get("rgba", [0.5, 0.5, 0.5, 1]),
            bs.FINISH.get(mat.get("name"), bs.FINISH["shell_mat"])))

        for p in obj.data.polygons:
            p.use_smooth = True
        obj.modifiers.new("smooth", "EDGE_SPLIT").split_angle = math.radians(40)

    return empties


def _linear_keys() -> None:
    """Insert every keyframe as LINEAR rather than the default Bezier.

    This matters more than it looks: each frame of the track is a measurement,
    so Bezier smoothing between keys invents motion MuJoCo never computed —
    overshoot on the nose-over, in particular — which quietly breaks the one
    promise this pipeline makes.

    Set as a preference rather than by walking F-curves afterwards: Blender 5
    moved Actions to slots/layers and `action.fcurves` no longer exists, so
    the old traversal is both broken and version-fragile.
    """
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"


def _animate(empties, track, n):
    for body, e in empties.items():
        pos, quat = track[f"pos/{body}"], track[f"quat/{body}"]
        for i in range(n):
            e.location = Vector(pos[i].tolist())
            e.rotation_quaternion = Quaternion(quat[i].tolist())
            e.keyframe_insert("location", frame=i + 1)
            e.keyframe_insert("rotation_quaternion", frame=i + 1)



def _tracking_camera(track, n, lens: float, lag: float, side: float, height: float,
                     release_frame: int = 0, back: float = 0.0,
                     static: bool = False, aim_up: float = 0.0):
    """A camera that travels with the board instead of watching it leave.

    The board covers 3.6 m. A locked-off camera would lose it in a second, and
    panning from a fixed point flattens the whole move into a rotation. So the
    camera dollies along X with the subject, at a slight lag so there is still
    relative motion in frame — the thing that makes a tracking shot read as a
    shot rather than a rigid attachment.
    """
    pos = track["pos/frame"]
    tgt = bpy.data.objects.new("focus_target", None)
    bpy.context.collection.objects.link(tgt)

    cam_data = bpy.data.cameras.new("clip")
    cam_data.lens = lens
    cam_data.dof.use_dof = True
    cam_data.dof.focus_object = tgt
    cam_data.dof.aperture_fstop = 2.8   # a touch deeper than the still: the
                                        # subject moves through the focal plane
    cam = bpy.data.objects.new("clip", cam_data)
    bpy.context.collection.objects.link(cam)

    c = cam.constraints.new("TRACK_TO")
    c.target = tgt
    c.track_axis, c.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    bpy.context.scene.camera = cam

    if static:
        # A locked-off wide shot, for a route that comes back. Tracking a
        # subject that returns to where it started cancels the reversal — the
        # whole point of the shuttle run is that it goes out, stops, comes
        # back and stops, and only a fixed frame lets a viewer see that.
        # Centre on the midpoint of the travelled RANGE, not the mean position:
        # the route holds station for four seconds at each end, so a mean is
        # dragged toward wherever it paused longest and the shot ends up
        # off-centre. Aim height is the subject's, not the axle's — with a
        # 1.4 m figure aboard, framing on the axle points the camera at its feet.
        lo, hi = pos.min(axis=0), pos.max(axis=0)
        cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
        tgt.location = (cx, cy, float(pos[0][2]) + aim_up)
        cam.location = (cx + back, cy + side, float(pos[0][2]) + height)
        return cam

    x0 = float(pos[0][0])
    # After release_frame the rig stops following and simply holds. For the
    # closed-loop run that beat IS the story: a camera locked to the subject
    # cancels the very thing the clip has to show, because the board balancing
    # while riding away looks identical to a board standing still. Letting it
    # leave frame is what makes "it holds attitude, then rides away" legible —
    # and the drift is the honest half of that sentence.
    # `back` parks the camera behind the start; with lag=0 it never moves, so
    # the board recedes into the garage instead of sliding sideways out of
    # frame. Freezing a side-tracking camera does not work here — the board is
    # still accelerating when the clip ends, so it clears a 1.2 m frame in well
    # under a second no matter how late the release. Shoot a departure from
    # behind and it simply gets smaller, which reads as "rides away" and never
    # runs out of frame to leave.
    hold = None
    for i in range(n):
        bx, by, bz = (float(v) for v in pos[i])
        if release_frame and i >= release_frame:
            if hold is None:
                hold = (tgt.location.copy(), cam.location.copy())
            tgt.location, cam.location = hold[0], hold[1]
        else:
            # aim_up applies here too: with a 1.4 m figure aboard, aiming at the
            # axle points the camera at its feet and crops the head.
            tgt.location = (bx, by, bz + aim_up)
            cam.location = (x0 + back + (bx - x0) * lag, by + side, bz + height)
        tgt.keyframe_insert("location", frame=i + 1)
        cam.keyframe_insert("location", frame=i + 1)

    return cam


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="EEVEE", choices=["EEVEE", "CYCLES"])
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--lens", type=float, default=50.0)
    ap.add_argument("--lag", type=float, default=0.92)
    ap.add_argument("--side", type=float, default=-1.62)
    ap.add_argument("--height-offset", type=float, default=0.06)
    ap.add_argument("--hdri-rot", type=float, default=115.0)
    ap.add_argument("--exposure", type=float, default=-0.5)
    ap.add_argument("--hdri-strength", type=float, default=0.70)
    ap.add_argument("--kicker", type=float, default=30.0)
    ap.add_argument("--floor-tint", type=float, default=0.16)
    ap.add_argument("--out", type=Path, default=ROOT / "out/impulse_clip.mp4")
    ap.add_argument("--track", type=Path, default=DEFAULT_TRACK)
    ap.add_argument("--aim-up", type=float, default=0.0,
                    help="raise the aim point above the axle (a ridden board is tall)")
    ap.add_argument("--static", action="store_true",
                    help="locked-off camera aimed at the route centre")
    ap.add_argument("--scene", default="garage", choices=["garage", "outdoor"])
    ap.add_argument("--cam-back", type=float, default=0.0,
                    help="park the camera this far behind the start (use with --lag 0)")
    ap.add_argument("--release-at", type=float, default=0.0,
                    help="seconds after which the camera stops following and holds")
    ap.add_argument("--frame", type=int, default=0,
                    help="render this single frame only, for iteration")
    args = ap.parse_args(argv)

    npz = np.load(args.track, allow_pickle=False)
    manifest = json.loads(str(npz["manifest"]))
    track = {"bodies": manifest["bodies"]}
    for b in manifest["bodies"]:
        track[f"pos/{b}"] = npz[f"pos/{b}"]
        track[f"quat/{b}"] = npz[f"quat/{b}"]

    n = manifest["time"]["n_frames"]
    fps = manifest["time"]["fps"]
    print(f"track: {n} frames @ {fps}fps, {manifest['time']['duration_s']:.2f}s, "
          f"kind={manifest['source']['kind']}, scenario={manifest['source']['scenario']}")
    for e in manifest["events"]:
        print(f"  event t={e['t']:.3f}s  {e['kind']}  ({e['note']})")

    bs._clear_scene()
    _linear_keys()
    if args.scene == "outdoor":
        bs._world(OUTDOOR_HDRI, args.hdri_strength, args.hdri_rot)
        _outdoor()
    else:
        bs._world(bs.HDRI, args.hdri_strength, args.hdri_rot)
        _floor(tint=args.floor_tint)
    empties = _rig(track, manifest["bindings"])
    _animate(empties, track, n)
    _tracking_camera(track, n, args.lens, args.lag, args.side, args.height_offset,
                     release_frame=int(args.release_at * fps) if args.release_at else 0,
                     back=args.cam_back, static=args.static,
                     aim_up=args.aim_up)
    bs._kicker(args.kicker)

    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = (args.frame, args.frame) if args.frame else (1, n)
    scene.render.fps = fps
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    # PNG sequence, then encode separately. This Blender build ships without
    # FFmpeg output (the file_format enum has no FFMPEG entry), and rendering
    # frames is the better shape anyway: an interrupted render resumes, and
    # the encode can be re-run at a different bitrate without re-rendering —
    # which matters because Notion caps uploads at 5 MiB on the free plan.
    frames_dir = args.out.with_suffix("") / "frame_"
    frames_dir.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(frames_dir)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Punchy"
    scene.view_settings.exposure = args.exposure

    # The board is moving now, so this is no longer optional: without it the
    # wheel strobes and every frame looks like a crisp still of a still object.
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.5

    if args.engine == "CYCLES":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        scene.cycles.device = "GPU"
        prefs = bpy.context.preferences.addons.get("cycles")
        if prefs:
            prefs.preferences.compute_device_type = "METAL"
            prefs.preferences.get_devices()
            for d in prefs.preferences.devices:
                d.use = (d.type == "METAL")
    else:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = args.samples
        for attr in ("use_raytracing", "use_shadows"):
            if hasattr(scene.eevee, attr):
                setattr(scene.eevee, attr, True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.frame:
        scene.frame_set(args.frame)
        scene.render.filepath = str(args.out.with_suffix("")) + f"_f{args.frame:04d}"
        bpy.ops.render.render(write_still=True)
        print(f"\nwrote single frame {scene.render.filepath}.png")
        return 0
    bpy.ops.render.render(animation=True)
    print(f"\nwrote {n} frames to {frames_dir.parent}")
    print(f"encode with:\n  ffmpeg -y -framerate {fps} -i '{frames_dir}%04d.png' "
          f"-c:v libx264 -pix_fmt yuv420p -crf 20 '{args.out}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
