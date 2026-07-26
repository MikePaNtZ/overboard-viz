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
import hashlib
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
WATERFRONT_HDRI = ROOT / "viz/assets/hdri/the_sky_is_on_fire_4k.hdr"
GRASS = ROOT / "viz/assets/textures/aerial_grass_rock"
ASPHALT = ROOT / "viz/assets/textures/asphalt_02"


def _bike_trail(length: float = 400.0, trail_w: float = 3.0,
                far_edge: float = 7.5, tint: float = 0.75):
    """A narrow asphalt trail with grass either side, and open water beyond.

    Three things this fixes over the first waterfront pass:

    * **Width.** A 13 m paved strip is an apron, not a path. A bike trail is
      about 3 m, and at that width the surface reads as somewhere a rider would
      actually ride rather than an undifferentiated grey plain.
    * **Surface.** Worn concrete had no directional cue at all. Asphalt with
      grass shoulders gives the eye an edge to follow, which is most of what
      makes a path look like a path.
    * **Water.** The ground now *ends* at `far_edge`. Previously it ran to the
      horizon, so the sea was never visible — the backdrop was the plane's own
      edge. Ending it closer, and raising the camera, widens the band of water
      between that edge and the horizon.

    Everything is flush at z = 0. No kerb, no camber: the plant is a flat rigid
    plane and the board would visibly ignore any relief added here.
    """
    def strip(name, y_centre, width, tex, uv_m, z=0.0, tnt=tint):
        bpy.ops.mesh.primitive_plane_add(size=1, location=(0, y_centre, z))
        o = bpy.context.object
        o.name = name
        o.scale = (length, width, 1)
        bpy.ops.object.transform_apply(scale=True)
        o.data.materials.append(bs._textured(
            name, tex, uv_scale=(length / uv_m, width / uv_m, 1.0), tint=tnt))
        return o

    trail = strip("trail", 0.0, trail_w, ASPHALT, 2.5)
    # Shoulders sit a hair below the asphalt so the trail edge stays crisp
    # instead of z-fighting along its whole length.
    far = (far_edge + trail_w / 2) / 2 + trail_w / 4
    strip("verge_far", far, far_edge - trail_w / 2, GRASS, 2.0, z=-0.004, tnt=tint * 0.75)
    strip("verge_near", -14.0, 26.0, GRASS, 2.0, z=-0.004, tnt=tint * 0.75)
    return trail


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


# Finishes for the bench rig. Keyed by geom name because this plant is made of
# primitives rather than named mesh materials, so there is nothing else to key
# on. Appearance only — every dimension comes from the exported geom table, and
# nothing here may change a size, a position or an axis.
BENCH_FINISH = {
    # 6061 aluminium, machined and anodised amber. The one part whose inertia
    # is *known* (weighed and measured), so it is also the one the eye should
    # land on first.
    #
    # Metallic is held well below a physical anodised-aluminium value on
    # purpose. A fully metallic surface has no diffuse colour of its own — it
    # shows only what is around it — and the garage HDRI is a dim room, so at
    # metallic 0.9 the amber disappeared and the disc rendered near-black. This
    # is the "sRGB vs linear" trap's cousin: the material was *correct* and the
    # image was wrong. Keep enough diffuse for the brand amber to survive.
    "flywheel":  dict(roughness=0.30, metallic=0.45, coat=0.10),
    "rotor_can": dict(roughness=0.38, metallic=0.45, coat=0.00),
    "stator":    dict(roughness=0.42, metallic=0.40, coat=0.00),
    "hub":       dict(roughness=0.34, metallic=0.40, coat=0.00),
    "plate":     dict(roughness=0.52, metallic=0.10, coat=0.05),
    # Painted-on index mark. Matte, because in hardware it is tape or paint —
    # if it reads as metal it looks like a machined feature with mass, and the
    # MJCF is explicit that it has none.
    "index":     dict(roughness=0.88, metallic=0.00, coat=0.00),
    # The bench. A visual liberty is taken on its FINISH — the MJCF gives it a
    # debug checker, which is a viewer aid, not a claim about a real desk — but
    # not on its size or position, which come from the model like everything
    # else. This is the same latitude the garage floor already takes.
    "desk":      dict(roughness=0.62, metallic=0.05, coat=0.00),
}
_BENCH_DEFAULT_FINISH = dict(roughness=0.45, metallic=0.30, coat=0.00)

# Appearance-only colour overrides, for geoms whose MJCF material is a viewer
# aid rather than a claim about the object. `grid_mat` is a procedural checker
# used to read motion in MuJoCo's own viewer; it carries no rgba, so the geom
# arrives white and the bench renders as a featureless white slab that pulls
# the eye straight off the rig. A plain dark benchtop is the honest reading of
# "a desk". Sizes and positions are never overridden — only colour.
BENCH_RGBA_OVERRIDE = {
    "desk": [0.208, 0.184, 0.161, 1.0],
}

# Geoms the MJCF models as solids but which are surface markings in hardware.
#
# `index` is the stripe that makes shaft rotation legible — the reason it
# exists at all is so a render or a phone video can be checked against reported
# ERPM. The model gives it the SAME y-extent as the flywheel (both span
# 0.040–0.052), i.e. buried inside the disc and exactly coplanar with both
# faces. MuJoCo does not care: it is massless with contype=0, so this costs the
# physics nothing and the header is explicit that in hardware it is "paint or a
# strip of tape". A renderer very much does care — coplanar faces z-fight, and
# across 300 frames that flickers.
#
# So it is rendered as what it physically is: a decal of tape thickness sitting
# ON the named geom's outward face. This is the one place the render departs
# from the model's geometry, it is declared in the render manifest, and it
# moves a massless marker by under a millimetre. Nothing load-bearing moves.
BENCH_DECAL = {
    "index": dict(on_face_of="flywheel", axis=1, thickness_m=0.0008),
}


def _bench_rig(manifest):
    """Build the bench rig from the geom table the exporter read off the MJCF.

    Nothing here is transcribed by hand. Sizes, offsets and orientations all
    come from the compiled model, so if Mechanical changes `bench_rig.xml` the
    next export moves the render with it — and a render that disagrees with the
    plant cannot be produced by editing this file alone.

    MuJoCo → Blender needs no axis conversion: both are right-handed, +Z up,
    with scalar-first (w,x,y,z) quaternions. Do not add a swap.
    """
    empties = {}
    for body in set([g["body"] for g in manifest["geoms"]]) | set(manifest["bodies"]):
        e = bpy.data.objects.new(f"body_{body}", None)
        bpy.context.collection.objects.link(e)
        e.rotation_mode = "QUATERNION"
        empties[body] = e

    by_name = {g["name"]: g for g in manifest["geoms"]}
    decals = []

    for g in manifest["geoms"]:
        size, name = list(g["size"]), g["name"]
        pos = list(g["pos"])

        # Lift surface markings onto the face they are painted on — see BENCH_DECAL.
        if name in BENCH_DECAL:
            d = BENCH_DECAL[name]
            host = by_name.get(d["on_face_of"])
            if host is None:
                # The surface this is painted on is not in the model, so the
                # marking cannot be either. This is not hypothetical: the
                # identification's bare-rotor variant strips the `flywheel`
                # geom but not the `index` mark that sits on it, leaving a
                # white bar rotating in mid-air 52 mm off the shaft. It costs
                # the physics nothing (the mark is massless, contype=0) so the
                # fit is unaffected — but rendered, it reads as a broken scene.
                # Dropping it is the only honest option available here; adding
                # a marker somewhere else would be inventing geometry.
                print(f"  dropped decal '{name}': host geom "
                      f"'{d['on_face_of']}' is absent from this model")
                continue
            if host:
                ax = d["axis"]
                # The host cylinder's half-length is size[1]; its outward face
                # sits that far from its own centre along the rotation axis.
                face = host["pos"][ax] + host["size"][1]
                size[ax if ax < len(size) else 1] = d["thickness_m"]
                pos[ax] = face + d["thickness_m"]
                decals.append(name)
        if g["type"] == "box":
            bpy.ops.mesh.primitive_cube_add(size=2.0)
            obj = bpy.context.object
            obj.scale = Vector(size[:3])          # MuJoCo box size = half-extents
        elif g["type"] == "cylinder":
            # MuJoCo: size = (radius, half-length), axis along the geom's local
            # +Z — the same convention Blender's primitive uses, which is why
            # the euler="90 0 0" in the MJCF arrives already baked into quat.
            bpy.ops.mesh.primitive_cylinder_add(radius=size[0], depth=2.0 * size[1],
                                                vertices=96)
            obj = bpy.context.object
        elif g["type"] == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=size[0], segments=48, ring_count=24)
            obj = bpy.context.object
        else:
            continue
        obj.name = name

        # Parent first, then set the local transform — assigning .parent in
        # Python leaves matrix_parent_inverse identity, so these are read
        # directly in the parent body's frame, which is what the table holds.
        obj.parent = empties[g["body"]]
        obj.location = Vector(pos)
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = Quaternion(g["quat"])

        obj.data.materials.append(bs._principled(
            f"bench_{name}", BENCH_RGBA_OVERRIDE.get(name, g["rgba_srgb"]),
            BENCH_FINISH.get(name, _BENCH_DEFAULT_FINISH)))

        for p in obj.data.polygons:
            p.use_smooth = True
        # EDGE_SPLIT alongside use_smooth, always. Smooth-shading a flat end cap
        # without it turned the hub disc into a chrome eyeball once already, and
        # this scene is nothing but end caps.
        obj.modifiers.new("smooth", "EDGE_SPLIT").split_angle = math.radians(35)

    return empties


def _bench_camera(lens: float, dist: float, azim: float, elev: float,
                  target=(0.055, 0.02, 0.118)):
    """Locked off, and it has to be.

    The rig does not translate — one hinge, and the only motion in the frame is
    the disc turning. A moving camera here would be pure authorship: it would
    add apparent motion that the plant does not contain, on a clip whose entire
    claim is that every frame is a measurement.
    """
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.lens = lens
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    a, e = math.radians(azim), math.radians(elev)
    t = Vector(target)

    # `_kicker` aims itself at this by name. The tracking camera creates one as
    # a side effect of following the board; a locked-off camera has to make it
    # explicitly, or the rim light points at the world origin instead of the rig.
    focus = bpy.data.objects.new("focus_target", None)
    bpy.context.collection.objects.link(focus)
    focus.location = t

    cam.location = t + Vector((dist * math.cos(e) * math.cos(a),
                               dist * math.cos(e) * math.sin(a),
                               dist * math.sin(e)))
    d = (t - cam.location).normalized()
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()

    # A 75 mm disc half a metre away: depth of field is a real effect at this
    # scale, not a stylistic add. Focused on the disc face so the desk behind
    # falls away and the eye is not asked to read the whole bench at once.
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = (t - cam.location).length
    cam.data.dof.aperture_fstop = 3.2
    return cam


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


def _write_render_manifest(args, track_manifest, n, fps) -> Path:
    """Emit the per-render manifest: lane, and the hash of the track it replays.

    Required of every render by the two-lane rule. The hash is the part that
    does the work — it is what lets someone who is not us check a Lane A claim
    instead of taking it on trust. Without it, "reproducible from the committed
    track" is an assertion; with it, anyone can hash `viz/scenes/lane_a/*.otrk.npz`
    and see whether this is the run they were shown.

    The lane is read from the track, not chosen here, so a render cannot
    quietly upgrade itself to Lane A by passing a flag.
    """
    lane = track_manifest.get("lane")
    if lane not in ("A", "B"):
        raise SystemExit(
            f"track {args.track.name} declares lane={lane!r}; expected 'A' or 'B'. "
            f"Every track must declare its lane — refusing to render an "
            f"unlabelled artefact.")

    ts = track_manifest["time"]
    doc = {
        "lane": lane,
        "track": {
            "file": str(args.track.relative_to(ROOT)) if args.track.is_relative_to(ROOT)
                    else str(args.track),
            "sha256": hashlib.sha256(args.track.read_bytes()).hexdigest(),
            "scenario": track_manifest["source"]["scenario"],
            "model_file": track_manifest["source"].get("model_file"),
            "model_sha256": track_manifest["source"].get("model_sha256"),
        },
        "render": {
            "scene": args.scene, "engine": args.engine, "samples": args.samples,
            "resolution": [args.width, args.height], "lens": args.lens,
            "exposure_ev": args.exposure, "hdri_rot_deg": args.hdri_rot,
            "hdri_strength": args.hdri_strength, "kicker": args.kicker,
            "n_frames": n, "playback_fps": fps,
        },
        # Carried up from the track so the fact that this is slow motion cannot
        # be lost by someone reading only the render manifest.
        "time_scale": ts.get("time_scale", 1.0),
        "playback_note": ts.get("playback_note", "real time"),
        # Lane A carries no signature by construction: it is a replay, and the
        # mark exists to disclose authorship, of which there is none here.
        "signature": None if lane == "A" else "required",
        "attribution": "Board meshes: Openwheel (MIT). HDRI/textures: Poly Haven (CC0). "
                       "See viz/assets/MANIFEST.json.",
    }
    out = args.out.with_suffix(".render.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"lane {lane} · track sha256 {doc['track']['sha256'][:12]}… · "
          f"{doc['playback_note']}")
    return out


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
    ap.add_argument("--scene", default="garage",
                    choices=["garage", "outdoor", "waterfront", "bench"])
    ap.add_argument("--cam-dist", type=float, default=0.62,
                    help="bench scene: camera distance from the disc, metres")
    ap.add_argument("--cam-azim", type=float, default=58.0,
                    help="bench scene: camera azimuth, degrees about +Z")
    ap.add_argument("--cam-elev", type=float, default=17.0,
                    help="bench scene: camera elevation, degrees")
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
    if args.scene == "bench":
        ts = manifest["time"]
        print(f"  bench rig: 1 frame = {ts['frames_per_timestep']} timestep "
              f"({ts['sim_dt_s']*1e3:.2f} ms) → {ts['playback_note']}")
        bs._world(bs.HDRI, args.hdri_strength, args.hdri_rot)
        empties = _bench_rig(manifest)
        # Only the tracked bodies get keys. `world` carries the desk, plate and
        # stator — it is static by construction (they are ground in the MJCF),
        # and its empty stays at identity so those geoms sit where the model
        # puts them. Keyframing it would be inventing motion for the bench.
        _animate({b: empties[b] for b in track["bodies"]}, track, n)
        _bench_camera(args.lens, args.cam_dist, args.cam_azim, args.cam_elev)
        # Small, close and warm. The garage kicker is sized and placed for a
        # board on a floor; at bench scale it is both too far away and far too
        # strong, and it washes the disc face flat.
        bs._kicker(args.kicker, warm=True, loc=(0.45, 0.55, 0.55), size=0.5)
    elif args.scene == "waterfront":
        bs._world(WATERFRONT_HDRI, args.hdri_strength, args.hdri_rot)
        _bike_trail(tint=args.floor_tint)
    elif args.scene == "outdoor":
        bs._world(OUTDOOR_HDRI, args.hdri_strength, args.hdri_rot)
        _outdoor()
    else:
        bs._world(bs.HDRI, args.hdri_strength, args.hdri_rot)
        _floor(tint=args.floor_tint)

    if args.scene != "bench":
        empties = _rig(track, manifest["bindings"])
        _animate(empties, track, n)
        _tracking_camera(track, n, args.lens, args.lag, args.side, args.height_offset,
                         release_frame=int(args.release_at * fps) if args.release_at else 0,
                         back=args.cam_back, static=args.static,
                         aim_up=args.aim_up)
    # The waterfront key is a low sun roughly behind the subject, so the rim
    # comes from that side and is warm; the garage key is overhead and cool.
    if args.scene == "bench":
        pass
    elif args.scene == "waterfront":
        # Far back and physically large. Close in, an area light lays a bright
        # elliptical pool on the paving that reads as a film-set spotlight on
        # a beach at dusk. Distance flattens the falloff across the ground so
        # the rim survives without the giveaway.
        bs._kicker(args.kicker, warm=True, loc=(-9.0, 26.0, 5.0), size=12.0)
    else:
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
    _write_render_manifest(args, manifest, n, fps)
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
