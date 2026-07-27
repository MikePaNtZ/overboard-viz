#!/usr/bin/env python3
"""LANE B — an authored concept of how the bench fixture could be mounted.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This is **not** a replay of anything. No `.otrk`, no simulation, no measurement.
It is a drawing of a fixture that has not been designed yet, made to argue for a
shape. It therefore lives in Concept and obeys Concept's rules without exception:

  * it carries a **persistent signature**, default-on, burned into the frame;
  * it carries **no engineering numbers** of any kind;
  * everything written about it is future/subjunctive — *what it would look
    like*, never what it is or was.

WHY IT EXISTS
-------------
`bench_rig.xml` mounts the motor on a 200 mm panel that stands on its bottom
edge, touching the desk along a 50 x 6 mm strip while carrying an overhung
motor. It would tip over. The model's own comment calls that panel "clamped flat
to the desk", which its geometry is not. Filed to Mechanical; their file, their
fix.

Meanwhile the fixture design itself is still open (Bench Test-Stand doc, §10
increment 2), so a concept is worth something rather than nothing.

THE SHAPE, AND THE TWO CONSTRAINTS IT RESPECTS
-----------------------------------------------
An **L-section running along X**: a horizontal foot clamped flat to the desktop,
a vertical riser at the desk edge, motor cantilevered past it.

  1. **The overhang survives.** The header is explicit that it is load-bearing,
     not styling — a flush plate would put the disc over the desk surface, kill
     the pendulum upgrade, and let a shed set screw take a chunk out of the desk.
     Here the disc still swings in free air past the edge.
  2. **The hinge axis is untouched.** The riser is still a panel in the X-Z
     plane, thin in Y, so the motor's mounting face and its Y axis are exactly
     where they were. The disc keeps turning in the plane the board pitches in,
     which is what makes the later arm-and-mass swap a genuine inverted pendulum
     rather than a turntable.

The one thing deliberately NOT done: rotating the shaft to vertical. That was
floated and it forfeits (2).

    blender --background --factory-startup --python viz/src/bench_concept.py -- [args]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_parts as bp  # noqa: E402
import build_scene as bs  # noqa: E402
import render_clip as rc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# Dimensions are the concept's own. They are chosen to sit around the rig the
# model describes, but nothing here is read from the model and nothing here may
# be read back into it — that is the whole point of the lane.
MOUNT_Y = -0.045
DESK_POS, DESK_SIZE = (-0.30, 0.0, -0.02), (0.30, 0.25, 0.02)
DESK_TOP = DESK_POS[2] + DESK_SIZE[2]
CAN_R, CAN_HL, CAN_Y = 0.0315, 0.018, 0.0
DISC_R, DISC_HT, DISC_Y = 0.075, 0.006, 0.046
AXIS_X, AXIS_Z = 0.060, 0.120


def build(args):
    bs._clear_scene()
    bs._world(bs.HDRI, args.hdri_strength, args.hdri_rot)
    rc._floor(size=60.0, tint=args.floor_tint).location.z = -bp.DESK_HEIGHT_M

    world = bpy.data.objects.new("concept_world", None)
    bpy.context.collection.objects.link(world)

    # Desk, as the model has it, plus the legs it does not.
    desk = bpy.data.objects.new("desk_root", None)
    bpy.context.collection.objects.link(desk)
    bp._box(DESK_SIZE, DESK_POS,
            bs._principled("cn_desk", [0.268, 0.230, 0.190, 1.0],
                           dict(roughness=0.62, metallic=0.05, coat=0.00)),
            world, name="desk")
    bp.desk_legs(world, DESK_POS, DESK_SIZE, DESK_TOP - bp.DESK_HEIGHT_M)

    # The bracket, and the clamps that are the point of it.
    # The foot runs BACK across the desk to its free front edge, because that
    # is the only place a G-clamp can actually bite. A foot that stopped short
    # would need clamps in the middle of the desktop, with nothing under them.
    foot_t = 0.005
    desk_front_y = DESK_POS[1] + DESK_SIZE[1]
    bp.angle_bracket(world, MOUNT_Y,
                     riser_x=(-0.050, 0.110), riser_z=(0.0, 0.200),
                     foot_x=(-0.155, 0.004),
                     foot_y=(MOUNT_Y - 0.005, desk_front_y + 0.012),
                     thick=foot_t)
    for nm, cx in (("clamp_a", -0.128), ("clamp_b", -0.022)):
        holder = bpy.data.objects.new(f"{nm}_holder", None)
        bpy.context.collection.objects.link(holder)
        holder.parent = world
        # Sits at the desk's front edge, throat reaching back over foot + desk.
        holder.location = Vector((cx, desk_front_y - 0.004, DESK_TOP + foot_t))
        holder.rotation_mode = "XYZ"
        holder.rotation_euler = (0.0, 0.0, math.radians(-90.0))
        bp.gclamp(holder, 2.0 * DESK_SIZE[2] + foot_t, 0.0, nm)

    # The motor, on a rotor empty that is simply parked at an angle — there is
    # no track here and nothing is animated. A concept does not move.
    rotor = bpy.data.objects.new("rotor_static", None)
    bpy.context.collection.objects.link(rotor)
    rotor.parent = world
    rotor.location = Vector((AXIS_X, 0.0, AXIS_Z))
    rotor.rotation_mode = "XYZ"
    rotor.rotation_euler = (0.0, math.radians(args.disc_angle), 0.0)
    bp.outrunner_can(rotor, CAN_R, CAN_HL, CAN_Y)
    bp.flywheel(rotor, DISC_R, DISC_HT, DISC_Y)

    stator = bpy.data.objects.new("stator_static", None)
    bpy.context.collection.objects.link(stator)
    stator.parent = world
    stator.location = Vector((AXIS_X, 0.0, AXIS_Z))
    bp.stator_half(stator, CAN_R, 0.0125, -0.0325)

    bp.controller(world, DESK_POS[0] + 0.10, DESK_POS[1] + 0.115, DESK_TOP)

    # Aimed between the motor and the clamps, not at the motor: the clamps ARE
    # the argument this concept is making, and a frame that crops them off is
    # showing the old picture with extra metal in it.
    rc._bench_camera(args.lens, args.cam_dist, args.cam_azim, args.cam_elev,
                     target=(args.target[0], args.target[1], args.target[2]),
                     fstop=args.fstop)
    bs._kicker(args.kicker, warm=True, loc=(0.45, 0.55, 0.55), size=0.5)


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="EEVEE", choices=["EEVEE", "CYCLES"])
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--lens", type=float, default=45.0)
    ap.add_argument("--cam-dist", type=float, default=0.72)
    ap.add_argument("--cam-azim", type=float, default=28.0)
    ap.add_argument("--cam-elev", type=float, default=14.0)
    ap.add_argument("--fstop", type=float, default=3.5)
    ap.add_argument("--disc-angle", type=float, default=28.0)
    ap.add_argument("--hdri-rot", type=float, default=115.0)
    ap.add_argument("--hdri-strength", type=float, default=0.35)
    ap.add_argument("--exposure", type=float, default=-0.9)
    ap.add_argument("--kicker", type=float, default=70.0)
    ap.add_argument("--floor-tint", type=float, default=0.30)
    ap.add_argument("--target", type=float, nargs=3, default=[0.01, 0.09, 0.075])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "out/V1.4_bench_fixture_concept.png")
    args = ap.parse_args(argv)

    build(args)

    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Punchy"
    scene.view_settings.exposure = args.exposure
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    if args.engine == "CYCLES":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
    else:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = args.samples

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.out.with_suffix(""))
    bpy.ops.render.render(write_still=True)

    # Concept declares itself in the scene directory AND in the manifest. There
    # is no track to hash, and that absence is exactly what makes this Concept.
    decl = ROOT / "viz/scenes/concept/bench_fixture_concept.json"
    decl.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "lane": "B",
        "kind": "authored concept",
        "track": None,
        "reproduces_a_run": False,
        "may_carry_engineering_numbers": False,
        "copy_tense": "future/subjunctive — what it would look like",
        "signature": "required, burned into the frame",
        "subject": "alternative bench-rig fixture: L-section clamped to the desk",
        "constraints_respected": [
            "overhang preserved — the disc still swings clear past the desk edge",
            "hinge axis unchanged (Y) — the disc still turns in the board's "
            "pitch plane, so the arm-and-mass upgrade survives",
        ],
        "explicitly_not_done": [
            "rotating the shaft to vertical — that makes it a turntable and "
            "forfeits the inverted-pendulum upgrade",
        ],
        "render": {"engine": args.engine, "samples": args.samples,
                   "resolution": [args.width, args.height], "lens": args.lens,
                   "exposure_ev": args.exposure},
        "attribution": "HDRI/textures: Poly Haven (CC0). See viz/assets/MANIFEST.json.",
    }
    decl.write_text(json.dumps(doc, indent=2) + "\n")
    out_manifest = args.out.with_suffix(".render.json")
    out_manifest.write_text(json.dumps(
        doc | {"scene_declaration": str(decl.relative_to(ROOT)),
               "scene_declaration_sha256":
                   hashlib.sha256(decl.read_bytes()).hexdigest()},
        indent=2) + "\n")
    print(f"lane B concept -> {args.out.name}  (declared in "
          f"{decl.relative_to(ROOT)})")
    print("REMEMBER: stamp the signature before this leaves the machine:")
    print(f"  python3 viz/src/stamp_frames.py <dir>  # or stamp the still directly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
