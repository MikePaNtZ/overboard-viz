#!/usr/bin/env python3
"""Build the V1.0 garage hero shot in Blender and render it.

Runs headless, always:

    blender --background --factory-startup --python viz/src/build_scene.py -- [args]

`--factory-startup` is not incidental. V1 §2 makes headless reproducibility a
hard requirement, and starting from the user's saved preferences would make
the render depend on whatever add-ons and colour settings happen to be on this
laptop. Factory startup means this file plus the asset manifest are the whole
description of the image.

The one exception: the Cycles add-on is not loaded under factory startup, so
`--engine CYCLES` has to run without the flag. Preview and final renders
therefore do not have identical startup conditions — worth knowing before
trusting a preview as a proxy for a final.

Scene composition, deliberately minimal:
  · the board, placed from the MuJoCo model via board_rest_pose.json
  · a benchtop slab, top surface at z = 0 so the board rests on it
  · the `garage` HDRI doing everything else — key light, bounce, and the
    visible background

There is no floor, no walls and no bench legs. Modelling them would mean
competing with an HDRI that already contains a real garage floor, real walls
and real light coming through a real roll-up door, and losing. This is the
"solve the missing environment with shot design, not geometry" principle from
V1 §3, and it is why the garage scene costs hours instead of weeks.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Quaternion, Vector

ROOT = Path(__file__).resolve().parents[2]
POSE = ROOT / "viz/scenes/board_rest_pose.json"
HDRI = ROOT / "viz/assets/hdri/garage_4k.hdr"
WOOD = ROOT / "viz/assets/textures/wood_table_worn"

# How each MuJoCo material should behave as a real surface. MuJoCo carries
# colour but its `specular`/`shininess` are a viewport approximation, not PBR,
# so the finish is specified here — this is the one place in the pipeline where
# an aesthetic judgement lives, and it is deliberately small enough to read.
FINISH = {
    # Powder-coated aluminium rails and battery/controller enclosures.
    "shell_mat":    dict(roughness=0.34, metallic=0.15, coat=0.20),
    # ABS bumpers. Injection-moulded plastic: softer highlight, no coat.
    "bumper_mat":   dict(roughness=0.44, metallic=0.00, coat=0.00),
    # Grip tape. The most matte thing on the board by a wide margin — getting
    # this wrong is the single fastest way to make the render look like CG.
    "footpad_mat":  dict(roughness=0.96, metallic=0.00, coat=0.00, grip=True),
    # Anodised aluminium electronics platform.
    "platform_mat": dict(roughness=0.30, metallic=0.80, coat=0.00),
    # Tyre rubber. A onewheel tyre is very nearly a slick — fine sipes, no
    # blocky tread — so plain rubber is *accurate*, not a shortcut. This is the
    # answer to V1 §12's open question: no sourced tread mesh is needed.
    "tire_mat":     dict(roughness=0.74, metallic=0.00, coat=0.00),
}


# ---------------------------------------------------------------- utilities

def _set(node, name: str, value) -> None:
    """Set a shader input by name, tolerating Blender's renames across versions."""
    if name in node.inputs:
        node.inputs[name].default_value = value


def _clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _srgb_to_linear(c: float) -> float:
    """The MJCF's rgba values are sRGB, despite looking like plain floats.

    `overboard_onewheel.xml` documents its palette as hex — ink #16232E, amber
    #F2A24A — and writes them as bytes over 255: amber becomes
    `rgba="0.949 0.635 0.290"`. Blender's Base Color socket is *linear*, so
    passing those through unconverted renders #F2A24A as pale peach and the
    navy shells as light slate. This is the difference between a board that
    looks like a product and one that looks like untextured CG, and it is
    invisible until you compare against the brand colour directly.
    """
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _principled(name: str, rgba, finish: dict):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    lin = [_srgb_to_linear(c) for c in rgba[:3]]
    _set(b, "Base Color", (lin[0], lin[1], lin[2], 1.0))
    _set(b, "Roughness", finish["roughness"])
    _set(b, "Metallic", finish["metallic"])
    _set(b, "Coat Weight", finish["coat"])

    if finish.get("grip"):
        # Grip tape is grit glued to a deck. Rendered as a flat matte slab it
        # reads as plastic; a fine high-frequency bump is what sells it, and it
        # is the closest thing the board has to a texture the eye can land on.
        nt = mat.node_tree
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 780.0
        noise.inputs["Detail"].default_value = 2.0
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.30
        bump.inputs["Distance"].default_value = 0.0006
        nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return mat


def _textured(name: str, folder: Path, uv_scale: float, tint: float = 1.0):
    """A PBR material from a Poly Haven texture folder (Diffuse/Rough/nor_gl).

    `tint` multiplies the albedo down. Needed because a mid-grey concrete
    texture lit by a bright HDRI renders *lighter* than the same garage's own
    floor inside that HDRI — so the subject ends up standing on a white
    cyclorama with a dim garage pasted behind it. Matching the plane's value to
    the environment map's floor is what fuses the two into one room.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    b = nt.nodes["Principled BSDF"]

    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    # Accept a per-axis scale. A plane's UVs stay 0..1 across the mesh no matter
    # how far it is stretched, so a single number on a long thin strip repeats
    # the texture every few metres along its length and every few centimetres
    # across it — which reads as a smeared, featureless surface.
    sc = uv_scale if isinstance(uv_scale, (tuple, list)) else (uv_scale,) * 3
    mapping.inputs["Scale"].default_value = (sc[0], sc[1], sc[2] if len(sc) > 2 else 1.0)
    nt.links.new(coord.outputs["UV"], mapping.inputs["Vector"])

    def tex(fname: str, non_color: bool):
        p = folder / fname
        if not p.exists():
            return None
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(str(p))
        if non_color:
            n.image.colorspace_settings.name = "Non-Color"
        nt.links.new(mapping.outputs["Vector"], n.inputs["Vector"])
        return n

    if (d := tex("Diffuse.jpg", False)):
        if tint != 1.0:
            # VectorMath rather than the Mix node. Mix exposes a Factor/A/B
            # socket per data type under the same names, so index lookups are
            # easy to get wrong; VectorMath has exactly two vector inputs and
            # no ambiguity. (Measured: the Mix version was in fact working —
            # this is chosen for legibility, not to fix a bug.)
            mul = nt.nodes.new("ShaderNodeVectorMath")
            mul.operation = "MULTIPLY"
            mul.inputs[1].default_value = (tint, tint, tint)
            nt.links.new(d.outputs["Color"], mul.inputs[0])
            nt.links.new(mul.outputs["Vector"], b.inputs["Base Color"])
        else:
            nt.links.new(d.outputs["Color"], b.inputs["Base Color"])
    if (r := tex("Rough.jpg", True)):
        nt.links.new(r.outputs["Color"], b.inputs["Roughness"])
    if (n := tex("nor_gl.jpg", True)):
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(n.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    return mat


# ------------------------------------------------------------ scene pieces

def _world(hdri: Path, strength: float, rot_deg: float) -> None:
    world = bpy.data.worlds.new("garage")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0, 0, math.radians(rot_deg))
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(hdri))
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputWorld")

    nt.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def _board(pose: dict) -> list:
    """Import each STL and place it from the exported world transform."""
    objs = []
    for g in pose["geoms"]:
        if g["kind"] == "mesh":
            before = set(bpy.data.objects)
            bpy.ops.wm.stl_import(filepath=g["file"], global_scale=g["scale"])
            new = list(set(bpy.data.objects) - before)
            if not new:
                raise RuntimeError(f"STL import produced nothing: {g['file']}")
            obj = new[0]
        else:  # the tyre — see _tyre()
            obj = _tyre(g)

        obj.name = g["name"]
        obj.location = Vector(g["pos"])
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = Quaternion(g["quat"])  # (w,x,y,z), as-is

        mat_name = g.get("material", {}).get("name", "shell_mat")
        rgba = g.get("material", {}).get("rgba", [0.5, 0.5, 0.5, 1])
        obj.data.materials.append(
            _principled(f"{mat_name}_{obj.name}", rgba,
                        FINISH.get(mat_name, FINISH["shell_mat"])))

        # Shells are CAD: faceted STL normals read as chunky under a real
        # highlight. Shade smooth with an angle threshold so the chamfers and
        # fastener bosses keep their hard edges.
        for p in obj.data.polygons:
            p.use_smooth = True
        obj.modifiers.new("smooth", "EDGE_SPLIT").split_angle = math.radians(40)
        objs.append(obj)
    return objs


def _tyre(g: dict):
    """A crowned tyre + hub, from the MJCF cylinder's own dimensions.

    MuJoCo models the wheel as a plain cylinder because contact only needs a
    radius. Photographed, a square-edged cylinder reads instantly as sim
    geometry — it is the one part of the board that is not real CAD, and it is
    also the largest object in frame. A heavy bevel on the two rim circles
    gives the barrel profile a real onewheel tyre has, which is the whole fix.
    """
    r, half_w = g["radius"], g["half_width"]
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=r, depth=half_w * 2)
    tyre = bpy.context.object

    bev = tyre.modifiers.new("crown", "BEVEL")
    # A onewheel tyre is fat but its crown is flatter than a doughnut: too much
    # bevel here and the board reads as sitting on a rubber roller.
    bev.width = r * g.get("crown", 0.18)
    bev.segments = 16
    bev.limit_method = "ANGLE"
    bev.angle_limit = math.radians(30)

    # Hub motor face: a shallow metal disc inset on each side. Without it the
    # tyre reads as a solid rubber puck rather than a direct-drive hub.
    hub_mat = _principled("hub_mat", [0.42, 0.44, 0.47, 1],
                          dict(roughness=0.52, metallic=0.85, coat=0.0))
    for side in (-1, 1):
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=96, radius=r * 0.55, depth=half_w * 0.24,
            location=(0, 0, side * half_w * 0.92))
        hub = bpy.context.object
        hub.data.materials.append(hub_mat)
        hub.parent = tyre
        for p in hub.data.polygons:
            p.use_smooth = True
        # Without this the flat motor face is smoothed into a dome.
        hub.modifiers.new("smooth", "EDGE_SPLIT").split_angle = math.radians(30)
    return tyre


def _bench(width=2.4, depth=0.95, thick=0.055):
    """Workbench slab with its top surface exactly at z = 0.

    The board's rest pose puts the tyre contact point at z = 0, so aligning the
    benchtop there means the board sits on the bench with no fudge factor and
    no hand-placed offset.
    """
    # size=1 already spans -0.5..0.5, so scale is the full extent, not the
    # half-extent. Halving it here once made a 2.4 m workbench render as a
    # 1.2 m chopping board sitting under a 0.94 m machine.
    bpy.ops.mesh.primitive_cube_add(size=1)
    top = bpy.context.object
    top.name = "benchtop"
    top.scale = (width, depth, thick)
    top.location = (0, 0, -thick / 2)
    bpy.ops.object.transform_apply(scale=True)

    bev = top.modifiers.new("edge", "BEVEL")
    bev.width, bev.segments = 0.004, 3

    top.data.materials.append(_textured("bench_wood", WOOD, uv_scale=1.6))
    return top


def _camera(lens: float, az_deg: float, el_deg: float, dist: float,
            fstop: float, target=(-0.02, 0.0, 0.16)):
    """Place a camera on a spherical rig aimed at the board.

    Distance is scaled with focal length by the caller so that changing the
    lens changes *compression* — the actual look — rather than just how big the
    board is in frame. Aiming and focus both go through one empty via a
    TRACK_TO constraint, which is the standard scripted-camera pattern and
    means focus can never drift off the subject.
    """
    tgt = bpy.data.objects.new("focus_target", None)
    bpy.context.collection.objects.link(tgt)
    tgt.location = target

    az, el = math.radians(az_deg), math.radians(el_deg)
    offset = Vector((math.cos(el) * math.cos(az),
                     math.cos(el) * math.sin(az),
                     math.sin(el))) * dist

    cam_data = bpy.data.cameras.new("hero")
    cam_data.lens = lens
    cam_data.dof.use_dof = True
    cam_data.dof.focus_object = tgt
    cam_data.dof.aperture_fstop = fstop
    cam = bpy.data.objects.new("hero", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = Vector(target) + offset

    c = cam.constraints.new("TRACK_TO")
    c.target = tgt
    c.track_axis, c.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"

    bpy.context.scene.camera = cam
    return cam


def _kicker(energy: float, warm: bool = False, loc=(1.5, 1.7, 1.5), size: float = 1.6):
    """A soft rim light from behind, opposite the HDRI's key.

    The board is navy, black and near-black sitting in a warm mid-tone garage;
    without an edge it collapses into a silhouette. This is a product-photo
    move, not a physics claim — and it is exactly the kind of liberty V1 §1
    permits, since nothing is tuned from a render.
    """
    light = bpy.data.lights.new("kicker", "AREA")
    light.energy, light.size = energy, size
    # Cool by default, to separate the board from a warm garage. At dusk the
    # sign flips: the key IS warm and low, so a cool rim reads as a mistake —
    # the figure needs a warm edge from the sun side or it goes to silhouette
    # and the character, which is the whole point of showing a rider, is lost.
    light.color = (1.0, 0.72, 0.42) if warm else (0.85, 0.90, 1.0)
    obj = bpy.data.objects.new("kicker", light)
    bpy.context.collection.objects.link(obj)
    obj.location = loc
    c = obj.constraints.new("TRACK_TO")
    c.target = bpy.data.objects["focus_target"]
    c.track_axis, c.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    return obj


# ------------------------------------------------------------------ render

def _render(path: Path, res: tuple, engine: str, samples: int, exposure: float,
            look: str = "AgX - Punchy"):
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.film_transparent = False
    scene.render.filepath = str(path)
    scene.render.image_settings.file_format = "PNG"

    # AgX is Blender's photographic tone map; it is what keeps the bright
    # doorway in the HDRI from clipping to a flat white hole. Its cost is
    # desaturation as things get brighter — which on a first pass turned the
    # amber bumpers cream and the navy shells pale slate. "Punchy" pulls the
    # brand palette back without giving up the highlight rolloff.
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = look
    scene.view_settings.exposure = exposure

    if engine == "CYCLES":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.cycles.device = "GPU"
        prefs = bpy.context.preferences.addons.get("cycles")
        if prefs:
            prefs.preferences.compute_device_type = "METAL"
            prefs.preferences.get_devices()
            for d in prefs.preferences.devices:
                # Enable the Metal GPU only. Adding the CPU to the pool sounds
                # free but makes Cycles split tiles onto a device an order of
                # magnitude slower, and the fast device ends up waiting.
                d.use = (d.type == "METAL")
            print("  cycles devices:",
                  [d.name for d in prefs.preferences.devices if d.use])
    else:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = samples
        for attr, val in (("use_raytracing", True), ("use_shadows", True)):
            if hasattr(scene.eevee, attr):
                setattr(scene.eevee, attr, val)

    scene.render.use_motion_blur = False  # a still; nothing is moving
    bpy.ops.render.render(write_still=True)


# ------------------------------------------------------------------ variants

BASE = dict(hdri_strength=1.05, hdri_rot=115.0, exposure=0.0, kicker=25.0,
            lens=50.0, az=235.0, el=6.0, fstop=2.0, dist=1.35,
            look="AgX - Punchy", crown=0.18)

# One variable per row, exactly as the review protocol in V1 §7 requires: the
# owner picks a cell, so the cells have to differ in one legible way.
VARIANTS = {
    # Row A — lighting: where the garage is pointing, and how bright.
    "A1": dict(hdri_rot=0,    exposure=0.0,  label="light: door behind camera"),
    "A2": dict(hdri_rot=115,  exposure=0.3,  label="light: door camera-left, +0.3EV"),
    "A3": dict(hdri_rot=205,  exposure=-0.2, label="light: backlit, -0.2EV"),
    # Row B — framing: lens compression at constant subject size.
    "B1": dict(lens=35, dist=0.95, el=4,  label="framing: 35mm, low"),
    "B2": dict(lens=50, dist=1.35, el=6,  label="framing: 50mm (base)"),
    "B3": dict(lens=85, dist=2.30, el=9,  label="framing: 85mm, compressed"),
    # Row C — materials: how much the board reads as a finished product.
    "C1": dict(mat_scale=0.55, coat_boost=0.60, label="material: showroom gloss"),
    "C2": dict(mat_scale=1.00, coat_boost=0.0,  label="material: base finish"),
    "C3": dict(mat_scale=1.55, coat_boost=0.0,  label="material: utilitarian matte"),
}


def build(cfg: dict) -> None:
    _clear_scene()
    pose = json.loads(POSE.read_text())

    if (s := cfg.get("mat_scale", 1.0)) != 1.0:
        for f in FINISH.values():
            f["roughness"] = min(0.99, f["roughness"] * s)
    if (c := cfg.get("coat_boost", 0.0)):
        for name in ("shell_mat", "bumper_mat"):  # not the grip tape, ever
            FINISH[name]["coat"] = c

    for g in pose["geoms"]:
        if g["kind"] == "cylinder":
            g["crown"] = cfg["crown"]

    _world(HDRI, cfg["hdri_strength"], cfg["hdri_rot"])
    _board(pose)
    _bench()
    _camera(cfg["lens"], cfg["az"], cfg["el"], cfg["dist"], cfg["fstop"])
    _kicker(cfg["kicker"])


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="B2", help="a VARIANTS key, or 'all'")
    ap.add_argument("--engine", default="EEVEE", choices=["EEVEE", "CYCLES"])
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--out", type=Path, default=ROOT / "out/variants")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override any config key, e.g. --set kicker=0 --set lens=70")
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    args = ap.parse_args(argv)

    overrides = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        overrides[k] = float(v) if v.replace(".", "").replace("-", "").isdigit() else v

    keys = list(VARIANTS) if args.variant == "all" else [args.variant]
    for k in keys:
        cfg = {**BASE, **VARIANTS[k], **overrides}
        print(f"\n=== {k}: {cfg['label']} ===", flush=True)
        build(cfg)
        _render(args.out / f"{k}{args.tag}.png", (args.width, args.height),
                args.engine, args.samples, cfg["exposure"], cfg["look"])
    print(f"\nrendered {len(keys)} variant(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
