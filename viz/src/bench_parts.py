#!/usr/bin/env python3
"""Recognisable hardware for the bench rig, built inside the MJCF's envelope.

WHY THIS EXISTS
---------------
Rendered as the raw MJCF primitives, the rig is a black cylinder and an orange
disc on a white square. The owner's verdict on that was blunt and correct: it
reads as an abstract study, and a viewer cannot tell what they are looking at
without a paragraph of text underneath. A build-log asset that needs a caption
to be legible has failed at the only job it has.

So the parts here dress the same geometry as the hardware it stands for: a
**190 kv 6374 outrunner** (63 mm diameter, ~74 mm long — the motor named in
`bench_rig.xml`'s own header), a machined 6061 flywheel, an aluminium mounting
plate, and the desk clamp the MJCF describes in prose but does not model.

THE RULE THIS OBEYS, AND WHERE THE LINE IS
------------------------------------------
Every dimension that the physics can see comes from the compiled model and is
never overridden: the can is r = 31.5 mm because `rotor_can` is, the disc is
r = 75 mm and 12 mm thick because `flywheel` is, the axis is the axis. Detail is
added strictly *within* those envelopes — vents, windings, bolt circles,
chamfers, a shaft.

That is the latitude this repo has always taken and states in its README: "a
tyre with a nicer profile than the collision hull" is explicitly allowed,
precisely because no control decision is ever tuned from a render. Cosmetic
detail that dresses the same geometry does not change what a viewer believes
about the measurement.

Two things are genuinely *added* rather than dressed — the desk clamp and the
controller board. Both are static, carry no motion, and are named in
`COSMETIC_ADDITIONS` so the render manifest can declare them. A reader can
therefore always tell what is in frame that is not in the plant. Anything that
would change a dimension, add a moving part, or imply a measurement does not
belong here.
"""
from __future__ import annotations

import math

import bpy
from mathutils import Quaternion, Vector

import build_scene as bs

# Declared in the render manifest. Static set dressing that is NOT in the MJCF:
# the model calls the plate "clamped to a desk edge" and models neither the
# clamp nor the controller driving the motor, because both are ground.
COSMETIC_ADDITIONS = [
    "desk legs + apron (the MJCF desk is a floating static box with no legs)",
    "workshop floor (the bench scene has no ground plane in the model)",
    "desk clamp (MJCF describes the plate as clamped; the clamp is not modelled)",
    "controller board + CAN lead (the Little FOCer path under test; not modelled)",
    "vent slots, windings, bolt circles, chamfers and shaft (detail inside the "
    "MJCF envelope — no dimension the physics can see is changed)",
]

FINISH = {
    "alu_machined": dict(roughness=0.24, metallic=0.88, coat=0.00),
    "alu_anodised": dict(roughness=0.36, metallic=0.72, coat=0.00),
    "steel":        dict(roughness=0.20, metallic=0.95, coat=0.00),
    "copper":       dict(roughness=0.34, metallic=0.92, coat=0.00),
    "black_ano":    dict(roughness=0.40, metallic=0.55, coat=0.02),
    "pcb":          dict(roughness=0.58, metallic=0.10, coat=0.05),
    "rubber":       dict(roughness=0.88, metallic=0.00, coat=0.00),
    "wood":         dict(roughness=0.55, metallic=0.02, coat=0.06),
}

RGBA = {
    "alu":     [0.72, 0.73, 0.75, 1.0],
    "amber":   [0.949, 0.635, 0.290, 1.0],
    "ink":     [0.086, 0.137, 0.180, 1.0],
    "copper":  [0.72, 0.45, 0.20, 1.0],
    "steel":   [0.62, 0.64, 0.67, 1.0],
    "pcb":     [0.13, 0.34, 0.28, 1.0],
    "wood":    [0.30, 0.22, 0.15, 1.0],
    "black":   [0.10, 0.11, 0.12, 1.0],
}


def _mat(name: str, rgba_key: str, finish_key: str):
    return bs._principled(name, RGBA[rgba_key], FINISH[finish_key])


def _shade(obj, angle_deg: float = 32.0):
    for p in obj.data.polygons:
        p.use_smooth = True
    obj.modifiers.new("smooth", "EDGE_SPLIT").split_angle = math.radians(angle_deg)


def _cyl(r: float, half_len: float, loc, mat, parent, axis="y", verts=96,
         name="part"):
    """A cylinder whose length runs along `axis`, in the parent's frame."""
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=2.0 * half_len, vertices=verts)
    o = bpy.context.object
    o.name = name
    o.parent = parent
    o.location = Vector(loc)
    o.rotation_mode = "QUATERNION"
    if axis == "y":
        o.rotation_quaternion = Quaternion((0.7071068, -0.7071068, 0.0, 0.0))
    elif axis == "x":
        o.rotation_quaternion = Quaternion((0.7071068, 0.0, 0.7071068, 0.0))
    else:
        o.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
    o.data.materials.append(mat)
    _shade(o)
    return o


def _box(size, loc, mat, parent, name="part", rot=None):
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    o = bpy.context.object
    o.name = name
    o.parent = parent
    o.scale = Vector(size)
    o.location = Vector(loc)
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = Quaternion(rot) if rot else Quaternion((1, 0, 0, 0))
    o.data.materials.append(mat)
    _shade(o, 50.0)
    return o


# --------------------------------------------------------------- rotating half

def outrunner_can(parent, r: float, half_len: float, y: float):
    """The turning half of a 6374: vented bell, end face, bolt bosses, shaft.

    A plain cylinder is what made the first pass unreadable. What says "motor"
    to anyone who has seen one is the **vent slots with copper visible behind
    them** — so those are the detail worth spending geometry on.
    """
    m_black = _mat("bp_can", "ink", "black_ano")
    m_cu = _mat("bp_windings", "copper", "copper")
    m_alu = _mat("bp_can_alu", "alu", "alu_anodised")
    m_steel = _mat("bp_shaft", "steel", "steel")

    parts = [_cyl(r, half_len, (0, y, 0), m_black, parent, name="can")]

    # Windings, sitting just inside the vent line so they read as depth rather
    # than as a painted stripe.
    parts.append(_cyl(r * 0.93, half_len * 0.72, (0, y, 0), m_cu, parent,
                      name="windings", verts=64))

    # Vent slots: a ring of thin boxes standing proud of the can, which reads as
    # a slotted bell at this scale and costs nothing next to a boolean cut.
    n = 10
    for i in range(n):
        a = 2.0 * math.pi * i / n
        parts.append(_box(
            (r * 0.085, half_len * 0.60, r * 0.055),
            (r * 1.02 * math.cos(a), y, r * 1.02 * math.sin(a)),
            m_black, parent, name=f"vent_{i}",
            rot=Quaternion(Vector((0, 1, 0)), -a)))

    # Outer end face + bolt bosses.
    face_y = y + half_len * 0.98
    parts.append(_cyl(r * 0.99, half_len * 0.04, (0, face_y, 0), m_alu, parent,
                      name="bell_face"))
    for i in range(4):
        a = 2.0 * math.pi * i / 4 + math.pi / 4
        parts.append(_cyl(r * 0.075, half_len * 0.07,
                          (r * 0.62 * math.cos(a), face_y + half_len * 0.05,
                           r * 0.62 * math.sin(a)),
                          m_steel, parent, name=f"bell_bolt_{i}", verts=6))

    # Shaft, protruding past the bell the way a 6374's does.
    parts.append(_cyl(r * 0.20, half_len * 1.35, (0, y, 0), m_steel, parent,
                      name="shaft", verts=32))
    return parts


def flywheel(parent, r: float, half_t: float, y: float):
    """Machined 6061 disc: chamfered rim, bolt circle, raised centre boss.

    The bolt circle is what makes it read as a *machined part* rather than a
    coloured cylinder — it gives the eye something to track as it turns, which
    is the same job the index stripe does and the reason this disc exists.
    """
    m_amber = _mat("bp_disc", "amber", "alu_machined")
    m_alu = _mat("bp_disc_alu", "alu", "alu_machined")
    m_dark = _mat("bp_bore", "black", "black_ano")

    parts = [_cyl(r, half_t, (0, y, 0), m_amber, parent, name="disc", verts=128)]
    # Machined rim band: very slightly proud of the disc radius and NARROWER
    # than its thickness, so it reads as a turned edge seen side-on.
    #
    # The first attempt put a disc of radius 0.985r on each face to fake a
    # chamfer. At that radius it is not a ring, it is a lid — it covered the
    # entire amber face in aluminium grey and hid the bolt circle with it. The
    # disc rendered olive and only the rim stayed amber, which is the exact
    # inverse of the intent. Concentric primitives cannot make a ring; only
    # radius-out-and-thinner does.
    parts.append(_cyl(r * 1.004, half_t * 0.78, (0, y, 0), m_alu, parent,
                      name="disc_rim", verts=128))
    # Bolt circle — counterbores, dark so they read as holes.
    for i in range(6):
        a = 2.0 * math.pi * i / 6
        parts.append(_cyl(r * 0.075, half_t * 1.02,
                          (r * 0.60 * math.cos(a), y, r * 0.60 * math.sin(a)),
                          m_dark, parent, name=f"disc_bore_{i}", verts=24))
    # Centre boss + hub bore.
    parts.append(_cyl(r * 0.24, half_t * 1.35, (0, y, 0), m_alu, parent,
                      name="disc_boss", verts=64))
    parts.append(_cyl(r * 0.085, half_t * 1.45, (0, y, 0), m_dark, parent,
                      name="disc_bore_c", verts=32))
    return parts


# ------------------------------------------------------------------ static half

def stator_half(parent, r: float, half_len: float, y: float):
    """The bolted-down half, plus the mounting flange it hangs on."""
    m_black = _mat("bp_stator", "ink", "black_ano")
    m_alu = _mat("bp_flange", "alu", "alu_anodised")
    m_steel = _mat("bp_flange_bolt", "steel", "steel")

    parts = [_cyl(r * 0.97, half_len, (0, y, 0), m_black, parent, name="stator")]
    # Mounting flange against the plate, with a cross of bolts — the detail that
    # says "this is fastened to something" rather than floating.
    flange_y = y - half_len * 0.92
    parts.append(_cyl(r * 1.06, half_len * 0.16, (0, flange_y, 0), m_alu, parent,
                      name="flange"))
    for i in range(4):
        a = 2.0 * math.pi * i / 4
        parts.append(_cyl(r * 0.085, half_len * 0.22,
                          (r * 0.80 * math.cos(a), flange_y - half_len * 0.10,
                           r * 0.80 * math.sin(a)),
                          m_steel, parent, name=f"flange_bolt_{i}", verts=6))
    return parts


def plate_detail(parent, size, pos):
    """Corner bolts on the mounting plate. The plate itself is an MJCF geom."""
    m_steel = _mat("bp_plate_bolt", "steel", "steel")
    out = []
    for sx in (-1, 1):
        for sz in (-1, 1):
            out.append(_cyl(
                0.0045, size[1] * 1.25,
                (pos[0] + sx * size[0] * 0.82, pos[1],
                 pos[2] + sz * size[2] * 0.86),
                m_steel, parent, name=f"plate_bolt_{sx}{sz}", verts=6))
    return out


def clamp(parent, plate_x: float, plate_y: float, desk_top_z: float,
          desk_thick: float):
    """The G-clamp holding the plate to the desk.

    COSMETIC — declared. `bench_rig.xml` says the plate is "clamped flat to the
    desk" but models no clamp, because in the physics it is simply ground. With
    nothing there the plate appears to grow out of the desk, which is the single
    biggest reason the first pass did not read as a real bench setup.
    """
    m_steel = _mat("bp_clamp", "black", "black_ano")
    m_black = _mat("bp_clamp_pad", "black", "rubber")
    x = plate_x - 0.055
    jaw = 0.030
    parts = [
        # Upper jaw over the plate, lower jaw under the desk, spine joining them.
        _box((jaw, 0.012, 0.008), (x, plate_y, desk_top_z + 0.020), m_steel,
             parent, name="clamp_top"),
        _box((jaw, 0.012, 0.009), (x, plate_y, desk_top_z - desk_thick - 0.028),
             m_steel, parent, name="clamp_bot"),
        _box((0.009, 0.012, (desk_thick + 0.060) / 2),
             (x - jaw + 0.009, plate_y, desk_top_z - desk_thick / 2 - 0.010),
             m_steel, parent, name="clamp_spine"),
        _cyl(0.006, 0.016, (x + 0.008, plate_y, desk_top_z - desk_thick - 0.044),
             m_steel, parent, axis="z", name="clamp_screw", verts=24),
        _box((0.014, 0.004, 0.004),
             (x + 0.008, plate_y, desk_top_z - desk_thick - 0.058), m_black,
             parent, name="clamp_handle"),
    ]
    return parts


def controller(parent, x: float, y: float, z: float):
    """A VESC-class board and its CAN lead, off to one side.

    COSMETIC — declared. The rig's stated purpose is to characterise the
    Pi → CAN → VESC path, and that path is invisible in a frame containing only
    a motor. This is the Little FOCer named in the BoM, present as context.
    """
    m_pcb = _mat("bp_pcb", "pcb", "pcb")
    m_alu = _mat("bp_heatsink", "alu", "alu_anodised")
    m_blk = _mat("bp_cable", "black", "rubber")
    parts = [
        _box((0.042, 0.030, 0.0014), (x, y, z + 0.0014), m_pcb, parent, name="focer"),
        _box((0.022, 0.014, 0.004), (x, y, z + 0.0065), m_alu, parent,
             name="focer_heatsink"),
    ]
    # A short lead running off toward the motor — enough to imply the path.
    for i in range(7):
        t = i / 6.0
        parts.append(_cyl(0.0022, 0.010,
                          (x + 0.030 + t * 0.055, y - t * 0.010,
                           z + 0.002 + 0.004 * math.sin(t * 3.0)),
                          m_blk, parent, axis="x", name=f"cable_{i}", verts=12))
    return parts


DESK_HEIGHT_M = 0.74


def desk_legs(parent, desk_pos, desk_size, floor_z: float):
    """Four legs and an apron under the modelled desktop.

    COSMETIC — declared. `bench_rig.xml` models the desk as a single static box
    with its top at z = 0, because that is all the physics needs; it has no
    legs and nothing under it. That is invisible in a tight shot where the
    benchtop fills the lower frame, but any wider framing shows a 600 mm slab
    hanging unsupported in mid-air, which reads as a broken scene rather than a
    workshop. Legs cost nothing and are what a viewer already assumes is there.

    Nothing here touches the desktop's own dimensions or its z = 0 top surface,
    which is the datum the rig is positioned against.
    """
    m = _mat("bp_leg", "wood", "wood")
    x0, y0, z0 = desk_pos
    sx, sy, sz = desk_size
    leg_r, inset = 0.022, 0.045
    h = (z0 - sz - floor_z) / 2.0
    out = []
    for sgx in (-1, 1):
        for sgy in (-1, 1):
            out.append(_box(
                (leg_r, leg_r, h),
                (x0 + sgx * (sx - inset), y0 + sgy * (sy - inset),
                 z0 - sz - h),
                m, parent, name=f"leg_{sgx}{sgy}"))
    # Apron rail, so the legs read as a desk rather than four posts.
    out.append(_box((sx - 0.03, 0.012, 0.030),
                    (x0, y0 + sy - inset, z0 - sz - 0.055), m, parent,
                    name="apron_front"))
    return out


def gclamp(parent, desk_thick: float, grip_z: float, name: str = "gclamp"):
    """A G-clamp, built at the parent's local origin, opening along +X.

    Local frame: z = 0 is the top of the work being clamped, the throat reaches
    in from -X, and the screw comes up from below. The CALLER positions and
    rotates an empty — which matters, because a G-clamp can only reach a free
    EDGE. Placing one in the middle of a desktop puts its lower jaw inside the
    desk, which is exactly the mistake the first version of this concept made:
    two clamps floating 70 mm inboard with nothing under them to grip.

    Modelled rather than sourced. Mike asked whether a clamp model could be
    bought in; I looked and did not take one. The CC0 libraries this project
    already depends on (Poly Haven) are HDRIs, textures and architectural props,
    with no workshop tooling. Every external asset has to land in MANIFEST.json
    with a source, licence and sha256, and taking on that dependency to save
    fifty lines of primitives is a bad trade for a part seen at this size.

    C-frame, acme screw, swivel pad and tommy bar — enough silhouette to read as
    a clamp at any framing we would use.
    """
    m_body = _mat(f"{name}_body", "steel", "alu_anodised")
    m_screw = _mat(f"{name}_screw", "steel", "steel")
    m_pad = _mat(f"{name}_pad", "black", "rubber")

    reach = 0.052
    throat = grip_z + desk_thick + 0.034
    mid_z = (grip_z - desk_thick - 0.030) / 2.0
    fw = 0.010
    out = [
        _box((0.011, fw, throat / 2.0), (-reach - 0.011, 0.0, mid_z), m_body,
             parent, name=f"{name}_spine"),
        _box((reach, fw, 0.009), (-reach + 0.010, 0.0, grip_z + 0.009), m_body,
             parent, name=f"{name}_jaw_top"),
        _box((reach, fw, 0.010), (-reach + 0.010, 0.0, -desk_thick - 0.010),
             m_body, parent, name=f"{name}_jaw_bot"),
        _box((0.013, fw * 0.85, 0.005), (0.0, 0.0, grip_z + 0.004), m_body,
             parent, name=f"{name}_anvil"),
    ]
    zs = -desk_thick
    out.append(_cyl(0.0062, 0.020, (0.0, 0.0, zs - 0.020), m_screw, parent,
                    axis="z", verts=24, name=f"{name}_screw"))
    out.append(_cyl(0.0115, 0.0035, (0.0, 0.0, zs - 0.0035), m_pad, parent,
                    axis="z", verts=24, name=f"{name}_pad"))
    out.append(_cyl(0.0035, 0.030, (0.0, 0.0, zs - 0.038), m_screw, parent,
                    axis="y", verts=16, name=f"{name}_bar"))
    for sgn in (-1, 1):
        out.append(_cyl(0.0055, 0.0035, (0.0, sgn * 0.030, zs - 0.038), m_screw,
                        parent, axis="y", verts=16, name=f"{name}_barend{sgn}"))
    return out


def angle_bracket(parent, mount_y: float, riser_x, riser_z, foot_x, foot_y,
                  thick: float = 0.005):
    """An L-section running along X: horizontal foot on the desk, vertical riser.

    This is the concept the owner asked for, and it is the shape that satisfies
    both constraints at once. The riser stays a panel in the X-Z plane, thin in
    Y, so the motor's mounting face and its Y hinge axis are untouched — the
    disc still turns in the plane the board pitches in, which is what keeps the
    later arm-and-mass upgrade a real inverted pendulum instead of a turntable.

    What changes is only that the panel now has a foot to be clamped by, instead
    of balancing on its bottom edge.
    """
    m = _mat("bp_bracket", "alu", "alu_anodised")
    out = [
        _box(((riser_x[1] - riser_x[0]) / 2, thick / 2,
              (riser_z[1] - riser_z[0]) / 2),
             ((riser_x[0] + riser_x[1]) / 2, mount_y,
              (riser_z[0] + riser_z[1]) / 2), m, parent, name="riser"),
        _box(((foot_x[1] - foot_x[0]) / 2, (foot_y[1] - foot_y[0]) / 2, thick / 2),
             ((foot_x[0] + foot_x[1]) / 2, (foot_y[0] + foot_y[1]) / 2,
              thick / 2), m, parent, name="foot"),
    ]
    # Triangular gussets, approximated as thin plates. The stand's stiffness
    # requirement (first resonance above 50 Hz) is the reason a real one would
    # have them, so they belong in a concept that is arguing for this shape.
    for gx in (foot_x[0] + 0.030, foot_x[1] - 0.045):
        out.append(_box((0.028, thick / 2, 0.028),
                        (gx, mount_y + thick, 0.030), m, parent,
                        name=f"gusset_{gx:.3f}",
                        rot=None))
    return out
