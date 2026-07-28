# overboard-viz

Cinematic renders of the [Overboard](https://github.com/MikePaNtZ/overboard) self-balancing
onewheel — sim trajectories and, later, hardware logs, turned into publishable footage.

Third repo in the set, alongside `overboard` (controls + sim) and `overboard-web` (landing page).
Design doc: **V1 — Cinematic Visualization Pipeline** in Notion.

## The rule that keeps this repo honest

**The renderer never computes physics.** It replays motion MuJoCo already computed. Visual
liberties — a rim light that is not in the room, a tyre with a nicer profile than the collision
hull — are explicitly allowed, precisely because no control decision is ever tuned from a render.

One file crosses the boundary: a pose track. `overboard` writes it, `overboard-viz` reads it,
and neither imports the other.

## Quick start

```bash
python3 viz/src/fetch_assets.py
```

```bash
~/projects/overboard/.venv/bin/python viz/src/export_pose.py
```

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --python viz/src/build_scene.py -- --variant all --engine EEVEE
```

```bash
~/projects/overboard/.venv/bin/python viz/src/contact_sheet.py
```

That produces `out/V1.0_contact_sheet.png` — the nine-cell review sheet. Finals are the same
build with `--engine CYCLES --samples 256` at full resolution.

## Pipeline

| Stage | Script | Runs under | Output |
|---|---|---|---|
| Acquire assets | `fetch_assets.py` | system python | CC0 HDRI + textures, `MANIFEST.json` |
| Read the model | `export_pose.py` | the **controls repo's venv** (needs `mujoco`) | `viz/scenes/board_rest_pose.json` |
| Build + render | `build_scene.py` | **Blender** (`bpy`) | `out/variants/*.png` |
| Render a clip | `render_clip.py` | **Blender** (`bpy`) | PNG sequence + `*.render.json` |
| Burn the mark | `stamp_frames.py` | any python with Pillow | the same PNGs, marked |
| Review artefact | `contact_sheet.py` | any python with Pillow | `out/V1.0_contact_sheet.png` |

Only the two Blender stages run inside Blender. Keeping the others out of it means they are
ordinary Python that can be run and debugged without launching a 3D application. `delivery.py`
is imported by both sides and depends on neither — it holds the handful of numbers the renderer
and the stamp pass have to agree on.

```bash
~/projects/overboard/.venv/bin/python viz/tests/test_delivery.py
```

## Delivering vertical (9:16)

Short-form is watched vertically and muted. `render_clip.py --aspect 9:16` emits it from the same
scene, the same track and the same grade as the 16:9 cut.

```bash
blender --background --factory-startup --python viz/src/render_clip.py -- \
    --track viz/scenes/cruise.otrk.npz --scene waterfront --aspect 9:16 --out out/clip_v.mp4
python3 viz/src/stamp_frames.py out/clip_v --category sim-replay
```

**What the vertical frame is.** Not "render taller". Blender's default `AUTO` sensor fit applies
the lens's field of view to the larger image dimension, so changing only `resolution_y` swings a
50 mm lens onto the vertical axis and gives a much *wider* shot in which the board is a speck.
Instead the vertical frame is defined from the landscape one: **the whole 16:9 picture, uncropped,
placed inside the safe area.** Sensor height sets the band's size, lens shift sets its position;
the platform's interface bands end up filled with real sky and ground it may cover freely.

**Two things this costs, both real.**

- Horizontally it is a genuine crop — the vertical frame spans **47%** of the landscape frame's
  width. A tracking shot that drifts harmlessly in 16:9 can walk the subject out of the vertical
  one. Every render prints a **framing report** (measured, and recorded in the render manifest)
  giving the subject's extreme image coordinates and warning if it leaves frame or enters the
  interface bands. Read it before committing to 500 frames.
- The safe area is **not** centred — bottom fifth, top eighth — so the re-frame needs the lens
  shift as well as the sensor. With the sensor alone the composition is the right size and rides
  3.75% low, which is enough to put a board's contact patch under a caption.

**Stamp the vertical frames, never the landscape ones you then crop.** Both mark placements sit
outside the middle 31.6% that a centre 9:16 crop keeps, so "render once, stamp once, crop per
channel" delivers a vertical cut that looks *deliberately* unmarked — worse than one that never
carried a mark. `stamp_frames.py` places from the frame's own shape and refuses outright if the
mark would land under the interface; `viz/tests/test_delivery.py` asserts both.

## Things that are easy to get wrong here

These each cost a debugging cycle once. They are documented at length in the source; summarised
so the next person does not rediscover them.

- **MJCF colours are sRGB, not linear.** `rgba="0.949 0.635 0.290"` is `#F2A24A` over 255.
  Blender's Base Color is linear, so passing them through renders amber as pale peach.
- **MuJoCo re-centres mesh vertices on load** and records the transform in `mesh_pos`/`mesh_quat`.
  `geom_xpos`/`geom_xmat` place the *canonical* mesh, not the STL on disk, so that transform has
  to be composed out or every shell lands rotated. `export_pose.py` does this.
- **MuJoCo → Blender needs no axis conversion.** Both are right-handed, +Z up, with scalar-first
  `(w,x,y,z)` quaternions. Do not add a swap.
- **Do not smooth-shade a flat end cap.** Doing it to the hub disc turned it into a chrome
  eyeball. Every primitive gets an `EDGE_SPLIT` alongside `use_smooth`.
- **First Cycles run compiles Metal kernels** and takes ~2 minutes regardless of resolution.
  Every run after that is seconds. Do not conclude the GPU is broken from a cold first render.
- **Renders are not byte-reproducible, so never diff them by hash.** EEVEE on Metal disagrees with
  *itself*: rendering the same frame twice from the same commit gave 32 differing pixels out of
  2,073,600, each off by one 255th. Compare with a pixel diff and a threshold instead. Measured
  2026-07-28 on the bench scene, frame 150, and worth knowing before you spend an afternoon
  hunting for the change that "broke" a render.

## Reproducibility

Renders run `--factory-startup`, so the image depends on this repo and the asset manifest, not on
whatever preferences and add-ons happen to be on the machine. Cycles is the one exception: its
add-on is not loaded under factory startup, so final renders drop the flag.

`MANIFEST.json` records every external asset's source, licence and sha256. Everything is CC0
(Poly Haven) except the board meshes themselves, which are **MIT (Openwheel)** and require the
copyright notice to be retained in published material.

## Licence

**MIT** — see [LICENSE](LICENSE), which also carries the third-party attribution (Openwheel MIT,
Poly Haven CC0) and states plainly what these renders are.

Every frame here is a **simulation** replaying motion MuJoCo computed in the `overboard` repo.
None is a photograph, and none depicts hardware that has been built or ridden.
