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
| Review artefact | `contact_sheet.py` | any python with Pillow | `out/V1.0_contact_sheet.png` |

Only `build_scene.py` runs inside Blender. Keeping the other three out of it means they are
ordinary Python that can be run and debugged without launching a 3D application.

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

## Reproducibility

Renders run `--factory-startup`, so the image depends on this repo and the asset manifest, not on
whatever preferences and add-ons happen to be on the machine. Cycles is the one exception: its
add-on is not loaded under factory startup, so final renders drop the flag.

`MANIFEST.json` records every external asset's source, licence and sha256. Everything is CC0
(Poly Haven) except the board meshes themselves, which are **MIT (Openwheel)** and require the
copyright notice to be retained in published material.
