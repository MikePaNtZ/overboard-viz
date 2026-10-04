# overboard-viz — project notes

Offline renders of Overboard sim runs, for the moore-mike.com blog. The renderer replays motion
MuJoCo computed; it does not compute physics.

Earlier process rules (roles, escalation, category stamping) were archived on 2026-10-03 (git tag
`archive/pre-reset-2026-10-03`). They are historical and do not bind any work.

## Layout
- `viz/src/` — Blender pipeline (`build_scene.py`, `render_clip.py`) and track export.
- `out/` — gitignored output. `out/carve-lab/` is the Carve Lab progress page.
- `ops/serve-carve-lab.sh` — serves `out/carve-lab` to the tailnet over HTTPS
  (loopback file server with Range support + `tailscale serve`, port 8448).

## Notes
- Never check in renders or binaries.
- Track export needs the controls repo's venv: `~/projects/overboard/.venv/bin/python`.
