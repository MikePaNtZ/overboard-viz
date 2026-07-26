#!/usr/bin/env python3
"""Download every external asset the V1.0 garage scene needs, and write the
licence manifest that goes with it.

Everything here is Poly Haven, which is CC0: no attribution required, no
restrictions, and — the reason it beat the BlenderKit garage scene that the
scoping doc originally named — no account. Anonymous HTTP, so this script is
the whole acquisition step and it reruns unattended on any machine.

The manifest it writes is not bookkeeping for its own sake: V1 §9 makes
"every published frame is reproducible, with every asset's source and licence
recorded" a hard requirement, because the program is public.

    python3 viz/src/fetch_assets.py
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

API = "https://api.polyhaven.com"
ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "viz" / "assets"

# Poly Haven slug -> what it is here. Resolutions are chosen for a hero still on
# a laptop: 4k HDRI is plenty when the background sits behind an f/2 lens, and
# 2k textures are plenty for a benchtop that is mostly under the board.
HDRIS = {
    "garage": dict(res="4k", fmt="hdr", why="Key light and background. A real home garage — roll-up doors, concrete, junk shelving. Chosen over autoshop_01 (reads as a commercial dealership) and carpentry_shop_02 (wrong trade, very yellow)."),
    "the_sky_is_on_fire": dict(res="4k", fmt="hdr", why="Waterfront promenade at twilight — paved path, sea, railing, low warm sun. Dusk is the point: grading midday daylight down to sit on a dark page made it murky, whereas genuinely low light is dark for a reason. Rotated so the camera faces the water; there are apartment blocks on the far side and they stay out of frame, per the no-identifiable-place rule."),
    "approaching_storm": dict(res="4k", fmt="hdr", why="Outdoor scene for the shuttle run. Deliberately ANONYMOUS: open field, worn path, big sky, no skyline, landmarks, benches or pedestrians. Depicting an identifiable place would assert the board had been taken there, which it has not. Chosen over abandoned_pathway (bare winter trees, buildings on the horizon)."),
}
TEXTURES = {
    "wood_table_worn": dict(res="2k", fmt="jpg", why="Workbench top. In focus directly under the board, so it is the one texture that has to hold up."),
    "concrete_floor_worn_001": dict(res="2k", fmt="jpg", why="Garage floor. Almost entirely defocused; present so the contact shadow has somewhere to land."),
    "aerial_grass_rock": dict(res="2k", fmt="jpg", why="Outdoor ground. Texture only — the plant is a flat plane, so the render may not add camber, gravel or slope it does not model."),
    "dirt_floor": dict(res="2k", fmt="jpg", why="Alternate outdoor ground, path-like. Whichever of the two reads better under the anonymous-outdoor rule."),
}
# Which PBR channels to pull. Poly Haven names vary a little by asset, so these
# are tried in order and misses are tolerated rather than fatal.
MAPS = ["Diffuse", "diffuse", "Rough", "rough", "nor_gl", "Displacement"]


# Poly Haven's CDN 403s the stock "Python-urllib/x.y" agent. Any real UA is
# accepted; this one identifies the project so their logs are honest.
UA = {"User-Agent": "overboard-viz/1.0 (+https://github.com/MikePaNtZ/overboard)"}


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _download(url: str, dest: Path) -> str:
    """Fetch to dest unless already present. Returns the sha256 either way."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"  ↓ {dest.name}")
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
            f.write(r.read())
    else:
        print(f"  · {dest.name} (cached)")
    return hashlib.sha256(dest.read_bytes()).hexdigest()


def _record(entries: list, slug: str, kind: str, spec: dict, files: dict) -> None:
    entries.append({
        "slug": slug,
        "kind": kind,
        "source": "Poly Haven",
        "url": f"https://polyhaven.com/a/{slug}",
        "licence": "CC0",
        "attribution_required": False,
        "used_for": spec["why"],
        "files": files,
    })


def main() -> int:
    entries: list = []

    for slug, spec in HDRIS.items():
        print(f"HDRI {slug}")
        files = _get_json(f"{API}/files/{slug}")
        url = files["hdri"][spec["res"]][spec["fmt"]]["url"]
        dest = ASSETS / "hdri" / f"{slug}_{spec['res']}.{spec['fmt']}"
        _record(entries, slug, "hdri", spec,
                {dest.name: _download(url, dest)})

    for slug, spec in TEXTURES.items():
        print(f"texture {slug}")
        try:
            files = _get_json(f"{API}/files/{slug}")
        except Exception as e:  # a renamed slug should not kill the run
            print(f"  ! skipped: {e}")
            continue
        got = {}
        for m in MAPS:
            try:
                url = files[m][spec["res"]][spec["fmt"]]["url"]
            except KeyError:
                continue
            dest = ASSETS / "textures" / slug / f"{m}.{spec['fmt']}"
            got[f"{slug}/{dest.name}"] = _download(url, dest)
        if got:
            _record(entries, slug, "texture", spec, got)

    # The board's own meshes are not downloaded — they live in the controls
    # repo — but they are the one asset in frame that is NOT CC0, so the
    # manifest has to carry them or it is not a compliance record.
    entries.append({
        "slug": "openwheel",
        "kind": "mesh",
        "source": "Openwheel (via overboard repo, sim/models/meshes/openwheel/)",
        "url": "https://github.com/MikePaNtZ/overboard",
        "licence": "MIT",
        "attribution_required": True,
        "attribution": "Openwheel — MIT. Copyright notice must be retained; see meshes/openwheel/NOTICE.md.",
        "used_for": "The board itself: enclosures, footpads, bumpers, electronics platform.",
        "files": {},
    })

    manifest = ASSETS / "MANIFEST.json"
    manifest.write_text(json.dumps({
        "note": "Asset provenance for every published Overboard render. See V1 §9.",
        "assets": entries,
    }, indent=2) + "\n")
    print(f"\nwrote {manifest.relative_to(ROOT)}  ({len(entries)} assets)")

    needs_credit = [e for e in entries if e["attribution_required"]]
    if needs_credit:
        print("\n⚠️  Published material must credit:")
        for e in needs_credit:
            print(f"   - {e.get('attribution', e['slug'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
