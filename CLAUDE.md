# Overboard — visualization project rules

Extends the global `~/.claude/CLAUDE.md`. This repo turns **Overboard** simulation runs into
publishable visual content — video, stills and review artefacts. Role: **Digital Content
Production**. Branch prefix: **`feat/content/`**. Escalate to: **Sr. Digital Marketer**.

Handoff / onboarding doc (read first, it holds the context this file assumes):
[Handoff — Digital Content Production](https://app.notion.com/p/3a9472a5fb69811fb4bdc968830585a6).
Design of record: [V1 — Cinematic Visualization Pipeline](https://app.notion.com/p/3a9472a5fb698124927bffb8e8b690a4).

## The category rule (HARD — governs everything made here)

> 📖 **Definitions live in exactly one place:** [Overboard — Shared Vocabulary (canonical)](https://app.notion.com/p/3aa472a5fb6981ebaaa7cf2e996f1e8b).
> **Do not restate them here.** A definition kept in two places diverges within a week — that is
> what forced the Lane A/B rename. This file carries the *operational* rules only: paths, stamping,
> and what to do when you are unsure.

Every artefact this repo produces is **Sim Replay** or **Concept**. There is no unlabelled state.
`Footage` (a real camera) and `Hardware Replay` (this same scene driven by real telemetry) exist in
the canonical vocabulary and will land here at Phase 1 — **build the source tag as a parameter so
`HARDWARE` needs no code change.**

| | **Sim Replay** | **Concept** |
|---|---|---|
| Directory | `viz/scenes/replay/` | `viz/scenes/concept/` |
| Engineering numbers, HUD | **Allowed** | **Never** |
| Tense of accompanying copy | Past — it happened | **Future/subjunctive** — "what it *will* look like" |
| May depict an event as having occurred | Yes | **No** |
| Mark in frame | Source tag — `SIM` | `CONCEPT`, persistent, default-on |
| Visual freedom | Bounded by the plant | Unconstrained |

**A Replay always names its source. There is no bare "Replay."**

**The test — apply it to every frame:**

> Could a reader reproduce this frame from the committed `.otrk` plus the committed scene?
> **Yes → Sim Replay. Anything else → Concept.**

"Anything else" includes: a hand-keyed camera move, invented geometry, a pose that was nudged, a
composite, a trimmed cut that hides part of what happened. If you are unsure, it is Concept.
Uncertainty resolves downward, always.

### Slow motion is Sim Replay; retiming is not

These are not the same thing, and the difference decides whether fast events can be filmed at all.
The bench rig's identification run is **20 ms** long, and its flywheel passes 13 rev/s inside
150 ms — at 30 fps real time that is half a frame or an aliased blur. A blanket "any retiming is
Concept" would mean the fastest events could never be Sim Replay artefacts, which gets the rule exactly
backwards: the measurement is the thing we most want to show.

**Uniform slow motion stays Sim Replay** when all three hold:

1. **Integer-exact.** One rendered frame is a whole number of simulation timesteps — normally
   exactly one. No interpolation, no invented in-betweens, no duplicated frames.
2. **Uniform.** One time scale for the whole clip. No ramping, no speeding through dull stretches.
3. **Declared.** `time_scale` is recorded in the `.otrk` manifest *and* the render manifest, and
   the figure is legible on the frame.

All three together preserve the Sim Replay test exactly: a reader regenerates the sequence from the
committed track and the committed scene. **Non-uniform retiming, interpolated in-betweens, and
trims that hide are Concept**, because none of them can be reproduced that way.

### Declaring the category

**Choosing the directory IS the declaration.** It is not a metadata field someone can forget to
set, and it is not a caption that can be stripped.

```
viz/scenes/replay/     viz/src/replay/     — replay scenes
viz/scenes/concept/     viz/src/concept/     — authored scenes
```

⚠️ **Adaptation, because this repo has no `.blend` files on disk.** The rule as handed down says
"which directory you save the `.blend` in is the declaration". Today every scene here is *Python*
built procedurally by `build_scene.py` / `render_clip.py` — there are zero `.blend` files in the
tree. So the declaration is **the directory of the scene module**, and of the `.blend` too once
any are saved. Same principle, same guarantee: the path on disk carries the category.

### The signature — call it a signature, not a watermark

Concept carries a **persistent in-frame signature**, **default-on in the scene template**. Not
opt-in, not added at encode time. Burn it into frames so re-encoding, trimming or pulling a still
cannot drop it (`stamp_frames.py` exists for exactly this and was built that way deliberately).

Studios sign concept work. This is that. Do not call it a watermark in code, comments, commit
messages or copy — the word imports a defensive, anti-piracy framing that is wrong for what this
does.

### Every render emits a manifest

Recording **category** plus the **`.otrk` hash**. The hash is what makes a Sim Replay claim checkable by
someone who is not us; without it "reproducible from the committed track" is an assertion rather
than a fact. **Not built yet** — `render_clip.py` currently reads the manifest embedded in the
`.otrk` but emits nothing of its own. Until it does, no render is fully compliant. See Open items.

### Standing quota

**Every Concept publication is paired with or preceded by a Sim Replay one**, and **Sim Replay is the hero
format for milestones.** Concept never ships alone and never carries a milestone. The reason is
structural: a project whose thesis is honest engineering cannot let the pretty authored picture be
the thing that represents an achievement.

### Three former tenets, withdrawn for Concept (CEO, 2026-07-26)

These were binding on all output until 2026-07-26. The CEO has **overruled all three for Concept**:

- ~~No identifiable place~~ — withdrawn. **Seattle is explicitly wanted.**
- ~~No terrain the plant does not model~~ — withdrawn.
- ~~No photoreal rider~~ — withdrawn.

**Sim Replay keeps all three by construction**, not by agreement — Sim Replay is a replay of the plant, and
the plant is a flat rigid plane with no rider dynamics and no geography. Nothing enforces these in
Sim Replay; the category definition already excludes them.

Consequence worth stating plainly: the old "the abstraction is the disclosure" argument — a mint
stick-figure rider makes it self-evidently not real footage, so no mark is needed — **does not
survive into Concept**, because Concept may now be photoreal in a real place. The signature replaces
it. That is precisely why Concept's signature is non-optional.

## Provenance of authored assets

**When the photoreal rider asset lands, record its provenance THAT DAY** — in this repo, against a
public commit:

1. **Licence terms**, checked as adequate for a **public** repo and for commercial-adjacent use.
2. **Likeness rights**, if it was scanned from a real person.
3. **Whether it was AI-generated** — state it either way, explicitly.

On a project whose thesis is honest AI use, this question *will* be asked. It is cheap now and
expensive later. This applies to every authored Concept asset, not only the rider.

`viz/assets/MANIFEST.json` is the existing licence-compliance record — source, licence and sha256
per file — and stays current. Published material must retain the **MIT Openwheel** copyright
notice for the board meshes.

## Repo boundary

- **This repo is visualization only.** Controls, sim and hardware live in **`overboard`**
  (`~/projects/overboard`); the landing page and its markup/CSS live in **`overboard-web`**
  (`~/projects/overboard-web`). No control code here; no render code there.
- **The seam is one file: a pose track.** `overboard` writes `.otrk`, `overboard-viz` reads it,
  **and neither imports the other.** Do not add an import across that line — it is what makes a
  film provably of the run whose metrics are quoted beside it.
- **The renderer never computes physics.** It replays motion MuJoCo already computed. Visual
  liberties are allowed *precisely because* no control decision is ever tuned from a render.
- **Not owned here:** `index.html` and the page's markup and CSS. A separate session owns those,
  and several delivery constraints are enforced by that repo's CI (`check_page.py`) — no autoplay
  or loop, `preload="none"`, no CSS filter on video. Deliver files plus a written note of the
  constraints that travel with them; do not edit that repo's page.

## Git workflow — feature branch → PR → green CI (HARD)

- **Never commit to `master`.** All work goes on a **`feat/content/…`** branch and lands via PR.
- **CI success is the merge gate.** A red build cannot be merged. Do not merge red or bypass
  protection. (This repo has **no CI yet** — see Open items. Until it does, the gate is a PR that
  Mike can read, not a green tick that does not exist.)
- Open the PR with a body stating what changed and **why**, and call out any acceptance criteria
  that moved.
- **Before opening any PR**, read `docs/decisions/INDEX.md`. A stale context that violates a later
  ADR gets a red build. (Also does not exist yet — Open items.)
- **Answered ≠ Ratified.** An answer in Notion is an opinion. **Ratified** means it exists in git as
  an ADR in `docs/decisions/`, plus a CI check if it constrains code or public claims. Only
  Ratified binds anyone.

## Large binaries must never enter git

`masters/` and `out/` hold large binaries. **They must never enter git.** `out/` is scratch —
3.8 GB of it — and every byte is regenerable from committed inputs, which is the whole point of the
reproducibility requirement. Downloaded CC0 assets (`viz/assets/hdri/`, `viz/assets/textures/`) are
likewise re-fetched deterministically by `fetch_assets.py`; **the manifest is committed, the bytes
are not.**

What *is* committed and must stay committed: the `.otrk` tracks in `viz/scenes/` (small, and Sim Replay
is meaningless without them), `MANIFEST.json`, and source.

⚠️ **`masters/` is currently in violation** — 21 files, 27 MB, tracked since the repo was created,
and it is load-bearing: `masters/` being public over `raw.githubusercontent.com` is how media
reaches Notion, whose uploader accepts only a public HTTPS URL. **Do not simply delete it.** The
replacement is the pattern already proven in the controls repo — publish to a **GitHub Release**
and use the release asset URL. Migration is an Open item below; history is **not** to be rewritten
(force-push to a public repo is a one-way door, and `master` is protected).

Note for whoever does it: existing Notion embeds are safe. Notion re-hosts uploads on its own S3,
so the `raw.githubusercontent.com` URL was only ever the transport in. That CDN **caches 404s** —
push first, then fetch, or you poison the URL for minutes.

## Docs & source of truth

- **Notion is PRIMARY** for vision, design, scoping and review. The repo holds source, tracks and
  the licence trail; it does not hold the design narrative. Keep the Notion doc current in the same
  pass as the work — a render that ships without its page updated is half-delivered.
- **The review loop is a ballot, not a canvas.** Content goes to Notion for the owner to judge by
  eye, using a deliberately constrained protocol: a numbered contact sheet plus the fixed
  vocabulary *pick a cell, plus brighter/darker · tighter/wider · more/less blur · warmer/cooler*.
  The owner has stated plainly he cannot debug a render or give implementation guidance, so **he is
  never asked an open question.** Budget was ≤5 owner touches for V1.
- **Grade to a measured number, not to taste.** The page is dark-only (`--bg #0F1922`, mean
  **24.7/255**); shipped garage clips measure **15.4 / 15.2**. Load the frame in numpy, take the
  mean, compare. This has been got wrong twice, in opposite directions, by looking at the images —
  perceived brightness against a dark page is deeply unreliable.

## Model routing (project override of global)

- **Oracle = `opus5-oracle`, not `fable-oracle`.** Distill to one sharp question; one call;
  adjudicate-not-author; read-only.
- **Opus drives; Sonnet executes** well-scoped work (`sonnet-executor`, or `Explore`/
  `general-purpose` at `model: sonnet`). Opus reviews every hand-back.
- Effort is the lever: `low`/`medium` for execution, `high`/`xhigh` for oracle-grade judgment.
  Prefer lowering effort over adding scaffolding; do not add self-verification passes.

## Things that cost a debugging cycle once

Documented at length in the source and in `README.md`; summarised so they are not rediscovered.

- **MJCF `rgba` is sRGB, not linear.** Blender's Base Color is linear — passing them through
  renders amber as pale peach.
- **MuJoCo re-centres mesh vertices on load** and records it in `mesh_pos`/`mesh_quat`.
  `geom_xpos`/`geom_xmat` place the *canonical* mesh, not the STL on disk.
- **MuJoCo → Blender needs no axis conversion** (both right-handed, +Z up, scalar-first
  quaternions). Do not add a swap.
- **Insert keyframes LINEAR.** Every frame is a measurement; Bezier invents motion — which in
  Sim Replay is fabrication, not a smoothing choice.
- **Do not smooth-shade a flat end cap** — every primitive gets `EDGE_SPLIT` alongside `use_smooth`.
- **Blender 5 moved Actions to slots**; `action.fcurves` no longer exists.
- **This Blender build has no FFmpeg output.** Render PNGs, encode with `ffmpeg` separately.
- **The first Cycles run of a session spends ~2 min compiling Metal kernels** regardless of
  resolution. Do not diagnose a broken GPU from a cold render.
- **`--factory-startup` is required** for reproducibility — but **drop it for `--engine CYCLES`**,
  because the Cycles add-on is not loaded under factory startup.
- **Never stamp frames while a render is still running.** The stamp pass once overtook the render
  and marked 659 of 730 frames, leaving a silent hole in a guarantee. Stamping twice is not
  idempotent.

## Open items (named, so they are not silently inherited)

1. **`replay/` / `concept/` directories do not exist yet**, and the 21 shipped artefacts in
   `masters/` are unclassified. Classifying them is a prerequisite to the quota meaning anything.
2. **Render manifest (category + `.otrk` hash) is not implemented.**
3. **Concept signature is not implemented as a default-on scene-template element.**
4. **`masters/` migration to GitHub Releases** (see above). No history rewrite.
5. **No CI**, so "green CI is the gate" currently has nothing to enforce it.
6. **No `docs/decisions/`** in this repo or either sibling — nothing is Ratified anywhere.
7. **No LICENCE.** The repo is public, therefore all-rights-reserved — the opposite of the
   build-in-public intent. Must be MIT-compatible (it vendors MIT Openwheel meshes).
8. **No shot presets.** `render_clip.py` takes a dozen camera and grade arguments; every shipped
   clip used a different combination that exists only in commit messages and the handoff doc.
   Re-rendering a shipped asset today means guessing. Fix is a `SHOTS` dict; the values are
   recorded in handoff §3.1.
