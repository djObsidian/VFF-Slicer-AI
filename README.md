# VFF Slicer

Non‑planar slicing pre/post‑processor for **3‑axis FDM printers**. It takes an
STL, builds a **growth‑direction field** inside the model, deforms the mesh so
its natural layers become flat, lets an ordinary planar slicer (PrusaSlicer,
Cura, …) cut it, and then transforms that planar G‑code back into **curved
layers that follow the part's shape** — no stair‑stepping on curved tops, and
overhangs that print without support.

🇷🇺 [Русская версия — README_ru.md](README_ru.md) ·
🛠 [Internals & full reference — ARCHITECTURE.md](ARCHITECTURE.md)

> **Status: experimental.** The deformation pipeline is validated end‑to‑end on
> real sliced G‑code (the inverse puts 98.7 % of the toolpath back inside the
> original shape), and a real print is in progress. It is not a one‑click
> slicer: thin features and self‑intersecting meshes still need manual tuning.
> See [Limitations](#limitations--backlog).

---

## The backstory

I'd wanted a non‑planar slicer for years. I watched CNC Kitchen's
conical‑slicing experiments, read the papers, drooled over the demos — but none
of it generalised past one trick at a time. FullControl GCode had its
90°‑overhang party pieces, but those don't scale and the thin in‑air beads curl
up as the plastic shrinks.

Then one boring afternoon at an old job I opened my notes and made a shitpost.
It went straight into the drawer.

![First napkin sketch: grow a build‑direction field up from the bed, follow the
natural layers](docs/images/initial_shitpost1.png)
![Second sketch: surface approximation, the tilt‑clamp on the build vector, voxel
size = h_min/2](docs/images/initial_shitpost2.png)

The idea is simple: take the model, voxelize it, find the bed, flood everything
that touches the bed, run the fill outward, build vector fields from the fill
iterations, clamp the turn so you don't smash the print head, build surfaces
from those fields, then steal CNC Kitchen's conical trick — **deform → slice
with a normal slicer → deform the G‑code back** → ??? → profit.

But I was scared of it. I don't *like* programming; I've spent years trying to
escape the scary world of code, and it keeps chasing me down. So I chickened
out and the idea went back in the drawer.

Years passed. The **S4 Slicer** came out — almost exactly this — but it wants a
supercomputer to run in any reasonable time and is built around its own bespoke
4‑axis printer. Close, but NOT IT.

Then work handed me ~~a machine gun~~ Claude Code. I fed it the idea AND OFF WE
WENT.

Where we are now: we can print bridges and round holes.

![Bridge test sliced with non‑planar layers spanning the
gap](docs/images/bridge_test.png)
![Round‑hole test: layers arch over the bore instead of
bridging flat](docs/images/round_hole_test.png)

That's it for now — still very much in development.

---

## Installation

Python 3.10+, Windows / Linux / macOS. Dependencies are declared in
[`pyproject.toml`](pyproject.toml):

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate

pip install -e .            # core: deformation + G-code pipeline (headless)
pip install -e .[viewer]    # + interactive PyVista viewer / preview / section
pip install -e .[all]       # + viewer + FMM depth method (scikit-fmm)
```

The editable install also registers a `vff` command, so you can run
`vff model.stl …` instead of `python -m vff model.stl …`.

The **core** (`numpy scipy trimesh rtree embreex`) installs from wheels even on
brand‑new Python (tested on 3.14). The **viewer** (`pyvista`+`vtk`) is optional
because vtk wheels lag new Python releases — the headless paths (`--export`,
`--gcode-in`, `--deform-mode 3d`) never import it.

---

## Usage

The main workflow is the **full‑3D pipeline** (`--deform-mode 3d`). Three steps —
deform → slice → un‑deform:

```bash
# 1. Export the deformed mesh. --subdivide-error refines coarse flat regions
#    (e.g. a bore ceiling) so they can actually bow with the deformation.
vff part.stl --deform-mode 3d --max-tilt 30 --subdivide-error 0.1 \
    --export deformed.stl --no-viewer

# 2. Slice deformed.stl in PrusaSlicer / Cura.
#    IMPORTANT: relative extrusion (M83), model centered on the bed.

# 3. Un-deform the planar G-code back into the original part's coordinates.
#    Reuse the SAME map flags you exported with (see the warning below).
vff part.stl --deform-mode 3d --max-tilt 30 \
    --gcode-in deformed.gcode --gcode-out result.gcode \
    --gcode-direction inverse --subdiv-mm 0.5

# Preview the result (PrusaSlicer's own preview can't show non-planar layers):
vff.preview result.gcode        #  E = toggle extrusion,  T = toggle travel
```

> ⚠️ **Every flag that shapes the deformation map must be identical on the
> export and the inverse:** `--max-tilt`, `--pitch`, `--volume`,
> `--smooth-sigma`. Otherwise the inverse map won't match the mesh that was
> sliced and the layers land in the wrong place.

### Key flags

| Flag | What it does |
|---|---|
| `--max-tilt DEG` | **Main knob.** Max nozzle tilt from vertical the head can print at — depends on your hotend/fan shape. `20` bulky head · `30` default · `45` compact/pointed nozzle (stronger non‑planarity). |
| `--subdivide-error MM` | (export only) Refine the mesh until the worst per‑face deformation error drops below this, so flat regions built from few large triangles actually bow. **Default `0.1` (on)**; `0` disables. |
| `--remesh` | (export only) Refinement method: `adaptive` (default) = Rivara longest‑edge bisection — conforming, refines only the curved faces (~15× fewer faces than uniform). `uniform` = trimesh 1→4 subdivide (heavier fallback). |
| `--extrusion-comp-mode` | `vertical` (default, 3‑axis): rescale E by the layer‑height squish — correct for a fixed‑width vertical nozzle. `volume`: rescale by `1/det` (material‑conservative, for 4/5‑axis like S4). `--no-extrusion-comp` disables. |
| `--max-z-speed MM/S` | (gcode) **Hard cap** on the Z‑axis velocity component `F·\|dz\|/L`: F is recomputed per segment so a curved move never climbs in Z faster than this. Default `15`; `0` = off. Applies the firmware's Z clamp (Klipper `max_z_velocity`) in the toolpath, so the planner sees honest feedrates. Flat moves untouched. |
| `--z-slowdown FACTOR` | (gcode) **Soft ease** for steep non‑planar moves: ×1 flat → ×FACTOR at a 30°+ climb (e.g. `0.5` halves the steepest). Default `1.0` = off. Gentles the *transition* onto steep sections; use `--max-z-speed` to actually bound Z velocity. Composes with it (lower F wins). |
| `--pitch MM` | Voxel size (default 1.0). Smaller = finer field, more RAM. |
| `--smooth-sigma N` | Displacement smoothing in voxels (default 2.0). Higher = smoother mesh / fewer folds, softer domes; lower = sharper. |
| `--cool-overhangs` | (gcode `inverse`) Re‑detect overhangs/bridges on the *original‑space* toolpath and **ramp** the fan there by severity — surfaces the slicer saw as flat (well‑supported) but which become unsupported after the inverse. Fan lerps from `--cool-fan-min` (light overhang, default 128) to `--cool-fan-max` (full bridge, default 255), like a slicer's per‑overlap fan curve; only ever raises the slicer's fan. The feedrate is eased in step: `--cool-speed` (default 20 mm/s, 0 = off) lerps F from the slicer's speed down to that at a full bridge, giving the road time to set over the void (only ever lowers F; composes with `--z-slowdown`/`--max-z-speed`). `--no-cool-overhangs` disables; tune with `--cool-probe` (column depth). |

The interactive viewer (`vff part.stl`, no flags) shows voxels / growth field /
deformed mesh by hotkey, but currently visualizes the **Z‑only** legacy map.

---

## How it works & why

A normal slicer cuts the model with horizontal planes. On a curved top surface
that leaves a coarse staircase. Non‑planar slicing instead lays down **layers
that follow the surface**. On a 5‑axis machine the nozzle tilts; on a 3‑axis
machine the nozzle stays vertical but Z varies along each move, and as long as
the surface angle stays under a limit (`--max-tilt`) the bead bonds fine.

VFF gets there by **deform → slice → un‑deform**:

```
 original ──deform──▶ deformed mesh ──planar slice──▶ planar G-code
   part      (layers              (a normal slicer)        │
            flattened)                                     │ inverse
                                                           ▼
                                          G-code in the ORIGINAL part's
                                          coordinates — flat layers are now
                                          the curved "growth" layers.
```

1. **Growth field.** Fill the model with voxels and "grow" upward from the bed
   like sediment. Every point gets a local *build direction* (which way is up
   for printing here); its level surfaces are the natural layers.
2. **Deform (the 3D map).** Rotate every little chunk so its build direction
   points straight up, and stitch them back together (an ARAP / Poisson solve).
   All three axes move, so in‑plane distances are preserved instead of sheared.
   The result: the natural layers become flat horizontal planes.
3. **Slice** the deformed mesh with any planar slicer.
4. **Un‑deform** the G‑code (a fast vectorised Newton inverse of the map): the
   flat slicer layers become the original curved layers when printed.

Three practical pieces make the result printable on a 3‑axis machine:

- **Tilt clamp (`--max-tilt`)** keeps the layers within the angle a vertical
  nozzle can actually print.
- **Bed blend** holds the first couple of millimetres at identity so the first
  layers stay flat on the plate (no digging in).
- **Extrusion compensation** rescales E for the layer‑height change — the
  vertical nozzle has a fixed road width, so what matters is how the layer
  *spacing* compresses/stretches, not the full volume (the default `vertical`
  mode; `volume` is the 4/5‑axis choice).

On the test propeller the 3D map preserves in‑plane distance ~3× better than a
naive Z‑only shift (edge‑length CoV 0.044 vs 0.148), keeps 96 % of the volume,
and is invertible to ~0.0001 mm on 97 %+ of points.

The deeper math (BFS growth, the vector→potential integration, the 3‑coordinate
Poisson solve, the G‑code passes) lives in
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## Limitations & backlog

- **Tip non‑convergence** under steep overhangs — the inverse uses a damped
  (Levenberg–Marquardt) Newton with per‑point line search plus a despike pass,
  so most tips now converge; the few that can't (Φ genuinely folds there) are
  flagged and snapped to their best iterate. Mitigate the rest with a smaller
  `--max-tilt`.
- **Self‑intersecting / non‑manifold input** needs repair before voxelization.
- **Real FDM printing** of the output is being validated now; treat results as
  experimental.

A legacy **Z‑only** deformation (`--deform-mode z-only`, the default) and two
G‑code‑only modes (`--conform-to`, `--clip-to`) also exist — see
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## License

Experimental code; license not yet chosen.
