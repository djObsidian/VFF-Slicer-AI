# VFF Slicer — internals & reference

Implementation details, the deformation math, the G‑code pipeline and the full
CLI reference. For the user‑facing overview see [README.md](README.md).

## Project layout

```
vff/
├── __main__.py        CLI, orchestration, headless fast-paths
├── build_volume.py    BuildVolume — bed dimensions, index↔world transforms
├── mesh_io.py         load_and_place — load STL, drop to bed, centre on XY
├── voxelize.py        solid voxelization via trimesh.contains (+ embree)
├── growth.py          BFS growth from the bed + 26-neighbour vector field + tilt clamp
├── deform.py          depth field (vectors / FMM / Dijkstra), Z-only deform_mesh
├── deform3d.py        full-3D map (Poisson/ARAP), Newton inverse, BackTransform3D
├── backtransform.py   G-code forward/inverse, extrusion comp, conform-to, clip-to
├── viewer.py          interactive PyVista viewer (hotkeys) — Z-only visualization
├── preview.py         standalone G-code preview
├── gcode_preview.py   G-code parser → (extrusion polylines, travel polylines)
└── section.py         headless XZ cross-section render of the growth surfaces

tests/                 plain-assert regression + validation scripts (run with the venv python)
```

The package is importable without a GUI stack: `Viewer` is lazy‑imported, and
the `--export` / `--gcode-in` / `--deform-mode 3d` paths never touch
pyvista/vtk.

---

## The deformation field

### 1. Voxelization (`voxelize.py`)

The mesh is placed in the build volume, normalised in Z (`Z_min = 0`, bottom on
the bed) and filled with solid voxels at step `pitch` (default 1 mm) via
`trimesh.contains` (fast inside/outside test with embree). Interior voxels in
the lowest non‑empty Z layer become the **bed seeds**.

```
side view, pitch = 1 mm                voxels after fill:
                                          Z
         ▄▄▄▄▄▄▄                          ▲
        ▐  STL  ▌                       4 │       ░░░░░       (inside = 1)
        ▐       ▌                       3 │     ░░░░░░░░░
         ▀▀▀▀▀▀▀                        2 │    ░░░░░░░░░░░
 ───────────────────  bed z=0           1 │   ░░░░░░░░░░░░░
                                        0 │  ░░░░░░░░░░░░░░░  ← bed seeds
                                          └────────────────► X
```

### 2. BFS growth (`growth.py`)

A 26‑connected BFS runs from the bed seeds through interior voxels. Each voxel
gets a **step** = the BFS wave number it was reached on. For a flat‑bottom model
`step ≈ z/pitch`; for a part where material is reached *around* an obstacle
(under a blade, over a bore) the BFS detours, so the step there is larger than
the Euclidean height — this is the geodesic structure the deformation rides on.

### 3. Growth vector field + tilt clamp (`growth.py`)

For each voxel, look at its 26 neighbours, keep the *older* ones (`step < mine`)
in the nearest distance class (face / edge / vertex), average the offsets, and
negate → the local **growth direction**. It is then **clamped** so its angle to
+Z is at most `--max-tilt` (default 30°): the physical tilt limit of a 3‑axis
nozzle.

```
   before clamp        after clamp (max-tilt = 30°)
   ↗ 45°                ↗ 30°   (horizontal component shortened)
   │                     │
```

### 4. Depth field (`deform.py`)

A smooth scalar "depth from the bed" field, used by the Z‑only map and the
section viz (`--depth-method`):

- **`vectors`** (default): least‑squares scalar potential whose gradient matches
  the **clamped** growth vectors (`integrate_vectors_to_potential`, a discrete
  Poisson solve). Its level sets are perpendicular to the growth direction, so
  the tilt clamp actually shapes the layers.
- **`fmm`**: Eikonal `|∇φ| = 1` from the bed (scikit‑fmm); C¹, ignores the clamp
  (raw geodesic). Falls back to `dijkstra` if scikit‑fmm is absent.
- **`dijkstra`**: discrete weighted shortest path (face/edge/vertex = 1/√2/√3);
  C⁰, ignores the clamp.

Outside the model (but in the volume) the field is extended by the nearest
in‑model value (`distance_transform_edt`); without this the deformation tears
the mesh at its surface. A light Gaussian blur (`--smooth-sigma`) removes BFS
steps.

---

## Z‑only map (legacy, `--deform-mode z-only`)

`deform.py` / `BackTransform` in `backtransform.py`. Moves **only Z**, XY frozen:

```
w     = clip((z - bed_z) / blend_h, 0, 1)              # bed blend → flat first layers
new_z = (1 - w)·z  +  w·(depth(x,y,z)·dz_per_layer + bed_z)
new_x = x ;  new_y = y
```

The inverse is a vectorised 1‑D root‑find along each XY column of the depth
field. Fast (~700k pts/0.5 s, multiprocessing above ~5 M), but freezing XY
shears in‑plane distances on tilted layers — which is what the 3D map fixes.

`--dz-per-layer` scales the stretch; `--dz-auto-fit` picks it so the deformed
Z‑extent matches the STL (forward only — for inverse it must equal the dz the
sliced mesh was exported with).

---

## Full‑3D map (`deform3d.py`, `--deform-mode 3d`) — the main path

The Z‑only map flattens the growth surfaces but freezes XY. The 3D map moves all
three axes: it rotates the local frame so the growth direction becomes vertical,
so the surfaces flatten **without** in‑plane shear. It is the 3‑coordinate
generalisation of `integrate_vectors_to_potential` (same Laplacian).

**Solve** (`solve_deformation_map`):

1. Per model voxel `i`: rotation `R_i` taking the (clamped) growth direction
   `ĝ_i → +ẑ` (identity where growth is already vertical).
2. Per 26‑conn edge `(i,j)`: target offset `t_ij = R̄_ij·(p_j − p_i)`,
   `R̄ = ½(R_i + R_j)`.
3. Least squares `‖Φ_j − Φ_i − t_ij‖²` over all edges → normal equations
   `L·Φᶜ = bᶜ` per coordinate (Laplacian `L` shared). Solved for the
   **displacement** `U = Φ − P` with bed seeds pinned to `U=0` (pinning Φ to the
   absolute ~125 mm bed position blows the CG right‑hand side up to ~1e9 and the
   relative tolerance silently "converges" to garbage — solving for U keeps it
   well‑scaled).
4. **Displacement smoothing** (`--smooth-sigma`, default 2.0): a Gaussian on `U`
   tames the per‑voxel inconsistencies that fold the map at tips. Higher =
   smoother mesh / fewer folds but softer real curvature (a bore dome that
   should be +1.5 mm shrinks to +0.7 mm at sigma 2 — pair with
   `--subdivide-error` to keep it, or drop sigma for a sharper dome).
5. **Bed blend** (2 mm): ramp `U` to zero at the plate so the first layers stay
   flat at their sliced height. Without it the deformation tilts right above the
   pinned seeds and the inverse pushes the sliced first layer to a varying,
   partly negative original Z (nozzle digs in).

**Forward map** = trilinear sample of Φ. **Inverse map** = vectorised
**damped (Levenberg–Marquardt) Newton** on the trilinear Φ field (`q → p` with
`Φ(p)=q`) using precomputed Jacobian fields; converges in a few iterations on a
fold‑free Φ. Each per‑point step is `(JᵀJ + λI)⁻¹ Jᵀr` with a per‑point λ that
backs off (accept, shrink λ) or damps harder (reject, grow λ) by whether the
residual actually dropped — so a near‑singular Jacobian at a fold damps into a
short gradient step instead of exploding. It tracks the best iterate (not the
last clamped one) and a `_despike_path` pass repairs the occasional
wrong‑branch spike at a fold by neighbour interpolation. The rotations are
driven by the local BFS growth vectors (lowest distortion).

### Mesh resolution (`--subdivide-error`, export)

Φ is applied **per vertex**, so a flat region built from few large triangles
(e.g. a bore ceiling, one 11 mm triangle) stays flat instead of bowing — up to
~0.5 mm error on the propeller. `--subdivide-error MM` (default 0.1; 0 = off)
refines the mesh until the worst per‑face non‑affinity drops below the tolerance.

`--remesh` picks the method (both stay watertight — no T‑junction cracks):

- **`adaptive`** (default): **Rivara longest‑edge bisection** (`_adaptive_refine`).
  Each pass marks the longest edge of every over‑error face, takes the
  *longest‑edge closure* (if any edge of a face is marked, mark its longest too —
  iterated to a fixed point), then splits faces by their marked‑edge pattern.
  Edge midpoints are shared between neighbours, so it's conforming. It refines
  ONLY the curved faces: propeller 35k → **36k** faces at 0.1 mm (0.52 → 0.10 mm),
  vs uniform's 562k — **~16× lighter**, and faster.
- **`uniform`**: trimesh's 1→4 subdivide of every face each pass. Simple but
  blows the count up (562k); kept as a fallback.

### Extrusion compensation (`backtransform.py`)

The deformation stretches/compresses each road, so the slicer's E (computed for
the deformed geometry) is rescaled per piece (relative E only; retractions and
absolute E untouched). Two modes (`--extrusion-comp-mode`):

- **`vertical`** (default, 3‑axis): scale by the layer‑height squish
  `∂orig_z/∂def_z = (JΦ⁻¹)[z,z]`. The vertical nozzle has a fixed road **width**,
  so over/under‑extrusion is driven by how the layer *spacing* changes, not the
  full volume. Reduces E where layers compress (propeller mean ×0.93).
- **`volume`** (4/5‑axis, S4‑style): scale by `1/det(JΦ)` — material‑
  conservative, correct when the nozzle tilts and the road deforms in all axes
  (propeller mean ×1.05). The two differ ~12 % and **opposite sign** in the bulk
  on a 3‑axis machine; they agree at the stretched tips.

### Consistency

Every map‑shaping flag (`--max-tilt`, `--pitch`, `--volume`, `--smooth-sigma`)
must be identical on the export and the inverse, or the inverse won't match the
sliced mesh.

---

## G‑code pipeline (`backtransform.py`)

```
G-code ─▶ Pass 1: parse  ─▶ (units list, xyz_def[N,3])
                                   │
                                   ▼ Pass 2: vectorised transform (forward / inverse)
                                   │
                                   ▼ Pass 3: stream-write new G-code (format preserved)
```

- **Pass 1**: each G0/G1 move is split into pieces `≤ subdiv_mm` (default 0.5 mm)
  so a straight line in deformed space follows the curve in original space.
  M82/M83 (absolute/relative E) and G92 are tracked; non‑moves pass through
  verbatim. The running E is tracked so an **absolute‑E** move's filament is
  ramped across the pieces (not dumped in the last one).
- **Pass 2**: vectorised NumPy single pass (multiprocessing above ~5 M pts for
  the Z‑only depth‑field map; the 3D Newton inverse runs single‑process).
- **Pass 3**: writes `n_pieces` lines per move, distributing E (× the extrusion
  comp factor for the 3D path).

**XY alignment** (`quick_gcode_xy_bounds`): slicers centre the model on their own
bed (e.g. PrusaSlicer 220×220 → 110,110), which won't match our build‑volume
centre. A quick pre‑scan takes the bbox of **extrusion** moves only (travels to
corners ignored) and the depth field / 3D map is translated to that centre.
`--no-gcode-align` disables it.

### Alternative G‑code modes (experimental, depth‑field‑free)

- `--conform-to STL`: `new_z = z + H_bottom(x,y)` — lift each point by the
  bottom‑surface height of a target STL (ray‑cast from below). For "print this
  curved‑bottom mesh from G‑code sliced for a flat‑bottom variant".
- `--clip-to STL`: drop extrusion (G1 → G0) for points outside the target STL's
  interior. Visual verification only — leaves unsupported geometry.

---

## CLI reference

```
positional: STL                 path to STL (default ./propeller_fixed_flat.stl)

  --volume X x Y x Z            build volume in mm (default 250x250x250)
  --pitch FLOAT                 voxel pitch in mm (default 1.0)
  --max-tilt DEG                max nozzle tilt from vertical (default 30)
  --smooth-sigma FLOAT          Gaussian sigma in voxels (default 2.0);
                                3d smooths displacement, z-only the depth field
  --depth-method vectors|fmm|dijkstra   inside-model depth (default vectors; z-only)

  --deform-mode z-only|3d       deformation model (default z-only; 3d = recommended)
  --subdivide-error MM          (3d --export) refine coarse faces until error < MM (default 0.1; 0 = off)
  --remesh adaptive|uniform     (3d --export) refine method (default adaptive = Rivara longest-edge bisection)
  --extrusion-comp / --no-extrusion-comp   (3d) rescale E for the deformation (default on)
  --extrusion-comp-mode vertical|volume    (3d) comp model (default vertical = 3-axis)
  --cool-overhangs / --no-cool-overhangs   (3d inverse) ramp fan on re-detected overhangs/bridges (default on)
  --cool-fan-min 0..255         (3d inverse) fan PWM at the lightest overhang (default 128)
  --cool-fan-max 0..255         (3d inverse) fan PWM at a full bridge (default 255); lerp by severity
  --cool-speed MM/S             (3d inverse) print speed at a full bridge (default 20; 0 = off); lerp from slicer F
  --cool-probe MM               (3d inverse) downward support-probe column depth (default 0.8)
  --preview-bead / --no-preview-bead   (3d inverse) rewrite ;HEIGHT:/;WIDTH: so a loaded-gcode
                               viewer draws the real deformed road (H0*gap, W0*Emult/gap) — cosmetic,
                               comments only, default on

  --export PATH                 save the deformed mesh
  --no-viewer                   skip the interactive viewer (batch)
  --section-xz PATH             headless XZ cross-section of the growth surfaces
  --preview-gcode PATH          overlay a G-code file in the viewer

  --gcode-in PATH               input G-code to transform
  --gcode-out PATH              output (default <in>.{nonplanar,planar}.gcode)
  --gcode-direction forward|inverse   (default forward; inverse = un-deform a sliced deformed mesh)
  --subdiv-mm FLOAT             split G1 moves longer than this (default 0.5)
  --max-z-speed MM/S            hard-cap the Z-velocity component F*|dz|/L (default 15; 0 = off;
                               firmware Z clamp applied in the toolpath; flat moves untouched)
  --z-slowdown FACTOR          soft-ease F on steep non-planar moves (default 1.0 = off;
                               x1 flat -> xFACTOR at a 30deg+ climb; composes with --max-z-speed)
  --dz-per-layer FLOAT          (z-only) forward-map Z scale
  --dz-auto-fit                 (z-only, forward) pick dz so Z-extent matches the STL
  --no-gcode-align              don't align the map to the gcode XY bounds
  --jobs N                      (z-only inverse) parallel workers (-1 = auto)

  --conform-to STL              surface-offset G-code mode
  --clip-to STL                 inside-mesh clip G-code mode
```

Standalone viewer: `python -m vff.preview FILE.gcode [--volume ...]`
(`E` extrusion, `T` travel, `R` reset cam).

---

## Tests

Plain‑assert scripts under `tests/`, run with the project venv python:

- `test_gcode_transform.py` — E distribution (relative + absolute subdivision),
  forward/inverse round‑trip.
- `test_deform3d.py` — box→identity, distortion vs Z‑only, inverse round‑trip,
  extrusion comp (volume + vertical), bed blend, subdivision watertightness,
  bore dome.
- `validate_deform3d.py`, `validate_inverse_gcode.py` — diagnostic reports
  (distortion, containment of the inverse inside the original mesh).
