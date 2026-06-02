# CLAUDE.md

Guidance for Claude Code working in this repo. See [README.md](README.md) for the
user overview and [ARCHITECTURE.md](ARCHITECTURE.md) for the deep algorithm + full
CLI reference.

## What this is

VFF Slicer — a non-planar slicing pre/post-processor for **3-axis FDM**. It
voxelizes an STL, grows a direction field from the bed, **deforms the mesh** so
its natural layers flatten (exported → sliced by a normal planar slicer), then
**transforms the planar G-code back** into curved layers in the original part's
coordinates. Pure-Python (numpy/scipy/trimesh); no custom native code.

## Environment & commands

A project venv lives in `.venv` (Python 3.14, core deps only — **no pyvista**).
Use it directly; do NOT assume the bare `python` has the deps.

```bash
./.venv/Scripts/python.exe -m vff <args>          # run the CLI (Windows path)
./.venv/Scripts/python.exe tests/test_deform3d.py        # 3D map tests
./.venv/Scripts/python.exe tests/test_gcode_transform.py # G-code tests
```

Tests are **plain-assert scripts** (no pytest); exit 0 = pass, they print
PASS/FAIL per case. Run them after changing `deform3d.py`, `deform.py`,
`backtransform.py`, or `growth.py`.

Reinstall / new deps: `./.venv/Scripts/python.exe -m pip install -e .` (deps in
`pyproject.toml`; viewer/`pyvista` is the optional `[viewer]` extra — vtk wheels
lag new Python, keep it out of the headless paths). The `[solver]` extra
(`pyamg`) accelerates the full-3D deform solve ~2-4× at fine pitch (0.2-0.12);
optional with a graceful fall back to plain CG (`solve_deformation_map`), so a
missing wheel never breaks the core install. Install with `pip install -e
.[solver]` (or `pip install pyamg`).

## Architecture (where things live)

- `deform3d.py` — **the main path**: full-3D Poisson/ARAP map, Newton inverse,
  `BackTransform3D`, extrusion-comp helpers (`jacobian_det`, `layer_gap_ratio`).
- `deform.py` — legacy Z-only `deform_mesh` + the depth field (`vectors`/fmm/dijkstra).
- `backtransform.py` — G-code 3-pass parse/transform/write, extrusion comp,
  `conform-to`/`clip-to`.
- `growth.py` — BFS growth + clamped vector field. `voxelize.py`, `mesh_io.py`,
  `build_volume.py` — geometry setup. `viewer.py`/`preview.py`/`section.py` — pyvista.
- `__main__.py` — CLI + headless fast-paths (gcode / 3d export skip the viewer).

The pipeline: voxelize → BFS growth → vector field (+ tilt clamp) → 3D map
(rotate growth→vertical, solve `L·U=b` for displacement, smooth, bed-blend) →
forward (trilinear) / inverse (Newton) → G-code rewrite with extrusion comp.

## Project-specific gotchas

- **Consistency rule:** every map-shaping flag (`--max-tilt`, `--pitch`,
  `--volume`, `--smooth-sigma`, `--depth-method`) must be IDENTICAL on the
  `--export` and the `--gcode-direction inverse` run, or the inverse won't match
  the sliced mesh. When changing a default, update both paths.
- **`--depth-method` in 3d** picks the build direction the map rotates to
  vertical: `vectors` (default, clamped BFS dirs) or `harmonic` (∇φ of the
  Laplace layer potential — curl-free, NO tilt clamp, no closed layer surfaces;
  see [GROWTH_SURFACES_MATH.md](GROWTH_SURFACES_MATH.md)). On the mushroom both
  are fold-free/invertible; harmonic's win is conceptual (true gradient) + the
  scalar field has no closed shells. Harmonic has no clamp, so overhangs can
  exceed `--max-tilt`.
- **3-axis, not 4-axis:** the nozzle is vertical (that's the whole `--max-tilt`
  premise). Extrusion comp defaults to `vertical` (layer-height squish), NOT the
  volume/`1/det` that S4 uses for its tilting 4-axis nozzle.
- **Windows cp1251 console:** `print()` of non-ASCII (`Φ → ≈ √ ³`) raises
  `UnicodeEncodeError`. `__main__` reconfigures stdout to UTF-8, but any
  standalone script/`-c` snippet that prints those chars must do
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` or use ASCII.
- **Don't break the headless paths:** the package must import without pyvista
  (`Viewer` is lazy in `__init__.py`; the CLI gcode/3d-export branches return
  before importing the viewer). Keep new heavy/GUI imports lazy.
- **Relative E (M83) only** for extrusion comp; absolute E is left uncompensated.
- Deformed-mesh artifacts (`deformed_3d.stl`, `*.gcode`) and `.venv*` are
  gitignored; `S4_Slicer/` is a reference clone, also gitignored.

## Workflow conventions

- Work on `master` and **commit as you go** (the maintainer's explicit
  preference). End commit messages with the `Co-Authored-By` trailer.
- Match the surrounding style: thorough docstrings explaining the *why*, and
  inline comments on the non-obvious math. Validate changes by running the
  relevant test script (and, for deformation changes, the `validate_*.py`
  diagnostics).
