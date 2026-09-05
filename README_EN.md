# UW2-BadlandsDriver — Call Badlands natively from Underworld2

**English** | [中文](README.md)

> A single file, [`badlands_driver.py`](badlands_driver.py), that couples Badlands
> into an Underworld2 (UW2) model as an **in-process Python library** — bypassing
> the UWGeodynamics `surfaceProcesses.Badlands` wrapper. No third-party
> dependencies beyond numpy/scipy (both ship with underworld2).
>
> Provenance: a line-for-line semantic port of UWG `surfaceProcesses.Badlands`
> (underworld2 2.17.3 source, `surfaceProcesses.py` L48-537), with only two
> deliberate deviations (§1.2). Equivalence was verified on a full 2D model with a
> two-path A/B numerical comparison (all datasets of 84 frames bit-identical).

---

## 1. What it is, and the call chain

### 1.1 In one sentence

`BadlandsDriver` is a duck-typed surface-process driver class: mount it on
`GEO.Model.surfaceProcesses`, and the UW2 main loop automatically calls its
`solve(dt)` every timestep — sampling the UW surface velocity, injecting it into
Badlands as tectonic displacement, advancing Badlands, then reading the new
surface back and converting UW particle materials (air↔sediment). Badlands runs
entirely as an **in-process Python library** (no subprocess, no file exchange).
Duck typing means the caller only checks *what an object can do* (which
methods/attributes it has), not *which class it inherits from* — hence the driver
needs no UWG base class at all.

```
GEO.Model.run_for(...)
 └── Model._update() (every step, after advection/pop-control)
      └── surfaceProcesses.solve(dt)          ← mount point: duck-typed, only requires solve(dt)
           └── BadlandsDriver.solve(dt)
                ├─ help_gen_sample_point()    dynamic-surface sample points (bcast)
                ├─ velocityField.evaluate_global(nd_coords)   collective sampling on all ranks
                ├─ _inject_badlands_displacement()  writes force.T_disp time window + force.injected_disps
                ├─ bdm.force.next_display = t+dt    aligns Badlands checkpoints with UW steps
                ├─ bdm.run_to_time(t+dt)            ← badlands.Model lives on rank 0 only
                └─ _update_material_types()   TIN elevations bcast → per-rank local mask conversion
```

Why the UWG mount point accepts a foreign class: the `surfaceProcesses` setter in
`_model.py` only assigns `obj.timeField = ...; obj.Model = ...` (which triggers
initialisation), and the main loop only calls `solve(dt)` — inheritance is never
checked.

### 1.2 The two deliberate deviations from the UWG wrapper (everything else is line-equivalent)

| Deviation | What | Why |
| --- | --- | --- |
| No ABC | Does not inherit the `SurfaceProcesses` abstract base class (a py2-era relic); a plain class with a `Model` property setter that triggers `_init_model` | Mount contract unchanged, one less upstream dependency |
| Factory method | Badlands instantiation is funnelled into `_create_badlands_model()`, the single place in the module that does `from badlands.model import Model` | A test seam (in the sense of Feathers: a point where behaviour can be swapped without editing code) for injecting stubs in unit tests |

---

## 2. Prerequisites

### 2.1 Software versions

| Component | Requirement | Notes |
| --- | --- | --- |
| underworld2 | 2.17.x (includes UWGeodynamics, same package) | The driver only uses the underworld core (mpi/scaling/function); mounting requires UWG's `GEO.Model` |
| badlands | 2.3.x, must contain the UW coupling hooks | The hooks (XML `<udw>`, `force.injected_disps`, `getSea(udw)`) ship in official badlands 2.3.1; see the fingerprint check below |
| numpy / scipy | via the above | `scipy.interpolate.griddata/interp1d/CloughTocher2DInterpolator`, `scipy.ndimage.filters.gaussian_filter` |

### 2.2 Coupling-hook fingerprint self-check (30 seconds)

Run this against your installation; all three lines True means direct coupling is possible:

```python
import inspect
from badlands.forcing import forceSim, xmlParser
from badlands.model import Model
print("injected_disps:", "injected_disps" in inspect.getsource(forceSim))      # in-memory displacement injection
print("udw xml tag  :", "udw" in inspect.getsource(xmlParser))                 # coupling gate
print("udw tEnd gate:", "udw" in inspect.getsource(Model.run_to_time))         # run_to_time re-callable
```

### 2.3 badlands robustness patches (optional, but recommended)

The driver itself does not depend on them; they fix correctness issues in **long
coupled runs**. Patched source files (model.py / simulation/buildFlux.py /
underland/strataMesh.py) are available at
[Badlands_coupling_fixedStrata](https://github.com/HonghaoXiong/Badlands_coupling_fixedStrata):

| Patch | Consequence if missing | Needed when |
| --- | --- | --- |
| Optional strata (`strata_enabled`) | xmlParser misbehaves without a `<strata>` node | Users who run without stratigraphy |
| fluxTarget fix | `run_to_time` sediments to the wrong target time | All coupled runs |
| tNow alignment + minDT clamp | tNow drift / CFL overshoot when `run_to_time` is called repeatedly | All coupled runs (incremental advancement depends on it) |

### 2.4 MPI thread discipline (environment variables before running)

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

Mixing multi-threaded OpenBLAS with MPI makes things *slower* with more cores and
corrupts timings.

---

## 3. Where to put the module (location → import statement)

**One sentence: copy `badlands_driver.py` into the directory of your run script
and simply `from badlands_driver import BadlandsDriver` — zero configuration.**
For a set-and-forget setup, drop it into site-packages; or clone this repository
and put it on PYTHONPATH.

| Option | Location | Import statement | Suits |
| --- | --- | --- | --- |
| ① Same dir as script (**recommended start**) | `badlands_driver.py` next to your main script (Python puts the script dir on the search path automatically) | `from badlands_driver import BadlandsDriver` | Zero-config, single model |
| ② site-packages (set-and-forget) | `cp badlands_driver.py $(python -c "import site; print(site.getsitepackages()[0])")` | `from badlands_driver import BadlandsDriver` (importable from any directory) | Long-term use across models/notebooks |
| ③ Standalone lib dir | e.g. `~/libs/badlands_driver.py` with `export PYTHONPATH=~/libs` | `from badlands_driver import BadlandsDriver` | One shared copy across environments |
| ④ Clone this repository | `git clone https://github.com/HonghaoXiong/UW2-BadlandsDriver.git`, `export PYTHONPATH=<clone dir>` | `from badlands_driver import BadlandsDriver` | Tracks repository updates (`git pull`) |

30-second verification after placing (run from any unrelated directory):

```bash
python -c "from badlands_driver import BadlandsDriver; print('OK')"
```

Note: options ①②③ are **copies** — re-copy after this repository is updated, or
you will run a stale driver; option ④ keeps a single source of truth.

---

## 4. Minimal integration (five steps)

### Step 1: Prepare the material system

The UW side needs two material indices, an **air phase** and a **sediment phase**
(the driver converts particles between them by mask):

```python
from underworld import UWGeodynamics as GEO
from underworld.scaling import units as u

air = GEO.Material(name="Air", density=1.)          # sticky air, density ≈ 0
sediment = GEO.Material(name="Sediment", density=2400.)
# airIndex may be a list (all listed air-like materials get converted);
# sedimentIndex is a single index
```

Trap (a general UWG pitfall, unrelated to the driver but unavoidable): when using
a custom initial geotherm, `Model.init_model(temperature=None)` must pass `None`
explicitly — the default `"steady-state"` **silently overwrites** your initial
temperature field.

### Step 2: Initial topography

Two sources:

- **Constant / function**: `BadlandsDriver(..., surfElevation=<UW function or constant>)`
  — the driver evaluates it on a regular grid to build the DEM (written in metres
  to `tempfile.gettempdir()/dem.csv` before being handed to Badlands);
- **Analytic topography** (recommended): subclass and override `_generate_dem()`
  to return an `(nx*ny, 3)` array (in metres); its z column is your initial
  elevation function.

### Step 3: Prepare the Badlands XML

Minimal template:

```xml
<badlands>
  <grid>
    <resfactor>1</resfactor>
    <boundary>wall</boundary>   <!-- 2D coupling convention: closed side walls -->
    <udw>1</udw>                <!-- ★ UW coupling gate: tEnd yields to the value passed to
                                     run_to_time; run_to_time becomes re-callable; final strata
                                     writes are forced -->
    <nopit>0</nopit>
  </grid>
  <time>
    <start>0.</start>
    <end>200000.</end>          <!-- ★ ≥ 2× the actual coupled duration (strata arrays are
                                     pre-allocated against tEnd) -->
    <display>250.</display>     <!-- overwritten by the driver's checkpoint_interval; keep consistent -->
  </time>
  <precipitation>...</precipitation>   <!-- as needed -->
  <sp_law>...</sp_law>                 <!-- as needed: erodibility/m/n/fillmax etc. -->
  <outfolder>outbdls</outfolder>       <!-- in practice overwritten by the driver's outputDir -->
</badlands>
```

Do **not** set `<demfile>`: the DEM is generated from the UW surface by the
driver and injected via `_build_mesh` (Badlands explicitly supports deferred mesh
construction when the XML has no demfile, model.py:133-136). For `<strata>`
requirements, see §5.6.

### Step 4: Mount (before solving starts)

```python
from badlands_driver import BadlandsDriver

Model.surfaceProcesses = BadlandsDriver(
    airIndex=[air.index],                # list: every material index treated as "air"
    sedimentIndex=sediment.index,
    XML="badlands.xml",                  # your parameter file
    resolution=1000.0 * u.meter,         # Badlands DEM resolution (with units)
    checkpoint_interval=250.0 * u.year,  # overwrites tDisplay: Badlands frames align with UW steps
    outputDir="/abs/path/to/outbdls",    # absolute path recommended (see pitfall 2, §7)
    aspectRatio2d=0.25,                  # 2D only: aspect ratio of the fabricated 2nd horizontal dim
)
```

Initialisation happens at the moment of assignment (the setter triggers
`_init_model`: load_xml → generate DEM → build mesh → attribute overrides →
initial material conversion).

### Step 5: Run

```python
Model.run_for(duration=10.0 * u.kiloyear, dt=250.0 * u.year, checkpoint=...)
```

The main loop calls `solve(dt)` automatically every step. **Do not loop over
solve yourself** — unless you are writing a special path such as an ALE-IB free
surface (see §5.5).

---

## 5. Design / model-setup requirements (prerequisites for correct coupling)

| # | Setting | Requirement & reason |
| --- | --- | --- |
| 1 | **Resolution relationship** | The Badlands DEM is usually **finer** than the UW mesh (e.g. 1 km vs 5-10 km). Material classification uses `griddata(nearest)` — precision degrades if Badlands is coarser than UW |
| 2 | **2D third dimension** | Badlands is internally a 3D TIN; 2D coupling fabricates the second horizontal dimension with `aspectRatio2d`, injected displacements are tiled along `recGrid.rny`, and y-displacement is always 0. Pick a value that keeps the fabricated dimension narrow (0.25 is typical) |
| 3 | **minCoord/maxCoord** | Defaults to the UW mesh extent (non-dimensionalised); in 2D both dims derive from the x-range × `aspectRatio2d`. Can be overridden explicitly (with units) |
| 4 | **Timing cadence** | `checkpoint_interval` = UW step → one Badlands frame per step, aligned; `<laytime> ≤ tDisplay` (integer divisor); the udw semantics are **one forced frame + one forced layer per coupled step** (this is the stratigraphic recording contract) |
| 5 | **Free surface / ALE-IB** | On the `FreeSurfaceProcessor` path, `solve(dt)` has a **return-value contract**: when `Model._freeSurface_ALEIB` is true it returns the Badlands surface interpolant (2D = `interp1d`, 3D = `CloughTocher2DInterpolator`), otherwise None. Ordinary coupling ignores the return value |
| 6 | **With strata enabled** | `strataMesh` captures the **CWD-relative outDir of xmlParser** during `_build_mesh`, earlier than the driver's `input.outDir` override — without the fix, sed files go to a stray directory. After mounting, add (rank-safe by construction; the `badlands_model` attribute simply does not exist on non-zero ranks): `bdm = getattr(Model.surfaceProcesses, "badlands_model", None)` → `if bdm is not None and getattr(bdm, "strata", None) is not None: bdm.strata.folder = <your outputDir>` |
| 7 | **Restart** | Uses **constructor-time** restart: `BadlandsDriver(..., restartFolder=<run>/outbdls, restartStep=<frame number>)`; the driver parses `xmf/tin.time{N}.xmf` to restore `time_years` and enables Badlands' internal restart. Note: UWG's `Model.restart()` Badlands-rebuild path dispatches on `isinstance(obj, surfaceProcesses.Badlands)` and **does not trigger** for this driver — call `Model.restart()` first, then mount the driver |
| 8 | **Disk planning** | With strata on, full sed frames grow linearly with layer count (O(n_layers) per frame; multi-Myr runs can reach TB scale) — estimate layers × frame size up front |

---

## 6. Unit conventions (three coexisting systems; bugs have happened)

| Side | Convention | Remember |
| --- | --- | --- |
| UW2 fields | Non-dimensional (nd) | Temperature ×KT(=700) → K; particle coordinates in nd numbers = km (when the `[length]` scale = 1000 m) |
| Badlands | **metres / years** | DEM, displacement, elevation, tNow all in metres and years |
| Interface conversion points (three, inside the driver) | — | ① `_generate_dem` ends with `dimensionise(..., u.meter)` writing metres; ② `solve` converts velocity×dt with `dimensionise(..., u.meter)` before injection; ③ TIN elevations read back are `/fact` to nd. When writing your own interpolation/comparison across systems, write the conversion comment before the code |

---

## 7. Caveats and pitfall checklist

| # | Pitfall | Symptom / defence |
| --- | --- | --- |
| 1 | **MPI collective discipline** | `solve`/`_init_model` are collective call points on all ranks: every `comm.bcast`/`Barrier`/`evaluate_global` must be reached identically by all ranks. Never wrap `solve` in `if rank==0` (deadlock); the driver already confines Badlands calls to rank 0 |
| 2 | **Relative output directory** | Prefer an absolute `outputDir`; a relative path is resolved against the **CWD at XML-parse time** (xmlParser creates directories in its constructor). Strata has the separate pitfall 6 (§5.6) |
| 3 | **XML `<end>` pre-allocation** | `<end>` ≥ 2× the coupled duration, otherwise strata arrays overflow/truncate |
| 4 | **Temp-DEM race** | The DEM is written to a fixed `tempfile.gettempdir()/dem.csv` — concurrent coupled runs on one machine overwrite each other. For parallel runs, subclass and rename `_demfile` |
| 5 | **pint float noise** | `dimensionise` year conversion carries ~1e-14 noise (250 → 249.999...97); the driver already removes it with `np.round(...,6)` on `dt_years`; do the same in your own time comparisons |
| 6 | **airIndex is list-valued** | Material masks use `np.in1d`: particles of *any* index in airIndex below the surface become `sedimentIndex` (single value); erosion converts back to `airIndex[0]` |
| 7 | **Velocity sampling style** | `solve(dt, uw_sample_style=0)` (default) samples on Badlands' current dynamic surface; `1` = legacy method (static initial recGrid elevations). `sigma>0` Gaussian-smooths the injected displacement (in step units) |
| 8 | **Badlands serial bottleneck** | Badlands runs on rank 0 only: returns from more UW cores diminish (Amdahl's law). Attribute the Badlands share first before adding cores |
| 9 | **Verbose output** | Prints a purple "Processing surface with Badlands..." every step by default; pass `verbose=False` for quieter production logs |
| 10 | **Swarm particle identity** | Track particles across checkpoints with a `globalIndex` variable (`rank*10**10+arange`); row numbers break under pop-control/MPI |

---

## 8. Extension points and testing

### 8.1 Subclass override points

| Method | Purpose |
| --- | --- |
| `_generate_dem()` | Analytic initial topography (returns a metre-based array) |
| `_update_material_types()` | Multi-facies sediment classification (native supports a single sedimentIndex only; dispatch multiple sediment materials by water depth / sedimentation rate / basin position) |
| `_init_model()` | Append attribute overrides after `super()` (strata folder, random seed, ...) |
| `_create_badlands_model()` | Inject a stub for tests (next item) |

### 8.2 Stub-based unit testing

Write a fake Badlands that records calls without evolving state (with
`input`/`force`/`recGrid` namespaces and the three methods `load_xml` /
`_build_mesh` / `run_to_time`) plus a fake UW Model (the four-piece
mesh/velocityField/swarm/materialField), override `_create_badlands_model` in a
test subclass to return the stub — you can then test DEM generation, injection
bookkeeping, mask directions and override sequences offline, without real
Badlands. Caveat: if a test depends on the default scaling (1 nd = 1 m), it must
first reset `GEO.scaling_coefficients` (the global dictionary in
`underworld.scaling` can be polluted by other modules in the same process) and
restore it afterwards.

### 8.3 Equivalence / regression verification methodology

After changing the driver or the Badlands version: run both paths with the same
random seed, then diff frame by frame — `outbdls/h5/tin.time*.hdf5`
(elevation/cumdiff/lake) + `sed.time*.hdf5` + the final material field; the
passing criterion is max&#124;diff&#124; = 0.

---

## 9. Relationship to other approaches

- **Native UWG wrapper** (`underworld.UWGeodynamics.surfaceProcesses.Badlands`):
  semantically equivalent to this driver (except the two deviations in §1.2) and
  still a valid reference implementation.
- **Out-of-process coupling** (subprocess + file exchange): not what this driver
  does; consider it only when you need Badlands version decoupling (you would
  lose the two core hooks: `injected_disps` in-memory injection and incremental
  `run_to_time`).
- **Other surface-process backends**: the driver funnels the Badlands API surface
  into one file (1 class + 3 methods + ~20 attributes); swapping backends means
  rewriting only the attribute accesses in this file — one of the design
  motivations for direct coupling.

---

## License

- This repository is released under **LGPL-3.0** (see [LICENSE](LICENSE)):
  `badlands_driver.py` is a semantic port of
  [underworld2/UWGeodynamics](https://github.com/underworldcode/underworld2)
  `surfaceProcesses.py` and is a derivative work thereof.
- [badlands](https://github.com/badlands-model) itself is an independent upstream
  dependency under its own license; patched robustness source files are available
  at
  [Badlands_coupling_fixedStrata](https://github.com/HonghaoXiong/Badlands_coupling_fixedStrata).

---

## Appendix: 10-point pre-flight checklist

1. Module placed per §3, and both `python -c "from badlands_driver import BadlandsDriver"` and `python -c "import underworld, badlands"` succeed?
2. All three fingerprint assertions in §2.2 True?
3. `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` exported?
4. XML has `<udw>1</udw>`, no `<demfile>`, and `<end>` ≥ 2× coupled duration?
5. `<laytime>` (if strata on) ≤ and an integer divisor of `checkpoint_interval`?
6. Material system has air (may be several) and sediment (single) indices? Are air particles actually present in the swarm?
7. `init_model(temperature=None)` (if using a custom geotherm)?
8. `outputDir` absolute, with `outbdls/h5` and `outbdls/xmf` subdirectories created?
9. With strata: `badlands_model.strata.folder` synced after mounting (§5.6)?
10. Multiple runs on one machine: temp-dir dem.csv race handled (pitfall 4, §7)?
