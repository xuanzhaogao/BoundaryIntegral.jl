# Unify the four-index campaign into BoundaryIntegral.jl with a `.toml` input

**Date:** 2026-06-10
**Branch:** `multi_rhs` (BoundaryIntegral.jl) + thin deployment layer in
`~/work/four_index_integral_solver/codes/lattice_scale/`
**Supersedes input format of:** `2026-06-03-per-center-multi-rhs-design.md` (`.bie`/SystemInput)
**Builds on:** `2026-06-10-multinode-lattice-campaign-design.md` (the campaign pipeline,
now folded into the package)

## 1. Goal

Fold the standalone `CampaignLib` (currently in the work repo) into
`BoundaryIntegral.jl`, replace the `.bie` text input with a single `.toml` format, make
orbital placement an explicit per-orbital list (arbitrary positions, not an `nx×ny`
sweep), emit the four-index result as a text table, and move the deprecated
`ClusterManagers.SlurmManager` to `SlurmClusterManager.jl` behind a package extension.

Four changes, one coherent spec because they all touch the same input/output/packaging
surface:
1. **V output** as a text table `i j k l V` (replacing `V_full.jls`).
2. **Explicit orbital list** `(type, x, y, z)` (replacing `[lattice] nx/ny`).
3. **SlurmClusterManager.jl** (replacing deprecated `ClusterManagers.SlurmManager`).
4. **Merge `CampaignLib` → `BoundaryIntegral.jl`**; `.toml` fully replaces `.bie`.

Decisions fixed during brainstorming (2026-06-10), in order asked:
- Orbital placement: **snap `(x,y,z)` to the nearest integer grid offset** (keeps the
  exact commensurate-grid pair-product machinery; no resampling).
- Input: explicit `[[orbital]]` list, `type` indexes a `templates` list; **pairs by
  distance cutoff** (with optional explicit overrides).
- Merge boundary: **core library logic into BI.jl; Distributed + SlurmClusterManager via
  a package extension** (`ext/`). The numerics package stays free of cluster deps.
- `.bie` fate: **full replacement** — delete the `.bie` parser; `.toml` is the only input.
- V output: **text only, all entries** (dense; ~6.8M rows at 10×10 is acceptable).

## 2. Package architecture

```
BoundaryIntegral.jl/
  src/
    campaign/                         # NEW core (no Distributed/Slurm deps)
      toml_input.jl   # parse .toml → CampaignInput
      geometry.jl     # explicit orbitals → snapped OrbitalInstances; cutoff → pairs
      manifest.jl     # CenterInfo/BatchSpec, enumerate, build_batches, TSV  (from CampaignLib)
      tasks.jl        # prepare/solve_batch/consolidate/eval_batch/assemble_v (SERIAL)
      v_output.jl     # write_v_table → V_full.tsv
    solver/lattice_batch.jl, batch_io.jl    # already present (Tasks 1–13)
  ext/
    BoundaryIntegralDistributedExt.jl # NEW: run_phase parallel (addprocs(SlurmManager()) + pmap)
  Project.toml        # [weakdeps] Distributed, SlurmClusterManager; [extensions]
```

- **Core** holds all pure logic plus the **serial** phase functions, so
  `using BoundaryIntegral; c = load_campaign("x.toml"); prepare(c); solve_batch(c, 1)`
  works with zero parallel dependencies (this is the inline/pilot path).
- **Extension** `BoundaryIntegralDistributedExt` is triggered by loading `Distributed`
  **and** `SlurmClusterManager`; it adds `run_phase(c, phase)` = spawn workers
  (`addprocs(SlurmManager())` on Slurm, or `addprocs(n)` locally) + `pmap` over the core
  phase functions. No cluster-manager weight for solver-only users.
- **Work repo** (`codes/lattice_scale/`) keeps only deployment: `sbatch` templates, the
  specific campaign `.toml`s, and a ~10-line `driver.jl` that loads the extension and
  calls `run_phase`. `CampaignLib` (its `src/`, Project.toml) is removed; tests that
  exercised it move into the package suite.

## 3. The unified `.toml` format

```toml
[campaign]
name = "graphene_2x2"
root = "/mnt/ceph/users/xgao1/four_index/graphene_2x2"
templates = [".../graphene_00001.xsf", ".../graphene_00002.xsf"]   # type index (1-based) → file

[[orbital]]                  # explicit list — replaces [lattice] nx/ny
type = 1
x = 0.00; y = 0.00; z = 7.5
[[orbital]]
type = 2
x = 0.00; y = 1.42; z = 7.5
# ... one block per orbital, arbitrary Cartesian positions (in the templates' frame, Å)

[pairing]
neighbor_cutoff = 2.6        # on-site + neighbors within cutoff (distance on snapped centers)
# pairs = [[1,1],[1,2],...]  # OPTIONAL: explicit pair list; if present, overrides cutoff

[dielectrics]
eps_out = 1.0
boxes = [[cx, cy, cz, Lx, Ly, Lz, eps], ...]

[solve]
n_quad = 6
edge_refine_level = 2        # OR l_ec = <Float64>
rhs_tol = 1e-3
lhs_tol = 1e-5
gmres_rtol = 1e-5
support_rtol = 1e-4
volume_tol = 1e-5
max_order = 8
max_depth = 128

[batching]
n_centers_per_batch = 1

[eval]
far_pad_steps = 2.0
```

Parsed into a `CampaignInput` struct (the merged successor to both `Campaign` and
`SystemInput`): templates, the explicit orbital list (type + Cartesian position),
pairing config, dielectric boxes/epses/eps_out, solve params, batching, eval. Orbital
**id** = 1-based index in declaration order. Coordinate frame = the templates' native
xsf frame (Ångström); no unit conversion; the dielectric box must be given in the same
frame (validated: every orbital's snapped support should lie inside or near the boxes —
warn otherwise).

## 4. Geometry: snap-to-grid orbital placement

For each `[[orbital]]` (type t, position p):
1. Let `dg = templates[t]` datagrid; `c0 = density_centroid(dg)`; grid-step basis =
   `(At/nx, Bt/ny, Ct/nz)` from `true_cell_vectors(dg)`.
2. Solve `p - c0 = sx·(At/nx) + sy·(Bt/ny) + sz·(Ct/nz)` for real `(sx,sy,sz)` (3×3
   solve in the grid basis — handles the hexagonal a₁/a₂ skew automatically).
3. **Round to nearest integers** `steps = round.(Int, (sx,sy,sz))` → `OrbitalInstance(id,
   t, steps)`. The realized center = `c0 + grid_basis * steps`.
4. **Warn** if `‖p − realized‖ > ½·max grid step` (~0.04 Å) — the request was off-lattice
   and got snapped. **Error** if the z-component can't be represented (the campaign is a
   planar sheet; a non-representable z is almost certainly a unit/frame mistake).

`OrbitalInstance` (template_id + integer steps) is exactly the existing `lattice_batch.jl`
type — reused unchanged. Pair enumeration runs the existing distance-cutoff logic on the
realized centers, so `manifest.jl`'s `enumerate_pairs`/`build_batches` are reused verbatim
(the `enumerate_centers` `nx×ny` generator is replaced by "read the explicit list").

## 5. `.bie` removal and the in-memory single-group API

- Delete `src/utils/system_input.jl`; remove exports `read_system_input`, `SystemInput`,
  `resolved_l_ec`, and the `.bie` overloads of `solve_dielectric_box3d_group` /
  `four_index_integrals`.
- New in-memory convenience: **`four_index_integrals(toml_path)`** runs the full pipeline
  (prepare→solve→consolidate→eval→assemble) **in-process, no files**, returning
  `(; pair_ids, V, ...)` for small systems. Used by package tests and the cross-check.
  The batched campaign path consumes the same `.toml` through the phase functions.
- `resolved_l_ec` logic (min box Lz / 2^level · 1.01) moves into the campaign module as
  `campaign_l_ec(::CampaignInput)`.
- The `multi_rhs.jl` functions currently taking `SystemInput`
  (`assemble_rhs_group`, `build_group_interface`, `solve_dielectric_box3d_group`,
  `four_index_matrix`) are repointed at `CampaignInput` or reached only via the campaign
  path; **each call site is resolved explicitly in the implementation plan** (no silent
  signature drift).
- Migrate the `.bie` test fixtures (`system_small.bie`, `system_lat.bie`,
  `system_spike.bie`, `system_centroid.bie`, `system_pair.bie`, `system_smooth_lat.bie`)
  to `.toml`, and update `test/solver/multi_rhs.jl`, `multi_rhs_vector.jl`,
  `test/utils/system_input.jl` accordingly (the last becomes a `.toml`-parsing test).

## 6. V_ijkl text output

`assemble_v` writes **`V_full.tsv`** (replacing `V_full.jls`):
- Header line: `i j k l V`.
- One whitespace-separated row per evaluated four-index entry: `i j k l value`, where
  `(i,j)` is the target pair ρ_ij and `(k,l)` the source pair ρ_kl, `value =
  V[(i,j),(k,l)]`. All evaluated entries (dense — long-range kernel, no truncation).
- `report.txt` (the V/Vᵀ symmetry diagnostic) is unchanged and remains the accuracy check.
- A small `write_v_table(path, pair_ids, V)` helper in `v_output.jl`; tested by a
  round-trip (write → parse back → equals the matrix).

## 7. SlurmClusterManager migration (in the extension)

In `BoundaryIntegralDistributedExt`:
```julia
using Distributed, SlurmClusterManager
addprocs(SlurmManager())          # reads SLURM_NTASKS/nodelist from the env; one worker/task
```
replaces `ClusterManagers.SlurmManager(np)`. No worker count argument. Local testing path
stays `addprocs(n)`; inline path spawns nothing. The deprecation warning is gone. sbatch
scripts unchanged except the driver `using`s the extension trigger packages.

## 8. Testing

- **Package suite (`Pkg.test()`)** gains, on a tiny `.toml` fixture: `.toml` parsing,
  geometry snapping (incl. the off-lattice warn/error and the hexagonal-skew decomposition),
  manifest enumeration, the serial phase functions end-to-end (the existing mini-campaign
  pipeline test, now in-package), the in-memory `four_index_integrals(toml)`, and the
  `V_full.tsv` round-trip. Existing `lattice_batch.jl`/`batch_io.jl` coverage is retained;
  `.bie`-based tests are migrated to `.toml`, not dropped.
- **Extension / parallel path** validated as before: local `--workers` and the 2-node
  SlurmManager smoke test (`run_campaign.sbatch` on a fresh-root 2×2). Not in `Pkg.test()`.
- Acceptance: `Pkg.test()` green; the in-memory vs batched cross-check (`compare_anchor`,
  now `.toml`-based) agrees to solver tolerance; the 2-node smoke test reproduces the
  single-node `V` and symmetry diagnostic.

## 9. Sequencing (single implementation plan, in this order)

1. **`.toml` core + geometry + in-memory API + fixture migration + delete `.bie`** → get
   `Pkg.test()` green on the new format (the riskiest, most invasive step first).
2. **Fold `manifest.jl` + `tasks.jl` (serial) into `src/campaign/`**; the mini-campaign
   pipeline test moves in-package.
3. **`V_full.tsv` text output** in `assemble_v` + the round-trip test.
4. **`ext/BoundaryIntegralDistributedExt.jl`** with `SlurmClusterManager`; `Project.toml`
   weakdeps/extensions.
5. **Work-repo deployment layer**: thin `driver.jl` calling the extension, updated sbatch +
   campaign `.toml`s; remove the old `CampaignLib` package; re-run the 2×2 + 2-node smoke.

## 10. Out of scope

- The eval near-field cost optimization (shared interface + build-once near correction) —
  separate investigation, flagged in `project_lattice_campaign`.
- Any change to the numerics (FMM/TKM/hcubature/GMRES) — this is input/output/packaging
  only.
- Periodic boundary conditions; distance-truncated V (kernel is long-range — dense V stays).
- `.npy` / Python export of `V_full.tsv` — text is language-agnostic; add later if wanted.
