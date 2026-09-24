```@meta
CurrentModule = BoundaryIntegral
```

# Lattice campaigns

For large four-index runs over many orbital centers on a lattice, the package provides a
file-backed pipeline driven from a TOML campaign description. It groups pair densities
into batches that share one boundary operator, solves each batch, then evaluates and
contracts every pair against a shared target set.

## Pipeline

```
prepare  →  solve_batch  →  consolidate  →  eval_batch  →  assemble_v
```

- **prepare** — enumerate centers, neighbor pairs, and batches; write the manifest.
- **solve_batch** — per batch: assemble the shared interface, block-GMRES, store σ + screened ρ.
- **consolidate** — build the shared evaluation target set and the contraction store.
- **eval_batch** — per batch: evaluate Φ at the shared targets and contract all pairs into V columns.
- **assemble_v** — gather all V columns into the dense matrix; write `V_full.tsv` + `report.txt`.

Every phase writes its outputs atomically (temp file + rename), so a run is crash-safe and
restartable: re-running a phase skips already-completed batches, and a killed job is
recovered simply by rerunning the phase. `pending_batches(c, :solve | :eval)` reports
what is left to do.

## Campaign TOML

```toml
name = "demo"                          # campaign name
root = "/path/to/output"               # output dir: manifest.tsv, batches/, V/, logs/
templates = ["orb1.xsf", "orb2.xsf"]   # type index -> .xsf path (relative to this file)

[[orbital]]                            # one entry per orbital; id = 1-based order
type = 1                               # index into `templates`
x = 0.0                                # Cartesian center in the templates' frame
y = 0.0
z = 7.5

[pairing]
neighbor_cutoff = 2.6                  # pair orbitals within this distance (default: Inf)
# pairs = [[1, 2], [1, 3]]             # OR give explicit pair overrides instead

[dielectrics]
eps_out = 1.0
boxes = [[0.0, 0.0, 7.5, 90.0, 90.0, 3.35, 3.5]]   # rows of [cx cy cz Lx Ly Lz eps]

[solve]
n_quad = 6
edge_refine_level = 2                  # or set `l_ec` directly
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

## Running serially

```julia
using BoundaryIntegral

c = load_campaign("campaign.toml")
prepare(c)
for id in pending_batches(c, :solve); solve_batch(c, id); end
consolidate(c)
for id in pending_batches(c, :eval); eval_batch(c, id); end
assemble_v(c)
```

The whole pipeline can also be run in memory, without writing files, via
`four_index_integrals("campaign.toml")` (returns `(; pair_ids, V)`).

## Running distributed

The `:solve` and `:eval` phases parallelize over batches. `run_phase` is provided by a
package extension; load its weak dependencies (`Distributed` and `SlurmClusterManager`) to
enable it. Spawning policy: inside a Slurm allocation with more than one task it uses
`SlurmManager()` (one worker per task); otherwise it spawns `workers` local processes; with
`workers = 0` and no allocation it runs inline.

```julia
using BoundaryIntegral, Distributed, SlurmClusterManager

c = load_campaign("campaign.toml")
prepare(c)
run_phase(c, :solve; workers = 4)   # local workers (or SlurmManager inside an allocation)
consolidate(c)
run_phase(c, :eval; workers = 4)
assemble_v(c)
```

A campaign is typically driven by a small script over this API (the package itself ships
no CLI). On Slurm, run one task per node and give each task the whole node via threads:

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --output=logs/solve_%j.out
set -euo pipefail

# Pin both thread pools — unpinned threads silently corrupt timings.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_GLUE_THREADS=8

julia --project -t "$JULIA_GLUE_THREADS" driver.jl campaign.toml solve
```

where `driver.jl` parses `<campaign.toml> <phase>` and calls the phase functions above.
Notes for cluster runs:

- **Precompile on the head process first** (`using Pkg; Pkg.precompile()` before loading
  `Distributed`/`SlurmClusterManager`) so workers don't race to precompile over a shared
  filesystem.
- **One worker per node** (`--ntasks-per-node=1`, `--cpus-per-task=<cores>`); the FMM
  saturates many cores, so node-sized tasks are the right grain.
- **Crash/walltime recovery:** just resubmit the same phase — completed batches are skipped.
