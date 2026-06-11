# Multi-node lattice campaign: full V_{ij,kl} on a 10×10 graphene flake

**Date:** 2026-06-10
**Branch:** `multi_rhs` (BoundaryIntegral.jl) + new campaign layer in
`~/work/four_index_integral_solver/codes/lattice_scale/`
**Builds on:** `2026-06-03-per-center-multi-rhs-design.md` (Vector{VolumeSource} core,
block GMRES, `.bie`/SystemInput layer)

## 1. Goal and scope decisions

Push the four-index BIE solver to lattice scale by running many independent batched
solves across cluster nodes, storing the layer densities, and post-evaluating the full
coupling matrix from stored data.

Decisions fixed during brainstorming (2026-06-10):

- **System:** 10×10 lattice cells × 2 sublattice atoms = **200 orbitals**. Each orbital
  is the production localized Wannier function (`graphene_0000{1,2}.xsf`,
  `k_323201_nb_144_c_15`, grid 150×150×192 spanning a 5×5 supercell, sheet at z=7.5)
  **translated** to its lattice site. Sources: 200 on-site ρ_cc plus ~1800 deduped
  neighbor pairs ρ_ij (≈18 neighbors per orbital before dedup) → **≈2000 pair densities**.
- **Output:** the **full V matrix** — every source pair contracted against every target
  pair (≈2000×2000), finite-flake edge effects included. No translation-symmetry
  shortcut, no target cutoff.
- **Accuracy target:** moderate, **~1e-4 in V** (tolerances around RHS 1e-4 / LHS 1e-6 /
  GMRES 1e-6). Shakeout runs may use looser (1e-3-level) settings.
- **Primary deliverable:** **capability demo** — a working, restartable multi-node
  pipeline at this scale. Timing logged per batch, but no paper-grade scaling-figure
  tooling yet.
- **Orchestration:** **Julia Distributed.jl driver + file manifest** (user's choice).
  One Slurm allocation spanning M nodes; one Julia worker per node; all coordination
  state lives in files on ceph so the campaign is restartable by resubmission.

## 2. Orbital frames and the virtual global grid

The production xsf DATAGRID is the **template frame**: 150×150×192 points spanning a
5×5×1 supercell (span vectors 12.2428 × 10.603 (skew) × 14.92), i.e. **30 grid points
per lattice constant in-plane**. Lattice translations are integer multiples of the grid
spacing along the lattice vectors, so:

- Every orbital instance = (template data, integer frame-origin offset). **Translation,
  not `circshift`** — today's `LATTICE` images wrap on the periodic template grid, which
  cannot represent >5 distinct cells per direction. The new instance type carries a
  global integer offset and never wraps.
- All frames are commensurate: a **virtual global index space** (never materialized as a
  dense array) identifies every grid point campaign-wide. Pair densities φ_i·φ_j are
  exact pointwise products on the **intersection of two offset frames**, truncated at
  `support_rtol` and screened pointwise by 1/ε(x) (multi-box-aware, `eps_src = 1`),
  exactly as in the existing group pipeline.
- A warning is emitted if a truncated support touches its template-frame boundary
  (would indicate the 5×5 frame clips the orbital tail at the chosen `support_rtol`).
- **Dielectric geometry:** one global slab box (as in `graphene_8neighbors.bie`,
  eps 3.5, thickness 3.35, sheet plane z=7.5) sized to cover the 10×10 flake plus
  margin; vacuum outside. Defined once in `campaign.toml`.

## 3. Campaign data model (ceph)

```
/mnt/ceph/users/xgao1/four_index/<campaign>/
  campaign.toml        # inputs: xsf paths, lattice (10,10), neighbor cutoff,
                       #   slab box + eps, tolerances, n_quad, support_rtol,
                       #   n_centers_per_batch, julia/thread settings
  manifest.csv         # one row per batch: batch_id, anchor center(s),
                       #   member pair labels (center ids + cell offsets), K
  targets.jls          # union target set T: global indices (+ weights) of all
                       #   pair supports; built by `prepare`
  batches/batch_0042.jls   # solve output (atomic tmp+rename)
  rho_store.jls        # consolidated truncated ρ for all pairs (indices, values,
                       #   weights, labels); built by `consolidate`
  V/V_0042.jls         # post-eval output: K × N_pairs rows (+ labels)
  V_full.jls           # assembled V + symmetry diagnostic report
  logs/                # per-batch logs and error files
```

- **Pair enumeration:** center c = (Rx, Ry, sublattice), Rx,Ry ∈ 0..9. Unique pairs =
  on-site + neighbor pairs within the cutoff, **deduped by canonical ordering**
  (id_i ≤ id_j); each pair is assigned to exactly one batch anchored at its first
  center. `n_centers_per_batch` (default 1, K ≈ 10) merges adjacent centers' lists for
  larger `nd`; the pilot decides its production value.
- **`batch_XXXX.jls` (`BatchResult`):** σ (N_surface × K), interface panels, per-pair
  truncated screened ρ (global indices, values, weights), pair labels, solve stats
  (DOF, iters, timings), parameter echo, format version, and a completeness marker
  written last. This file fully decouples the solve and eval phases.
- **Status lives on disk:** a batch is done iff its output exists and passes the
  completeness check. There is no mutable status column; restart recomputes the
  pending set by scanning the directory.

## 4. Distributed driver

```
driver.jl <campaign_dir> <phase>   # phase ∈ {prepare, solve, consolidate, eval, assemble}
```

Runs on the head node of a user-submitted multi-node allocation (`ccm`, `-C genoa`;
julia 1.12 binary; sbatch templates provided, **user submits**).

1. Head: `Pkg.precompile()` **before** spawning workers (avoids the known GPFS
   compiled-cache race), then `addprocs` via `ClusterManagers.SlurmManager` — **one
   worker per node** (each task uses whole-node OMP for the FMM, which saturates at
   32–64 cores). `exeflags`/env pin `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`.
2. Head reads the manifest, scans outputs → pending list.
3. `pmap(run_batch, pending)`: greedy dynamic scheduling absorbs per-batch cost
   variance; per-task retry; a task error writes `logs/batch_XXXX.err` and leaves no
   output (batch stays pending) without killing the campaign;
   `ProcessExitedException` (node death/OOM) → retried on another worker. Head prints
   rolling progress (done/total, last batch wall time, ETA).
4. Any crash or walltime kill → resubmit the same job; at most in-flight batches redo.
5. Single-node phases run on the head alone: `prepare` (manifest + targets),
   `consolidate` (ρ store), `assemble` (gather `V/*.jls` → V_full, report
   max|V_ab − V_ba|).

`solve` and `eval` are **separate job submissions** (uniform memory/time profile per
job; eval needs all supports anyway).

## 5. Solve task

Per batch (existing code path, new entry point): build the K pair densities on frame
intersections (truncated, screened) → group envelope →
`multi_dielectric_box3d_rhs_adaptive` (shared interface, 1 FMM per refinement depth) →
batched RHS → `BatchedDielectricOperator` → block GMRES → σ (N×K) → write
`BatchResult`. Identical numerics to `solve_dielectric_box3d_group`, except the group
is an **explicit pair list from the manifest** rather than `si.groups[center]`.

## 6. Post-eval task

Each eval task handles one batch: compute Φ_a = u_inc[ρ_a] + u[σ_a] for its K sources
at the shared target set T, then contract against every stored pair density.

- **u[σ_a]:** one `nd=K` batched corrected-FMM surface evaluation
  (`laplace3d_pottrg_fmm3d_corrected_hcubature` path) — all K columns share the same
  surface points, the ideal batching case. Near-panel targets get the hcubature
  correction. Most of T lies inside the slab near the sheet, so the near-correction
  cost at ~10⁷ targets is the **single biggest unknown**; the pilot measures it before
  the campaign is sized.
- **u_inc[ρ_a]:** split by distance from the support bounding box. *Near* (inside or
  within ~2 grid spacings): existing TKM (`ltkm3dc`) box-code evaluation on the
  source's local frame. *Far*: ρ_a's quadrature points as weighted point charges in one
  `nd=K` `lfmm3d` — the trapezoidal far field of a smooth compact density is well
  beyond the 1e-4 target.
- **Contraction:** V[a, kl] = Σ_p w_p ρ_kl(p) Φ_a(p) over kl's index set, for all
  ≈2000 kl — cheap dot products against the consolidated `rho_store.jls`, loaded once
  per worker per job.
- Output: `V/V_XXXX.jls` with K rows.

By symmetry, V_{ij,kl} is computed twice (once from each side's σ, on independently
adapted interfaces); the disagreement max|V_ab − V_ba| is the campaign's built-in
accuracy diagnostic, reported by `assemble`.

## 7. Code changes

**BoundaryIntegral.jl (`multi_rhs` branch) — four additions:**
1. Frame-**translation** orbital instances (global integer offset, no wrap) and pair
   products on frame intersections in the virtual global index space.
2. `assemble_rhs_group` variant taking an explicit pair list.
3. `evaluate_batch_potential(interface, σ::Matrix, sources, targets; ...)` — the
   near/far-split, nd-batched external-target evaluation of §6.
4. `BatchResult` (+ ρ-store) serialization with format version and completeness marker.

**Campaign layer (`codes/lattice_scale/`, work dir — not in the package):**
`campaign.toml` parsing, manifest generation, `driver.jl`, solve/eval task functions,
consolidate/assemble/diagnostic scripts, sbatch templates. Per workflow preference,
Codex writes these scripts from the implementation plan; Claude plans and reviews.

## 8. Validation and pilot ladder

1. **Correctness anchor:** 2×2-cell mini-campaign (8 centers) end-to-end on one node.
   Within-group V entries must match the existing `four_index_integrals` path to solver
   tolerance; cross-group entries checked via V_ab↔V_ba symmetry and hcubature spot
   checks on a few representative (near + far) pairs.
2. **Plumbing smoke test:** driver + SlurmManager on 2 nodes with trivial tasks
   (allocation submitted by the user).
3. **One-batch pilot at 10×10 scale:** measures solve wall time, eval wall time
   (u_inc far-eval and near-correction at ~10⁷ targets), and file sizes → fixes
   `n_centers_per_batch`, node count, and walltime requests.
4. **Full demo:** loose-tolerance 10×10 campaign on ~4 nodes, then the ~1e-4
   production campaign on 8–16 nodes.

## 9. Error handling summary

- Atomic writes (tmp + rename on the same filesystem); completeness marker validated on
  every load.
- Task failure → error log, no output file, batch remains pending; campaign continues.
- Worker loss → pmap retry on surviving workers; head loss/walltime → resubmit, pending
  recomputed from disk.
- Truncated-support-touches-frame-boundary and p_up-cap events are warned and counted
  in batch stats, not silently ignored.

## 10. Out of scope (this campaign)

- Translation-symmetry reduction of the source set (full flake solve is the point).
- Paper-grade scaling figures and instrumentation (retrofit on a later run).
- Distributed/MPI parallelism *within* a single solve (FMM saturates a node; tasks are
  node-sized by construction).
- Periodic boundary conditions; the flake is finite with a finite slab.
