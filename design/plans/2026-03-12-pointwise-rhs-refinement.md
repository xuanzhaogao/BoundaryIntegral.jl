# Pointwise RHS Refinement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Change RHS-based box refinement to classify generated targets individually for hybrid near/far evaluation, avoiding oversized TKM boxes on large panels.

**Architecture:** Reuse the existing `_classify_near_far_targets` helper in `src/shape/box3d.jl` when building the batched quadrature and verification targets inside `_rhs_panel3d_resolved_volume_fmm`. Preserve the current `h_factor * h` radius and `_rhs_volume_targets_hybrid` split so runtime and refinement paths share the same target-level semantics.

**Tech Stack:** Julia, BoundaryIntegral, NearestNeighbors, Test

---

### Task 1: Add failing refinement classification tests

**Files:**
- Modify: `test/shape/box3d.jl`

**Step 1: Write the failing test**

Add tests that:
- Build a localized `VolumeSource` near one large face.
- Generate the same refinement targets used by `_rhs_panel3d_resolved_volume_fmm`.
- Assert `_classify_near_far_panels` marks a whole panel near while `_classify_near_far_targets` marks only a strict subset of those targets near.

**Step 2: Run test to verify it fails**

Run: `julia --project=. --color=no -e 'using Pkg; Pkg.test("BoundaryIntegral"; test_args=["shape/box3d"])'`

Expected: the new test fails because refinement still promotes panel-level near/far tags.

**Step 3: Commit**

Leave uncommitted unless explicitly requested.

### Task 2: Implement pointwise refinement classification

**Files:**
- Modify: `src/shape/box3d.jl`

**Step 1: Write minimal implementation**

- Remove panel-level near/far promotion inside `_rhs_panel3d_resolved_volume_fmm`.
- Classify the generated `targets` directly with `_classify_near_far_targets`.
- Pass the resulting Boolean mask into `_rhs_volume_targets_hybrid`.
- Keep the current `h_factor` default and function signatures unless tests require a small helper extraction.

**Step 2: Run targeted tests to verify they pass**

Run: `julia --project=. --color=no -e 'using Pkg; Pkg.test("BoundaryIntegral"; test_args=["shape/box3d"])'`

Expected: the new tests and existing shape tests pass.

### Task 3: Re-check the orbital reproduction

**Files:**
- No code changes.

**Step 1: Run the reproduction script that previously OOMed**

Run: `julia --project=/mnt/home/xgao1/work/four_index_integral_solver/codes/volume_source /mnt/home/xgao1/work/four_index_integral_solver/codes/volume_source/scripts/orbital_rhs_accuracy.jl`

Expected: interface construction proceeds past the previous OOM point, or at minimum reports far fewer near targets and avoids the huge initial TKM target box.

**Step 2: Commit**

Leave uncommitted unless explicitly requested.
