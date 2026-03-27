# Box3D Batched FMM (All Faces) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Batch volume-source RHS adaptivity across all six box faces per refinement iteration, using only `solved` and `unsolved` vectors.

**Architecture:** Replace the per-face loop in the volume-source overload of `single_dielectric_box3d_rhs_adaptive` with a single adaptive loop that builds initial panels for all faces, then runs one FMM per iteration across all unsolved panels. Use `solved` and `unsolved` vectors (no per-depth storage) and keep edge/corner refinement and discretization unchanged.

**Tech Stack:** Julia, FMM3D (`lfmm3d`), existing BoundaryIntegral panel utilities.

### Task 1: Add regression test for cross-face batching behavior

**Files:**
- Modify: `/Users/xgao/codes/BoundaryIntegral.jl/.worktrees/codex/box3d-batched-fmm/test/shape/box3d.jl`

**Step 1: Write the failing test**

```julia
@testset "volume source rhs adaptive batches all faces" begin
    xs = [-0.25, 0.0, 0.25]
    ys = [-0.25, 0.0, 0.25]
    zs = [-0.25, 0.0, 0.25]
    weights = fill(1.0, 3, 3, 3)
    density = fill(1.0, 3, 3, 3)
    vs = BI.VolumeSource{Float64, 3}((xs, ys, zs), weights, density)

    eps_src = 1.0
    rhs_atol = 1e-6

    interface = BI.single_dielectric_box3d_rhs_adaptive(
        1.0, 1.0, 1.0,
        3,
        vs,
        eps_src,
        0.4,
        rhs_atol,
        2.0,
        1.0,
        Float64;
        max_depth = 2,
    )

    @test length(interface.panels) >= 6
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project -e 'using Pkg; Pkg.test()'`
Expected: FAIL because batching across all faces is not yet implemented or helper API missing.

**Step 3: Write minimal implementation**

Implement a new helper to create the initial `TempPanel3D` list for all faces and update the volume-source adaptive path to run a single per-iteration FMM across all faces, using `solved` and `unsolved` vectors.

**Step 4: Run tests to verify it passes**

Run: `julia --project -e 'using Pkg; Pkg.test()'`
Expected: PASS

**Step 5: Commit**

```bash
git add /Users/xgao/codes/BoundaryIntegral.jl/.worktrees/codex/box3d-batched-fmm/test/shape/box3d.jl /Users/xgao/codes/BoundaryIntegral.jl/.worktrees/codex/box3d-batched-fmm/src/shape/box3d.jl

git commit -m "Batch volume-source RHS adaptivity across faces"
```

### Task 2: Refactor helper utilities

**Files:**
- Modify: `/Users/xgao/codes/BoundaryIntegral.jl/.worktrees/codex/box3d-batched-fmm/src/shape/box3d.jl`

**Step 1: Refactor helper functions**

Extract helpers for building initial box faces and for the adaptive loop to keep `single_dielectric_box3d_rhs_adaptive` concise.

**Step 2: Run tests again**

Run: `julia --project -e 'using Pkg; Pkg.test()'`
Expected: PASS

**Step 3: Commit**

```bash
git add /Users/xgao/codes/BoundaryIntegral.jl/.worktrees/codex/box3d-batched-fmm/src/shape/box3d.jl

git commit -m "Refactor batched RHS adaptivity helpers"
```
