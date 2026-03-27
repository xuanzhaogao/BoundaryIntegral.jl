# Box3D RHS FMM Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace per-panel RHS evaluation in `single_dielectric_box3d_rhs_adaptive` (volume source) with per-depth batched FMM evaluation, using `fmm_tol = rhs_atol * 0.1`.

**Architecture:** Add a batched RHS evaluator for volume sources in `src/shape/box3d.jl`, then update adaptive refinement to call it per depth. Use existing FMM3D (`lfmm3d`) for gradients at targets, with volume source points as sources.

**Tech Stack:** Julia, FMM3D (`lfmm3d`), existing BoundaryIntegral panels and quadrature utilities.

### Task 1: Add a regression test for volume-source RHS evaluation

**Files:**
- Modify: `test/shape/box3d.jl`

**Step 1: Write the failing test**

```julia
@testset "volume source rhs adaptive uses batched eval" begin
    Lx = 1.0
    Ly = 1.0
    Lz = 1.0
    n_quad = 4
    eps_src = 1.0
    l_ec = 0.5
    rhs_atol = 1e-6
    eps_in = 2.0
    eps_out = 1.0

    vs = GaussianVolumeSource((0.0, 0.0, 0.0), 0.2, 3, 1e-6)

    interface = BI.single_dielectric_box3d_rhs_adaptive(
        Lx, Ly, Lz, n_quad, vs, eps_src, l_ec, rhs_atol, eps_in, eps_out
    )

    # Compare RHS computed directly vs FMM-based evaluation on interface points
    rhs_direct = BI.Rhs_dielectric_box3d(interface, vs, eps_src)
    rhs_fmm = BI.Rhs_dielectric_box3d_fmm3d(interface, vs, eps_src, rhs_atol * 0.1)

    @test norm(rhs_direct - rhs_fmm) / max(norm(rhs_direct), eps()) < 1e-3
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project -e 'using Pkg; Pkg.test()'`
Expected: FAIL because `Rhs_dielectric_box3d_fmm3d` is Float64-only or because the current adaptive path does not use FMM (we will adapt as needed).

**Step 3: Write minimal implementation**

Implement a Float64-only batched RHS evaluator and ensure the adaptive path for volume sources uses it per depth. Adjust the test or implementation so the above test passes.

**Step 4: Run tests to verify it passes**

Run: `julia --project -e 'using Pkg; Pkg.test()'`
Expected: PASS

**Step 5: Commit**

```bash
git add test/shape/box3d.jl src/shape/box3d.jl

git commit -m "Use batched FMM for volume source RHS adaptivity"
```

### Task 2: Refactor and cleanup

**Files:**
- Modify: `src/shape/box3d.jl`

**Step 1: Refactor helper functions**

Extract helper(s) for building target points and mapping FMM gradients back to per-panel test grids. Keep functions local to `box3d.jl` and only export if required.

**Step 2: Run tests again**

Run: `julia --project -e 'using Pkg; Pkg.test()'`
Expected: PASS

**Step 3: Commit**

```bash
git add src/shape/box3d.jl

git commit -m "Refactor batched RHS adaptivity helpers"
```
