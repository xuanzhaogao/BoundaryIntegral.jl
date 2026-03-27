# TKM3D Volume Near-Field Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the box3d volume-source near-field backend with `TKM3D.ltkm3dc` and remove `FBCPoisson` from the package code path.

**Architecture:** Keep the existing near/far target classification and adaptive panel refinement logic in `src/shape/box3d.jl`. Route far targets through `FMM3D.lfmm3d` and near targets through `TKM3D.ltkm3dc`, using a helper that derives an explicit `kmax` from the source spacing estimate.

**Tech Stack:** Julia, TKM3D.jl (`ltkm3dc`), FMM3D.jl (`lfmm3d`), NearestNeighbors.jl, Test stdlib

---

### Task 1: Add a failing backend-wiring regression test

**Files:**
- Modify: `test/solver/dielectric_box3d.jl`

**Step 1: Write the failing test**

Add a testset that reads:

```julia
@testset "dielectric_box3d volume backend wiring" begin
    root = pkgdir(BoundaryIntegral)
    entrypoint = read(joinpath(root, "src", "BoundaryIntegral.jl"), String)
    box3d = read(joinpath(root, "src", "shape", "box3d.jl"), String)

    @test occursin("using TKM3D", entrypoint)
    @test !occursin("using FBCPoisson", entrypoint)
    @test occursin("ltkm3dc", box3d)
    @test !occursin("lfbc3d", box3d)
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project -e 'using Pkg; Pkg.test(test_args=["solver/dielectric_box3d.jl"])'`

Expected: FAIL because `src/BoundaryIntegral.jl` still imports `FBCPoisson` and `src/shape/box3d.jl` still calls `lfbc3d`.

### Task 2: Replace the near-field backend with `ltkm3dc`

**Files:**
- Modify: `src/BoundaryIntegral.jl`
- Modify: `src/shape/box3d.jl`

**Step 1: Add the minimal implementation**

- Replace `using FBCPoisson` with `using TKM3D`.
- Add a helper in `src/shape/box3d.jl` that computes an explicit `kmax` from `_estimate_source_spacing(vs)`.
- Update `_rhs_volume_targets_hybrid` so the near-target branch calls:

```julia
vals_near = ltkm3dc(fmm_tol, sources; charges = charges, targets = near_targets, pgt = 2, kmax = kmax)
grad_near = vals_near.gradtarg
```

- Keep the far-target `lfmm3d` branch intact.
- Preserve RHS scaling/sign conventions.

**Step 2: Run the test to verify it passes**

Run: `julia --project -e 'using Pkg; Pkg.test(test_args=["solver/dielectric_box3d.jl"])'`

Expected: PASS for the new backend-wiring test and the existing dielectric box3d tests.

### Task 3: Clean up dependency references and stale docs

**Files:**
- Modify: `docs/plans/2026-03-04-hybrid-fmm-fbc-rhs.md`

**Step 1: Update stale references**

- Rename the old FBC-specific descriptions to TKM terminology, or clearly mark the document as superseded by the new TKM plan/design.

**Step 2: Run a repo search**

Run: `rg -n "FBCPoisson|lfbc3d" src test docs`

Expected: no matches in active code/tests; only intentionally preserved historical context if explicitly marked.

### Task 4: Final verification

**Files:**
- No code changes required

**Step 1: Run targeted verification**

Run: `julia --project -e 'using Pkg; Pkg.test(test_args=["solver/dielectric_box3d.jl"])'`

**Step 2: Run a focused source check**

Run: `rg -n "TKM3D|ltkm3dc|FBCPoisson|lfbc3d" src test Project.toml`

Expected:
- `TKM3D` and `ltkm3dc` appear in the expected files
- `FBCPoisson` and `lfbc3d` do not appear in active code paths
