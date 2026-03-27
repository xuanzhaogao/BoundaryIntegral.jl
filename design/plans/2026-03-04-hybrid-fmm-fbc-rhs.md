# Hybrid FMM3D + FBCPoisson RHS Evaluation Implementation Plan

> Superseded on 2026-03-11 by `docs/plans/2026-03-11-tkm3d-volume-nearfield-design.md` and `docs/plans/2026-03-11-tkm3d-volume-nearfield.md`.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Modify `_rhs_panel3d_resolved_volume_fmm` and `single_dielectric_box3d_rhs_adaptive` (VolumeSource variant) to use FBCPoisson (`lfbc3d`) for panels whose targets are within 5h of source points, and FMM3D (`lfmm3d`) for the rest.

**Architecture:** Estimate grid spacing h from nearest-neighbor distances in source positions. Classify panels as "near" (any center point within 5h of a source) or "far". Evaluate near-panel targets via `lfbc3d(fbc_N, ...)` and far-panel targets via `lfmm3d(fmm_tol, ...)`. Merge gradient results back for resolution checking.

**Tech Stack:** Julia, FMM3D.jl (`lfmm3d`), FBCPoisson.jl (`lfbc3d`), NearestNeighbors.jl (`KDTree`, `knn`)

---

### Task 1: Add `_estimate_source_spacing` helper

**Files:**
- Modify: `src/shape/box3d.jl` (insert before `_rhs_panel3d_resolved_volume_fmm` at line ~285)

**Step 1: Write the helper function**

Add to `src/shape/box3d.jl` before line 286:

```julia
function _estimate_source_spacing(vs::VolumeSource{T, 3}) where T
    n = size(vs.positions, 2)
    n <= 1 && return zero(T)
    tree = KDTree(vs.positions)
    h = typemax(T)
    for i in 1:n
        idxs, dists = knn(tree, view(vs.positions, :, i), 2, true)
        h = min(h, T(dists[2]))
    end
    return h
end
```

This builds a KDTree on source positions and finds the minimum nearest-neighbor distance (skipping self at index 1).

**Step 2: Commit**

```bash
git add src/shape/box3d.jl
git commit -m "Add _estimate_source_spacing helper for VolumeSource grid spacing"
```

---

### Task 2: Add `_classify_near_far_panels` helper

**Files:**
- Modify: `src/shape/box3d.jl` (insert after `_estimate_source_spacing`)

**Step 1: Write the helper function**

```julia
function _classify_near_far_panels(panels::Vector{TempPanel3D{T}}, vs::VolumeSource{T, 3}, h::T, h_factor::T = T(5)) where T
    n_panels = length(panels)
    is_near = fill(false, n_panels)
    n_sources = size(vs.positions, 2)
    n_sources == 0 && return is_near

    tree = KDTree(vs.positions)
    radius = h * h_factor

    for (p, tpl) in enumerate(panels)
        cc = (tpl.a .+ tpl.b .+ tpl.c .+ tpl.d) ./ 4
        idxs = inrange(tree, collect(cc), radius)
        if !isempty(idxs)
            is_near[p] = true
        end
    end

    return is_near
end
```

Uses `inrange` (from NearestNeighbors.jl, already a dependency) to efficiently find if any source point is within `5h` of the panel center.

**Step 2: Commit**

```bash
git add src/shape/box3d.jl
git commit -m "Add _classify_near_far_panels helper for near/far panel classification"
```

---

### Task 3: Modify `_rhs_panel3d_resolved_volume_fmm` to accept `fbc_N` and use hybrid evaluation

**Files:**
- Modify: `src/shape/box3d.jl:286-392`

**Step 1: Update the function signature and implementation**

Replace the existing `_rhs_panel3d_resolved_volume_fmm` (lines 286–392) with:

```julia
function _rhs_panel3d_resolved_volume_fmm(
    panels::Vector{TempPanel3D{T}},
    vs::VolumeSource{T, 3},
    eps_src::T,
    ns::Vector{T},
    ws::Vector{T},
    atol::T,
    fmm_tol::T,
    fbc_N::Int,
) where T
    n_panels = length(panels)
    if n_panels == 0
        return Bool[]
    end
    resolved = fill(false, n_panels)
    n_quad = length(ns)
    λ = gl_barycentric_weights(ns, ws)
    n_pts = 10
    xs = range(-one(T), one(T); length = n_pts)
    ys = range(-one(T), one(T); length = n_pts)
    n_test = n_pts * n_pts
    n_per_panel = n_quad * n_quad + n_test

    # Build all targets and normals (same layout as before)
    n_targets = n_panels * n_per_panel
    targets = Matrix{T}(undef, 3, n_targets)
    normals = Matrix{T}(undef, 3, n_targets)

    t = 0
    for tpl in panels
        a, b, c, d = tpl.a, tpl.b, tpl.c, tpl.d
        cc = (a .+ b .+ c .+ d) ./ 4
        bma = b .- a
        dma = d .- a
        normal = tpl.normal

        for i in 1:n_quad
            u = ns[i]
            for j in 1:n_quad
                v = ns[j]
                t += 1
                p = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
                targets[1, t] = p[1]
                targets[2, t] = p[2]
                targets[3, t] = p[3]
                normals[1, t] = normal[1]
                normals[2, t] = normal[2]
                normals[3, t] = normal[3]
            end
        end

        for u in xs
            for v in ys
                t += 1
                p = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
                targets[1, t] = p[1]
                targets[2, t] = p[2]
                targets[3, t] = p[3]
                normals[1, t] = normal[1]
                normals[2, t] = normal[2]
                normals[3, t] = normal[3]
            end
        end
    end

    sources, charges = _volume_source_fmm_sources(vs)

    # Classify panels as near or far
    h = _estimate_source_spacing(vs)
    is_near = _classify_near_far_panels(panels, vs, h)

    # Collect target indices for near and far panels
    near_idxs = Int[]
    far_idxs = Int[]
    for p in 1:n_panels
        panel_range = ((p - 1) * n_per_panel + 1):(p * n_per_panel)
        if is_near[p]
            append!(near_idxs, panel_range)
        else
            append!(far_idxs, panel_range)
        end
    end

    n_near = length(near_idxs)
    n_far = length(far_idxs)

    @info "    rhs panel hybrid evaluation, source points: $(length(charges)), near targets: $n_near, far targets: $n_far"

    # Evaluate gradient at all targets
    rhs_vals = Vector{T}(undef, n_targets)

    # Far-field: FMM3D
    if n_far > 0
        far_targets = targets[:, far_idxs]
        vals_far = lfmm3d(fmm_tol, sources, charges = charges, targets = far_targets, pgt = 2)
        grad_far = vals_far.gradtarg
        for (k, i) in enumerate(far_idxs)
            rhs_vals[i] = -dot(normals[:, i], grad_far[:, k]) / (4π * eps_src)
        end
    end

    # Near-field: FBCPoisson
    if n_near > 0
        near_targets = targets[:, near_idxs]
        _, grad_near = lfbc3d(fbc_N, sources, charges, near_targets, fmm_tol, 2)
        for (k, i) in enumerate(near_idxs)
            rhs_vals[i] = -dot(normals[:, i], grad_near[:, k]) / (4π * eps_src)
        end
    end

    # Check resolution per panel (unchanged logic)
    idx = 0
    for p in 1:n_panels
        quad_vals = Matrix{T}(undef, n_quad, n_quad)
        for i in 1:n_quad
            for j in 1:n_quad
                idx += 1
                quad_vals[i, j] = rhs_vals[idx]
            end
        end

        err = zero(T)
        max_ref = zero(T)
        for u in xs
            rx = T.(barycentric_row(ns, λ, u))
            for v in ys
                ry = T.(barycentric_row(ns, λ, v))
                approx = zero(T)
                for i in 1:n_quad
                    for j in 1:n_quad
                        approx += quad_vals[i, j] * rx[i] * ry[j]
                    end
                end
                idx += 1
                exact = rhs_vals[idx]
                err = max(err, abs(exact - approx))
                max_ref = max(max_ref, abs(exact))
            end
        end
        resolved[p] = err <= atol
    end

    return resolved
end
```

Key changes from original:
- New parameter `fbc_N::Int`
- After building targets, classify panels via `_classify_near_far_panels`
- Split targets into `near_idxs` / `far_idxs`
- Far targets → `lfmm3d(..., pgt=2)`, near targets → `lfbc3d(fbc_N, ..., fmm_tol, 2)`
- Resolution checking logic is identical

**Step 2: Commit**

```bash
git add src/shape/box3d.jl
git commit -m "Modify _rhs_panel3d_resolved_volume_fmm to use hybrid FMM+FBC evaluation"
```

---

### Task 4: Propagate `fbc_N` through `single_dielectric_box3d_rhs_adaptive`

**Files:**
- Modify: `src/shape/box3d.jl:753-820`

**Step 1: Add `fbc_N` keyword argument and pass it through**

Update the function signature at line 753 to add `fbc_N::Int = 64` keyword and pass it to `_rhs_panel3d_resolved_volume_fmm`:

Change line 766-767 from:
```julia
    max_depth::Int = 100,
    alpha::T = sqrt(T(2)),
```
to:
```julia
    max_depth::Int = 100,
    alpha::T = sqrt(T(2)),
    fbc_N::Int = 64,
```

Change line 779 from:
```julia
        resolved = _rhs_panel3d_resolved_volume_fmm(unsolved, vs, eps_src, ns, ws, rhs_atol, fmm_tol)
```
to:
```julia
        resolved = _rhs_panel3d_resolved_volume_fmm(unsolved, vs, eps_src, ns, ws, rhs_atol, fmm_tol, fbc_N)
```

**Step 2: Commit**

```bash
git add src/shape/box3d.jl
git commit -m "Propagate fbc_N keyword through single_dielectric_box3d_rhs_adaptive"
```

---

### Task 5: Also propagate `fbc_N` through `rect_panel3d_rhs_adaptive_panels` (VolumeSource variant)

**Files:**
- Modify: `src/shape/box3d.jl:467-547` (the VolumeSource variant of `rect_panel3d_rhs_adaptive_panels`)

**Step 1: Update the function**

In `rect_panel3d_rhs_adaptive_panels` (VolumeSource variant, line 467), add `fbc_N::Int` parameter after `max_depth::Int`:

Change the signature (lines 467-482) to include `fbc_N::Int` after `max_depth`:
```julia
function rect_panel3d_rhs_adaptive_panels(
    a::NTuple{3, T},
    b::NTuple{3, T},
    c::NTuple{3, T},
    d::NTuple{3, T},
    n_quad::Int,
    vs::VolumeSource{T, 3},
    eps_src::T,
    normal::NTuple{3, T},
    is_edge::NTuple{4, Bool},
    is_corner::NTuple{4, Bool},
    alpha::T,
    l_ec::T,
    rhs_atol::T,
    max_depth::Int,
    fbc_N::Int;
) where T
```

Change the call to `_rhs_panel3d_resolved_volume_fmm` at line 512 to include `fbc_N`:
```julia
        resolved = _rhs_panel3d_resolved_volume_fmm(current, vs, eps_src, ns, ws, rhs_atol, fmm_tol, fbc_N)
```

**Step 2: Commit**

```bash
git add src/shape/box3d.jl
git commit -m "Propagate fbc_N through rect_panel3d_rhs_adaptive_panels VolumeSource variant"
```

---

### Task 6: Write a test for the hybrid evaluation

**Files:**
- Modify: `test/shape/box3d.jl` (or create if needed)

**Step 1: Write a test that verifies near/far classification and hybrid evaluation**

Add a test to `test/shape/box3d.jl`:

```julia
@testset "hybrid FMM+FBC rhs evaluation" begin
    import BoundaryIntegral as BI

    # A single Gaussian volume source centered near one face of the box
    center = (0.0, 0.0, 0.45)  # close to z=+0.5 face
    σ = 0.05
    vs = BI.GaussianVolumeSource(center, σ, 8, 1e-6)

    Lx, Ly, Lz = 1.0, 1.0, 1.0
    eps_in, eps_out = 4.0, 1.0

    # Build with default fbc_N
    interface = BI.single_dielectric_box3d_rhs_adaptive(
        Lx, Ly, Lz, 4, vs, eps_in, 0.2, 1e-4, eps_in, eps_out, Float64;
        max_depth = 3, fbc_N = 64,
    )

    @test BI.num_points(interface) > 0
    @test length(interface.panels) > 0
end
```

**Step 2: Run test**

```bash
julia --project -e 'using Pkg; Pkg.test()'
```
or run just the box3d test file if the full suite is slow.

**Step 3: Commit**

```bash
git add test/shape/box3d.jl
git commit -m "Add test for hybrid FMM+FBC rhs adaptive evaluation"
```

---

## Notes

- `lfbc3d` signature: `lfbc3d(N, sources, charges, targets, nufft_tol, pgt)` — note `charges` is NOT a keyword arg unlike `lfmm3d`
- `lfmm3d` signature: `lfmm3d(thresh, sources; charges=..., targets=..., pgt=2)` — `charges` IS a keyword arg
- `lfbc3d` with `pgt=2` returns `(pot, grad)` where `grad` is `(3, n_targets)` — same layout as `lfmm3d`'s `gradtarg`
- The sign convention: `lfmm3d` gradient has sign such that `rhs = -dot(n, grad) / (4π * eps_src)`. For `lfbc3d`, the gradient is `∂ϕ/∂x_d` of the Coulomb potential, same convention — so the same formula applies.
- `NearestNeighbors.inrange(tree, point, radius)` returns indices of all points within `radius` — already available since NearestNeighbors is a dependency.
