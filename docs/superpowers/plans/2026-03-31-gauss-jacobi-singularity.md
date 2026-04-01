# Gauss-Jacobi Corner Singularity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Gauss-Jacobi quadrature on innermost corner panels in the 2D box solver to analytically capture density singularities `phi(s) ~ s^{gamma-1}`, where `gamma` is computed from the van Bladel theta ODE.

**Architecture:** Extend `FlatPanel` with singular panel fields. Modify `box2d.jl` panel construction to use GJ nodes on innermost corner panels. Precompute HCubature-based near-field correction blocks for singular panel interactions. Wire corrections into the LHS matvec and dense assembly.

**Tech Stack:** Julia, FastGaussQuadrature (gaussjacobi), HCubature, Roots (for fzero), existing BoundaryIntegral.jl infrastructure.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/core/panels.jl` | Add `is_singular`, `singular_exponent` fields to `FlatPanel` |
| `src/utils/corner_singularity.jl` | Already exists. Add `corner_singularity_power(alpha, eps_in, eps_out)` helper |
| `src/utils/barycentric.jl` | Add `gj_barycentric_weights` for Gauss-Jacobi nodes |
| `src/shape/box2d.jl` | Modify `straight_line_adaptive_panels` and builders to create GJ panels at corners |
| `src/kernel/laplace2d_near_singular.jl` | New: HCubature-based near-field correction for singular panels |
| `src/kernel/laplace2d.jl` | Modify direct DT assembly to use correction for singular panels |
| `src/solver/dielectric_box2d.jl` | Wire near-field correction into FMM LHS matvec |
| `src/BoundaryIntegral.jl` | Add new include and Roots dependency |
| `test/utils/corner_singularity.jl` | Test singularity power computation |
| `test/solver/dielectric_box2d.jl` | Add convergence test with singular panels |

---

### Task 1: Extend FlatPanel with singular panel fields

**Files:**
- Modify: `src/core/panels.jl:5-22` (FlatPanel struct)
- Modify: `src/core/panels.jl:25-28` (convenience constructor)

- [ ] **Step 1: Add fields to FlatPanel struct**

In `src/core/panels.jl`, add two fields to `FlatPanel` after `is_edge::Bool`:

```julia
struct FlatPanel{T, D} <: AbstractPanel
    # information about the panel
    normal::NTuple{D, T}
    corners::Vector{NTuple{D, T}} # corners are arranged in a anti-clockwise order (lb, rb, rt, lt)
    is_edge::Bool
    is_singular::Bool
    singular_exponent::T  # gamma - 1, where phi ~ s^{gamma-1}

    # quadrature information (same order in each tangential direction)
    n_quad::Int
    gl_xs::Vector{T}
    gl_ws::Vector{T}

    # quadrature points and weights
    points::Vector{NTuple{D, T}}
    weights::Vector{T}

    # barycentric interpolation weights on GL nodes
    bary_weights::Vector{T}
end
```

- [ ] **Step 2: Update convenience constructor**

Update the existing constructor to accept the new fields, and add a backward-compatible constructor that defaults `is_singular=false, singular_exponent=zero(T)`:

```julia
# Full constructor with singular panel support
function FlatPanel(normal::NTuple{D,T}, corners, is_edge, is_singular, singular_exponent,
                   n_quad, gl_xs, gl_ws, points, weights) where {T, D}
    bary_weights = gl_barycentric_weights(gl_xs, gl_ws)
    return FlatPanel{T,D}(normal, corners, is_edge, is_singular, singular_exponent,
                          n_quad, gl_xs, gl_ws, points, weights, bary_weights)
end

# Backward-compatible constructor (non-singular panels)
function FlatPanel(normal::NTuple{D,T}, corners, is_edge, n_quad, gl_xs, gl_ws, points, weights) where {T, D}
    bary_weights = gl_barycentric_weights(gl_xs, gl_ws)
    return FlatPanel{T,D}(normal, corners, is_edge, false, zero(T),
                          n_quad, gl_xs, gl_ws, points, weights, bary_weights)
end
```

- [ ] **Step 3: Verify existing tests still pass**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'using Pkg; Pkg.test()'`

Expected: All existing tests pass (the backward-compatible constructor means no existing code breaks).

- [ ] **Step 4: Commit**

```bash
git add src/core/panels.jl
git commit -m "feat: add is_singular and singular_exponent fields to FlatPanel"
```

---

### Task 2: Add corner_singularity_power helper

**Files:**
- Modify: `src/utils/corner_singularity.jl:65` (append)
- Create: `test/utils/corner_singularity.jl`

The functions `theta_shooting_even`, `theta_shooting_odd`, and `theta_ODE_det` already exist. We need a convenience function that finds the leading singularity power given corner geometry and permittivities.

- [ ] **Step 1: Write the test**

Create `test/utils/corner_singularity.jl`:

```julia
using BoundaryIntegral
import BoundaryIntegral as BI
using Test

@testset "corner_singularity_power" begin
    # For a right-angle corner (alpha=pi/2) with eps=2, eps_out=1:
    # Known value from cornervanbladel2d.jl: gamma ≈ 1.1066
    # density power = gamma - 1 ≈ 0.1066
    alpha = pi / 2
    gamma = BI.corner_singularity_power(alpha, 2.0, 1.0)
    @test gamma ≈ 1.1066007580762274 atol=1e-6

    # For eps_in = eps_out = 1 (no contrast), gamma should be 1 (no singularity)
    gamma_trivial = BI.corner_singularity_power(alpha, 1.0, 1.0)
    @test gamma_trivial ≈ 1.0 atol=1e-6

    # For high contrast eps=100, gamma should be < 1 (singular density)
    gamma_high = BI.corner_singularity_power(alpha, 100.0, 1.0)
    @test gamma_high < 1.0
    @test gamma_high > 0.0

    # Multi-junction version
    gamma_multi = BI.corner_singularity_power_multi([pi/2], [2.0, 1.0])
    @test gamma_multi ≈ 1.1066007580762274 atol=1e-6
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'include("test/utils/corner_singularity.jl")'`

Expected: FAIL with "UndefVarError: corner_singularity_power not defined"

- [ ] **Step 3: Implement corner_singularity_power**

Append to `src/utils/corner_singularity.jl`:

```julia
using Roots: fzero

"""
    corner_singularity_power(alpha, eps_in, eps_out)

Compute the leading potential power `gamma` for a 2D dielectric wedge corner.
The interior material with permittivity `eps_in` occupies angle `alpha`.
The exterior has permittivity `eps_out`.

Returns `gamma` such that the potential behaves as `r^gamma` and the
SLP density as `s^{gamma-1}` near the corner. When `gamma < 1` the
density is singular.

Uses both even and odd parity shooting functions, returns the minimum gamma > 0.
"""
function corner_singularity_power(alpha::Real, eps_in::Real, eps_out::Real)
    eps_ratio = eps_in / eps_out
    # Even and odd parity eigenvalue equations
    g_even = fzero(g -> theta_shooting_even(Float64(alpha), Float64(eps_ratio), Float64(g)), 1.0)
    g_odd = fzero(g -> theta_shooting_odd(Float64(alpha), Float64(eps_ratio), Float64(g)), 1.0)
    # Return the leading (smallest positive) power
    candidates = filter(g -> g > 0, [g_even, g_odd])
    return minimum(candidates)
end

"""
    corner_singularity_power_multi(angles, epsilons)

Compute the leading potential power `gamma` for a multi-junction 2D dielectric corner.
`angles` is a vector of length nm-1 (the last angle is 2pi - sum(angles)).
`epsilons` is a vector of length nm of relative permittivities.

Uses `theta_ODE_det` to find the smallest positive root.
"""
function corner_singularity_power_multi(angles::AbstractVector, epsilons::AbstractVector)
    return fzero(g -> theta_ODE_det(Float64.(angles), Float64.(epsilons), g), 1.0)
end
```

- [ ] **Step 4: Add Roots dependency**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'using Pkg; Pkg.add("Roots")'`

Add `using Roots` to the imports in `src/BoundaryIntegral.jl` (line 9, after `using ForwardDiff`):

```julia
using Roots
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'include("test/utils/corner_singularity.jl")'`

Expected: All 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/utils/corner_singularity.jl src/BoundaryIntegral.jl Project.toml Manifest.toml test/utils/corner_singularity.jl
git commit -m "feat: add corner_singularity_power for computing density exponent"
```

---

### Task 3: Add barycentric weights for Gauss-Jacobi nodes

**Files:**
- Modify: `src/utils/barycentric.jl` (append)

Gauss-Jacobi nodes don't have the nice closed-form barycentric weights that GL nodes do. We use explicit Lagrange basis computation instead.

- [ ] **Step 1: Write test**

Add to a new file `test/utils/barycentric_gj.jl`:

```julia
using BoundaryIntegral
import BoundaryIntegral as BI
using FastGaussQuadrature
using Test

@testset "gj_barycentric_weights" begin
    # Test that interpolation of a polynomial is exact
    n = 8
    alpha = -0.3  # a generic Jacobi exponent
    xs, ws = gaussjacobi(n, alpha, 0.0)

    bw = BI.gj_barycentric_weights(xs)

    # Interpolate a polynomial of degree n-1 at a non-node point
    f(x) = 3x^3 - 2x^2 + x - 1
    fvals = f.(xs)

    xq = 0.37  # arbitrary test point
    r = zeros(n)
    BI.barycentric_row!(r, xs, bw, xq)
    interp_val = dot(r, fvals)
    @test interp_val ≈ f(xq) atol=1e-12
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'using LinearAlgebra; include("test/utils/barycentric_gj.jl")'`

Expected: FAIL with "UndefVarError: gj_barycentric_weights not defined"

- [ ] **Step 3: Implement gj_barycentric_weights**

Append to `src/utils/barycentric.jl`:

```julia
"""
    gj_barycentric_weights(x)

Compute barycentric interpolation weights for arbitrary nodes `x` (including
Gauss-Jacobi nodes). Uses the explicit formula:
    lambda_j = 1 / prod_{k != j} (x_j - x_k)
with normalization for numerical stability.
"""
function gj_barycentric_weights(x::AbstractVector{T}) where T
    n = length(x)
    TF = float(T)
    λ = ones(TF, n)
    for j in 1:n
        for k in 1:n
            k == j && continue
            λ[j] /= (x[j] - x[k])
        end
    end
    λ ./= maximum(abs, λ)
    return λ
end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'using LinearAlgebra; include("test/utils/barycentric_gj.jl")'`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/barycentric.jl test/utils/barycentric_gj.jl
git commit -m "feat: add gj_barycentric_weights for arbitrary node interpolation"
```

---

### Task 4: Create singular line panels in box2d

**Files:**
- Modify: `src/shape/box2d.jl:13-23` (`line_panel2d_discretize`)
- Modify: `src/shape/box2d.jl:44-78` (`straight_line_adaptive_panels`)
- Modify: `src/shape/box2d.jl:80-92` (`single_dielectric_box2d`)

- [ ] **Step 1: Write test for singular panel creation**

Create `test/shape/box2d_singular.jl`:

```julia
using BoundaryIntegral
import BoundaryIntegral as BI
using FastGaussQuadrature
using Test

@testset "singular panel creation" begin
    n_quad = 8
    # Create a box with singular panels
    box = BI.single_dielectric_box2d(1.0, 1.0, n_quad, 0.2, 0.05, 5.0, 1.0, Float64; use_singular=true)

    # Check that some panels are marked singular
    n_singular = count(p -> p.is_singular, box.panels)
    @test n_singular > 0

    # Each corner has 2 edges meeting -> 2 singular panels per corner, 4 corners -> 8 singular panels
    @test n_singular == 8

    # Check that singular panels have non-zero exponent
    for p in box.panels
        if p.is_singular
            @test p.singular_exponent != 0.0
        else
            @test p.singular_exponent == 0.0
        end
    end

    # Check that non-singular box still works
    box_regular = BI.single_dielectric_box2d(1.0, 1.0, n_quad, 0.2, 0.05, 5.0, 1.0, Float64; use_singular=false)
    @test count(p -> p.is_singular, box_regular.panels) == 0
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'include("test/shape/box2d_singular.jl")'`

Expected: FAIL (no `use_singular` kwarg)

- [ ] **Step 3: Add line_panel2d_singular_discretize**

Add a new function in `src/shape/box2d.jl` after `line_panel2d_discretize` (after line 23):

```julia
# 1d line panel with Gauss-Jacobi quadrature for singular density at endpoint a
# The density behaves as |s|^exponent near endpoint a, where exponent = gamma - 1
# GJ nodes are on [-1,1] with weight ((1+x)/2)^exponent; we map so singularity is at a
function line_panel2d_singular_discretize(a::NTuple{2, T}, b::NTuple{2, T}, n_quad::Int,
                                          exponent::T, normal::NTuple{2, T}) where T
    # gaussjacobi(n, alpha, beta): weight = (1-x)^alpha * (1+x)^beta on [-1,1]
    # We want singularity at endpoint a. Map: s = (1+t)/2 * L, so s=0 (at a) when t=-1.
    # Weight (1+t)^exponent corresponds to beta=exponent in gaussjacobi.
    gj_xs_f64, gj_ws_f64 = gaussjacobi(n_quad, 0.0, Float64(exponent))
    gj_xs = T.(gj_xs_f64)
    gj_ws = T.(gj_ws_f64)

    # Map from [-1,1] to physical panel [a, b]: p = (a+b)/2 + t*(b-a)/2
    points = [(b .+ a) ./ 2 .+ gj_xs[i] .* (b .- a) ./ 2 for i in 1:n_quad]
    L = norm(b .- a)
    weights = gj_ws .* L ./ 2

    @assert norm(normal) ≈ 1 "Normal is not a unit vector"
    @assert dot(normal, b .- a) < 1e-10 "Normal is not perpendicular to the line segment"

    bary_weights = gj_barycentric_weights(gj_xs)
    return FlatPanel{T,2}(normal, [a, b], true, true, exponent,
                          n_quad, gj_xs, gj_ws, points, weights, bary_weights)
end
```

- [ ] **Step 4: Modify straight_line_adaptive_panels to support singular innermost panels**

Replace the function `straight_line_adaptive_panels` in `src/shape/box2d.jl` (lines 44-78) with:

```julia
function straight_line_adaptive_panels(sp::NTuple{2, T}, ep::NTuple{2, T}, ns::Vector{T}, ws::Vector{T}, normal::NTuple{2, T}, l_panel::T, l_corner::T; use_singular::Bool=false, singular_exponent::T=zero(T)) where T

    l_line = norm(ep .- sp)
    n_divide_rough = ceil(Int, l_line / l_panel)

    # this gives a rough division
    rough_panels = divide_temp_panel2d(TempPanel2D(sp, ep, true, true, normal), n_divide_rough)
    fine_panels = Vector{TempPanel2D{T}}()

    while !isempty(rough_panels)
        tpl = popfirst!(rough_panels)
        # check if the panel is a corner panel
        if tpl.is_a_corner || tpl.is_b_corner
            # if the panel is a corner panel, and the length is less than or equal to l_corner, then it is a fine panel
            panel_length = norm(tpl.a .- tpl.b)
            if panel_length <= l_corner
                push!(fine_panels, tpl)
            else # refine the panel by two if the length is greater than l_corner
                refined_panels = divide_temp_panel2d(tpl, 2)
                append!(rough_panels, refined_panels)
            end
        else
            # if the panel is a none-corner panel, then it is a fine panel (we already have a rough division)
            push!(fine_panels, tpl)
        end
    end

    # discretize the fine panels
    panels = Vector{FlatPanel{T, 2}}()
    for tpl in fine_panels
        if use_singular && (tpl.is_a_corner || tpl.is_b_corner)
            # Innermost corner panel: use Gauss-Jacobi quadrature
            # Orient so singularity is at endpoint a (the corner vertex)
            if tpl.is_a_corner
                push!(panels, line_panel2d_singular_discretize(tpl.a, tpl.b, length(ns), singular_exponent, tpl.normal))
            else  # is_b_corner: flip so corner is at a
                push!(panels, line_panel2d_singular_discretize(tpl.b, tpl.a, length(ns), singular_exponent, tpl.normal))
            end
        else
            push!(panels, line_panel2d_discretize(tpl.a, tpl.b, ns, ws, tpl.normal))
        end
    end

    return panels
end
```

- [ ] **Step 5: Modify single_dielectric_box2d to pass singular options**

Replace `single_dielectric_box2d` in `src/shape/box2d.jl` (lines 80-92):

```julia
function single_dielectric_box2d(Lx::T, Ly::T, n_quad::Int, l_panel::T, l_corner::T, eps_in::T, eps_out::T, ::Type{T} = Float64; use_singular::Bool=false) where T
    ns, ws = gausslegendre(n_quad)
    hx = Lx / 2
    hy = Ly / 2
    t0 = zero(T)

    singular_exponent = zero(T)
    if use_singular
        gamma = corner_singularity_power(T(pi/2), eps_in, eps_out)
        singular_exponent = T(gamma - 1)
    end

    panels = Vector{FlatPanel{T, 2}}()
    for (sp, ep, normal) in zip([(-hx, hy), (hx, hy), (hx, -hy), (-hx, -hy)], [(hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy)], [(t0, one(T)), (one(T), t0), (t0, -one(T)), (-one(T), t0)])
        append!(panels, straight_line_adaptive_panels(sp, ep, ns, ws, normal, l_panel, l_corner; use_singular=use_singular, singular_exponent=singular_exponent))
    end

    return DielectricInterface(panels, fill(eps_in, length(panels)), fill(eps_out, length(panels)))
end
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'include("test/shape/box2d_singular.jl")'`

Expected: PASS

- [ ] **Step 7: Verify existing tests still pass**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'using Pkg; Pkg.test()'`

Expected: All existing tests pass (backward compatible via `use_singular=false` default).

- [ ] **Step 8: Commit**

```bash
git add src/shape/box2d.jl test/shape/box2d_singular.jl
git commit -m "feat: add Gauss-Jacobi singular panels at box2d corners"
```

---

### Task 5: Implement near-field correction for singular panels

**Files:**
- Create: `src/kernel/laplace2d_near_singular.jl`
- Modify: `src/BoundaryIntegral.jl` (add include)

- [ ] **Step 1: Write test for near-field correction**

Create `test/kernel/laplace2d_near_singular.jl`:

```julia
using BoundaryIntegral
import BoundaryIntegral as BI
using LinearAlgebra
using FastGaussQuadrature
using Test

@testset "laplace2d singular near correction" begin
    # Create a singular panel on [0, 0.05] along x-axis
    n_quad = 8
    exponent = 0.1066  # gamma - 1 for eps=2 right-angle corner
    a = (0.0, 0.0)
    b = (0.05, 0.0)
    normal = (0.0, 1.0)
    panel_src = BI.line_panel2d_singular_discretize(a, b, n_quad, exponent, normal)

    # Create a regular target panel on y-axis, close to origin (near-field)
    ns, ws = gausslegendre(n_quad)
    c = (0.0, 0.0)
    d = (0.0, 0.05)
    normal_trg = (-1.0, 0.0)
    panel_trg = BI.line_panel2d_discretize(c, d, ns, ws, normal_trg)

    # Compute near-field correction block
    A_near, A_direct = BI.laplace2d_near_singular_block(panel_src, panel_trg)

    # A_near should be n_trg x n_src
    @test size(A_near) == (n_quad, n_quad)
    @test size(A_direct) == (n_quad, n_quad)

    # Correction should be nonzero (panels are close)
    delta_A = A_near - A_direct
    @test norm(delta_A) > 1e-10

    # Verify A_near against a known integral for simple case:
    # The integral should be finite even though the density is singular
    @test all(isfinite, A_near)
    @test all(isfinite, A_direct)
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'include("test/kernel/laplace2d_near_singular.jl")'`

Expected: FAIL

- [ ] **Step 3: Implement laplace2d_near_singular.jl**

Create `src/kernel/laplace2d_near_singular.jl`:

```julia
using HCubature

"""
    laplace2d_near_singular_block(panel_src::FlatPanel{T,2}, panel_trg::FlatPanel{T,2};
                                   rtol=1e-12, atol=1e-14) where T

Compute the near-field interaction block between a singular source panel and a target panel.

Returns `(A_near, A_direct)`:
- `A_near[k,j]`: HCubature-computed integral of `K(s, t_k) * s^exponent * L_j(s) ds`
  where `L_j` is the j-th Lagrange basis function on the GJ nodes.
- `A_direct[k,j]`: Direct point evaluation `K(s_j, t_k) * w_j` (what FMM computes).

The correction is `delta_A = A_near - A_direct`.
"""
function laplace2d_near_singular_block(panel_src::FlatPanel{T,2}, panel_trg::FlatPanel{T,2};
                                        rtol::Float64=1e-12, atol::Float64=1e-14) where T
    @assert panel_src.is_singular "Source panel must be singular"

    n_src = num_points(panel_src)
    n_trg = num_points(panel_trg)
    exponent = panel_src.singular_exponent

    # Source panel endpoints
    a_src, b_src = panel_src.corners[1], panel_src.corners[2]
    L_src = norm(b_src .- a_src)

    # Precompute Lagrange basis coefficients on GJ nodes (monomial form for Horner eval)
    C = lagrange_mono_coeffs(panel_src.gl_xs)

    # Target points and normals
    trg_points = panel_trg.points
    trg_normal = panel_trg.normal

    A_near = zeros(T, n_trg, n_src)
    A_direct = zeros(T, n_trg, n_src)

    r_basis = zeros(T, n_src)

    for k in 1:n_trg
        t_point = trg_points[k]
        t_normal = trg_normal

        # A_direct: what FMM would compute (point source at GJ nodes)
        for j in 1:n_src
            s_point = panel_src.points[j]
            A_direct[k, j] = laplace2d_grad(s_point, t_point, t_normal) * panel_src.weights[j]
        end

        # A_near: HCubature integral over source panel
        # Parameterize: s in [0, L_src], physical point = a_src + s/L_src * (b_src - a_src)
        # Map to reference: t_ref = 2*s/L_src - 1 in [-1, 1]
        for j in 1:n_src
            function integrand(s_vec)
                s = s_vec[1]
                t_ref = 2 * s / L_src - 1  # map to [-1,1]
                # Physical source point
                s_point = a_src .+ (s / L_src) .* (b_src .- a_src)
                # Kernel evaluation
                K_val = laplace2d_grad(s_point, t_point, t_normal)
                # Lagrange basis L_j at t_ref
                eval_lagrange_horner!(r_basis, C, t_ref)
                L_j = r_basis[j]
                # Integrand: K(s, t_k) * s^exponent * L_j(s) * (2/L_src)^exponent
                # The weight s^exponent accounts for the singular density measure
                # We integrate in physical s, so the measure is s^exponent * ds
                return K_val * s^exponent * L_j
            end
            val, err = hquadrature(s -> integrand([s]), 0.0, Float64(L_src); rtol=rtol, atol=atol)
            A_near[k, j] = T(val)
        end
    end

    return A_near, A_direct
end

"""
    laplace2d_near_singular_corrections(interface::DielectricInterface{FlatPanel{T,2}, T};
                                         range_factor=3.0, rtol=1e-12, atol=1e-14) where T

Precompute all near-field correction blocks for singular panels in the interface.

Returns a vector of `(delta_A, src_indices, trg_indices)` tuples, where:
- `delta_A = A_near - A_direct` is the correction block
- `src_indices` are the global indices of the singular source panel's quadrature points
- `trg_indices` are the global indices of the near target panel's quadrature points
"""
function laplace2d_near_singular_corrections(interface::DielectricInterface{FlatPanel{T,2}, T};
                                              range_factor::Float64=3.0, rtol::Float64=1e-12, atol::Float64=1e-14) where T
    corrections = Vector{Tuple{Matrix{T}, UnitRange{Int}, UnitRange{Int}}}()

    n_panels = length(interface.panels)

    # Compute global index offsets for each panel
    offsets = zeros(Int, n_panels)
    for i in 2:n_panels
        offsets[i] = offsets[i-1] + num_points(interface.panels[i-1])
    end

    for (i, panel_src) in enumerate(interface.panels)
        !panel_src.is_singular && continue

        a_src, b_src = panel_src.corners[1], panel_src.corners[2]
        L_src = norm(b_src .- a_src)
        center_src = (a_src .+ b_src) ./ 2

        src_range = (offsets[i]+1):(offsets[i]+num_points(panel_src))

        for (j, panel_trg) in enumerate(interface.panels)
            # Skip self (same panel)
            i == j && continue

            a_trg, b_trg = panel_trg.corners[1], panel_trg.corners[2]
            center_trg = (a_trg .+ b_trg) ./ 2

            dist = norm(center_src .- center_trg)
            if dist < range_factor * L_src
                A_near, A_direct = laplace2d_near_singular_block(panel_src, panel_trg; rtol=rtol, atol=atol)
                delta_A = A_near - A_direct
                trg_range = (offsets[j]+1):(offsets[j]+num_points(panel_trg))
                push!(corrections, (delta_A, src_range, trg_range))
            end
        end
    end

    return corrections
end
```

- [ ] **Step 4: Add include to BoundaryIntegral.jl**

In `src/BoundaryIntegral.jl`, after line 67 (`include("kernel/laplace2d.jl")`), add:

```julia
include("kernel/laplace2d_near_singular.jl")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'include("test/kernel/laplace2d_near_singular.jl")'`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/kernel/laplace2d_near_singular.jl src/BoundaryIntegral.jl test/kernel/laplace2d_near_singular.jl
git commit -m "feat: add HCubature near-field correction for singular 2D panels"
```

---

### Task 6: Wire corrections into LHS assembly

**Files:**
- Modify: `src/solver/dielectric_box2d.jl`

- [ ] **Step 1: Write test for corrected LHS**

Add to `test/solver/dielectric_box2d.jl` (append a new testset):

```julia
@testset "dielectric_box2d with singular panels" begin
    eps_box = 5.0
    box = BI.single_dielectric_box2d(1.0, 1.0, 8, 0.2, 0.05, eps_box, 1.0, Float64; use_singular=true)
    lhs = BI.lhs_dielectric_box2d(box)
    lhs_fmm2d = BI.lhs_dielectric_box2d_fmm2d(box, 1e-12)
    rhs = BI.rhs_dielectric_box2d(box, BI.PointSource((0.1, 0.1), 1.0), eps_box)
    ws = BI.all_weights(box)

    x = BI.solve_lu(lhs, rhs)
    @test norm(lhs * x - rhs) < 1e-10

    total_flux = dot(ws, x)
    @test isapprox(total_flux + 1.0 / eps_box, 1.0, atol = 1e-3)

    x_gmres = BI.solve_gmres(lhs_fmm2d, rhs, 1e-12, 1e-12)
    @test norm(lhs_fmm2d * x_gmres - rhs) < 1e-10

    total_flux_gmres = dot(ws, x_gmres)
    @test isapprox(total_flux_gmres + 1.0 / eps_box, 1.0, atol = 1e-3)
end
```

- [ ] **Step 2: Run test to verify it fails or passes with poor accuracy**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'include("test/solver/dielectric_box2d.jl")'`

Note the accuracy — this establishes the baseline before correction.

- [ ] **Step 3: Modify lhs_dielectric_box2d to apply near-field corrections**

Replace `lhs_dielectric_box2d` in `src/solver/dielectric_box2d.jl`:

```julia
function lhs_dielectric_box2d(interface::DielectricInterface{P, T}) where {P <: AbstractPanel, T}
    D_transpose = laplace2d_DT(interface)
    Lhs = D_transpose

    # Apply near-field corrections for singular panels
    has_singular = any(p -> p.is_singular, interface.panels)
    if has_singular
        corrections = laplace2d_near_singular_corrections(interface)
        for (delta_A, src_range, trg_range) in corrections
            Lhs[trg_range, src_range] .+= delta_A
        end
    end

    offset = 0
    for i in 1:length(interface.panels)
        panel = interface.panels[i]
        eps_in = interface.eps_in[i]
        eps_out = interface.eps_out[i]
        n_pts = num_points(panel)
        t = 0.5 * (eps_out + eps_in) / (eps_out - eps_in)
        for j in 1:n_pts
            Lhs[offset + j, offset + j] += t
        end
        offset += n_pts
    end
    return Lhs
end
```

- [ ] **Step 4: Modify lhs_dielectric_box2d_fmm2d to apply corrections in matvec**

Replace `lhs_dielectric_box2d_fmm2d` in `src/solver/dielectric_box2d.jl`:

```julia
function lhs_dielectric_box2d_fmm2d(interface::DielectricInterface{P, T}, tol::Float64 = 1e-12) where {P <: AbstractPanel, T}
    D_transpose = laplace2d_DT_fmm2d(interface, tol)

    # Precompute near-field corrections for singular panels
    has_singular = any(p -> p.is_singular, interface.panels)
    corrections = has_singular ? laplace2d_near_singular_corrections(interface) : nothing

    function g(x)
        Dx = D_transpose * x

        # Apply near-field corrections
        if corrections !== nothing
            for (delta_A, src_range, trg_range) in corrections
                Dx[trg_range] .+= delta_A * x[src_range]
            end
        end

        offset = 0
        for i in 1:length(interface.panels)
            panel = interface.panels[i]
            eps_in = interface.eps_in[i]
            eps_out = interface.eps_out[i]
            n_pts = num_points(panel)
            t = 0.5 * (eps_out + eps_in) / (eps_out - eps_in)
            for j in 1:n_pts
                Dx[offset + j] += t * x[offset + j]
            end
            offset += n_pts
        end

        return Dx
    end

    Lhs = LinearMap{T}(g, num_points(interface), num_points(interface))

    return Lhs
end
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'include("test/solver/dielectric_box2d.jl")'`

Expected: All tests pass (both old and new testsets).

- [ ] **Step 6: Verify all existing tests still pass**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'using Pkg; Pkg.test()'`

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/solver/dielectric_box2d.jl test/solver/dielectric_box2d.jl
git commit -m "feat: wire near-field corrections into box2d LHS for singular panels"
```

---

### Task 7: Convergence test

**Files:**
- Create: `test/solver/dielectric_box2d_convergence.jl`

- [ ] **Step 1: Write convergence test**

Create `test/solver/dielectric_box2d_convergence.jl`:

```julia
using BoundaryIntegral
import BoundaryIntegral as BI
using LinearAlgebra
using Test

@testset "convergence: singular vs regular panels" begin
    eps_box = 5.0
    eps_out = 1.0
    # Point source near a corner to stress the singularity
    ps = BI.PointSource((0.01, 0.01), 1.0)

    errors_regular = Float64[]
    errors_singular = Float64[]
    n_quads = [4, 8, 12, 16]

    for nq in n_quads
        # Regular panels (no singular treatment)
        box_reg = BI.single_dielectric_box2d(1.0, 1.0, nq, 0.2, 0.02, eps_box, eps_out, Float64; use_singular=false)
        lhs_reg = BI.lhs_dielectric_box2d(box_reg)
        rhs_reg = BI.rhs_dielectric_box2d(box_reg, ps, eps_box)
        x_reg = BI.solve_lu(lhs_reg, rhs_reg)
        flux_reg = dot(BI.all_weights(box_reg), x_reg)
        err_reg = abs(flux_reg + 1.0 / eps_box - 1.0)
        push!(errors_regular, err_reg)

        # Singular panels (Gauss-Jacobi)
        box_sing = BI.single_dielectric_box2d(1.0, 1.0, nq, 0.2, 0.02, eps_box, eps_out, Float64; use_singular=true)
        lhs_sing = BI.lhs_dielectric_box2d(box_sing)
        rhs_sing = BI.rhs_dielectric_box2d(box_sing, ps, eps_box)
        x_sing = BI.solve_lu(lhs_sing, rhs_sing)
        flux_sing = dot(BI.all_weights(box_sing), x_sing)
        err_sing = abs(flux_sing + 1.0 / eps_box - 1.0)
        push!(errors_singular, err_sing)

        println("nq=$nq: regular=$err_reg, singular=$err_sing")
    end

    # The singular panel approach should converge faster
    # At higher quadrature orders, singular should be more accurate
    @test errors_singular[end] < errors_regular[end]
end
```

- [ ] **Step 2: Run convergence test**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'include("test/solver/dielectric_box2d_convergence.jl")'`

Expected: PASS, with printed convergence showing singular panels converge faster.

- [ ] **Step 3: Commit**

```bash
git add test/solver/dielectric_box2d_convergence.jl
git commit -m "test: add convergence comparison for singular vs regular panels"
```

---

## Notes for 3D Extension (future work)

Once the 2D prototype validates the approach:
1. Add `line_panel2d_singular_discretize` analog for 3D: tensor product with GJ in one direction (toward edge), GL in the other
2. Modify `rect_panel3d_adaptive_panels` to detect edge panels and apply GJ
3. Extend `laplace3d_near_hcubature.jl` with singular panel support
4. The singularity exponent computation generalizes: for 3D box edges, it's still a 2D cross-section problem (same `corner_singularity_power`)
