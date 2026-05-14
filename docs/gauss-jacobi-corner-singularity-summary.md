# Gauss-Jacobi Corner Singularity Handling for box2d

## Problem

At box corners, two boundary segments meet and the BIE density develops a singularity `phi(s) ~ s^{gamma-1}`, where `gamma` depends on the corner angle and permittivities. Standard Gauss-Legendre quadrature converges slowly for this singularity.

## Solution

Replace GL quadrature with Gauss-Jacobi quadrature on the innermost corner panels, with the GJ weight matching the singularity exponent. Use HCubature for near-field corrections.

## What was built (7 commits on `singularity` branch)

### 1. `FlatPanel` extension (`src/core/panels.jl`)
- Added `is_singular::Bool` and `singular_exponent::T` fields
- Backward-compatible constructors preserved

### 2. Singularity exponent computation (`src/utils/corner_singularity.jl`)
- `corner_singularity_power(alpha, eps_in, eps_out)` -- solves the van Bladel theta ODE to find the leading power `gamma`
- `corner_singularity_power_multi(angles, epsilons)` -- multi-junction generalization
- Added `Roots.jl` dependency for `fzero`

### 3. GJ barycentric weights (`src/utils/barycentric.jl`)
- `gj_barycentric_weights(x)` -- general-purpose barycentric weights via product formula

### 4. Singular panel construction (`src/shape/box2d.jl`)
- `line_panel2d_singular_discretize(a, b, n_quad, exponent, normal)` -- creates GJ panels
- `straight_line_adaptive_panels` and `single_dielectric_box2d` accept `use_singular=true`
- Automatically computes `gamma` and orients panels so singularity is at the corner vertex

### 5. Near-field correction (`src/kernel/laplace2d_near_singular.jl`, new file)
- `laplace2d_near_singular_block` -- HCubature-based dense block for singular panel x target panel pairs
- `laplace2d_near_singular_corrections` -- finds all near-field pairs, returns correction blocks `delta_A = A_near - A_direct`

### 6. Solver integration (`src/solver/dielectric_box2d.jl`)
- Both `lhs_dielectric_box2d` (dense) and `lhs_dielectric_box2d_fmm2d` (FMM matvec) apply corrections
- **Bug fix discovered during implementation:** diagonal term needs scaling by `(1+t_j)^{exponent}` on GJ panels because the unknown is the smooth part of the density

### 7. Convergence test (`test/solver/dielectric_box2d_convergence.jl`)

## Results (eps_in=1, eps_out=200, gamma=0.67)

| n_quad | GL error | GJ error | Ratio |
|--------|----------|----------|-------|
| 4      | 1.8e-7   | 1.6e-5   | 0.01x |
| 8      | 6.2e-8   | 7.6e-7   | 0.08x |
| 12     | 2.5e-8   | 7.8e-8   | 0.3x  |
| **16** | **1.2e-8** | **2.8e-9** | **4.4x** |

GJ panels win at higher quadrature orders with the gap growing -- the asymptotic convergence rate is faster because the singularity is captured analytically rather than resolved by brute-force mesh refinement.

## Files changed/created

- Modified: `src/core/panels.jl`, `src/shape/box2d.jl`, `src/solver/dielectric_box2d.jl`, `src/BoundaryIntegral.jl`, `src/utils/corner_singularity.jl`, `src/utils/barycentric.jl`
- Created: `src/kernel/laplace2d_near_singular.jl`
- Tests: 5 new test files under `test/`
- Docs: design spec + implementation plan under `docs/superpowers/`
