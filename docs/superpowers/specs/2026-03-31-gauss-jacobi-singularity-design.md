# Gauss-Jacobi Singularity Handling for box2d

## Problem

Two boundary segments meeting at a box corner produce a density singularity `phi(s) ~ s^{gamma-1}` where `gamma` is the leading eigenvalue of the van Bladel theta ODE for the dielectric wedge. When `gamma < 1`, the density is singular. The exponent depends on the corner angle and the permittivities of the materials meeting at the corner. Standard Gauss-Legendre quadrature on graded panels converges slowly for this singularity. Gauss-Jacobi quadrature with weight `s^{gamma-1}` captures it analytically.

### Singularity exponent computation

The potential power `gamma` is found by solving a transcendental eigenvalue problem from the theta ODE (van Bladel, Sec 4.13). For a two-material corner with interior permittivity `eps` occupying angle `alpha`:

- Even parity: `sin(g*alpha/2)*cos(g*(pi-alpha/2)) + cos(g*alpha/2)*sin(g*(pi-alpha/2))/eps = 0`
- Odd parity: `cos(g*alpha/2)*sin(g*(pi-alpha/2)) + sin(g*alpha/2)*cos(g*(pi-alpha/2))/eps = 0`

For multi-junction corners, the general `theta_ODE_det(angles, epsilons, g)` function from `cornervanbladel2d.jl` / `multicorner2dpowers.jl` handles arbitrary configurations.

The density power is `gamma - 1`. The Gauss-Jacobi weight parameter is `alpha_GJ = gamma - 1` (which is negative when the density is singular).

Reference: `/mnt/home/xgao1/work/four_index_integral_solver/codes/theory/cornervanbladel2d.jl`

## Scope

Extend the existing `box2d` solver pipeline. 2D prototype first, then extend to 3D later.

### What changes
1. Compute singularity exponent `gamma` per corner from permittivities and corner angle
2. Innermost panel at each corner uses Gauss-Jacobi(`gamma-1, 0`) quadrature instead of Gauss-Legendre
3. Near-field corrections for singular panels use HCubature-precomputed dense blocks
4. New `is_singular` flag and `singular_exponent` field on `FlatPanel`

### What doesn't change
- All other panels remain Gauss-Legendre
- Solver structure: LHS = DT + diagonal, RHS = point source eval
- `DielectricInterface` type
- FMM calling convention

## Design

### 1. Singularity Exponent Computation

New utility function `corner_singularity_power(alpha, eps_in, eps_out)` in `src/utils/`:
- Solves the theta ODE eigenvalue problem for the leading power `gamma`
- For a right-angle box corner (`alpha = pi/2`), solves even and odd parity equations, takes the minimum `gamma`
- Uses `Roots.fzero` (or a simple Newton iteration) with initial guess `gamma = 1.0`
- Returns the density exponent `gamma - 1`

For multi-junction corners, a general version `corner_singularity_power(angles, epsilons)` using the transmission matrix determinant.

### 2. Panel Construction

Modify `straight_line_adaptive_panels` in `src/shape/box2d.jl`:
- After existing adaptive refinement, identify the innermost panel at each corner (the panel whose endpoint touches the corner vertex)
- Compute `a = gamma - 1` via `corner_singularity_power` for the local corner geometry
- Replace its GL nodes/weights with `FastGaussQuadrature.gaussjacobi(n, a, 0.0)`
- GJ nodes are on `[-1, 1]` with weight `((1-x)/2)^a`. Map to `[0, h]` so singularity is at corner endpoint `s = 0`
- Add `is_singular::Bool` field to `FlatPanel` (default `false`)
- Add `singular_exponent::T` field to `FlatPanel` (stores `a = gamma - 1`)

### 3. FMM Charge Computation

No change to FMM matvec code. The GJ weights baked into the singular panel already absorb the `s^{gamma-1}` measure. The density vector entries for singular panels represent the smooth part `f(s_i) = s_i^{1-gamma} * phi(s_i)`, and charges are `q_i = gj_weight_i * f(s_i)` — same formula as GL panels.

### 4. Near-Field Correction

Precompute a dense correction block via HCubature for each singular panel x nearby target panel pair:

1. **Precompute `A_near`**: For each target point `t_k` within near-field range of a singular panel, compute the integral directly using HCubature:
   ```
   A_near[k, j] = integral_0^h K(s, t_k) * s^{gamma-1} * L_j(s) ds
   ```
   where `L_j` is the j-th Lagrange basis function on the GJ nodes.

2. **Precompute `A_fmm`**: Direct evaluation of the same kernel at GJ nodes (what FMM would compute):
   ```
   A_fmm[k, j] = K(s_j, t_k) * gj_weight_j
   ```

3. **Correction block**: `delta_A = A_near - A_fmm`

4. **Application**: In the matvec: `y = FMM(x) + delta_A * x_singular`

**Near-field criterion**: Target is "near" if distance to singular panel < `3 * panel_length`.

**Location**: New function `laplace2d_near_correction_singular(...)` in `src/kernel/`.

### 4. LHS Assembly

- `lhs_dielectric_box2d_fmm2d` matvec adds `delta_A * x_singular` after FMM eval
- Dense `lhs_dielectric_box2d` assembles with HCubature-corrected entries for singular panels
- Diagonal term `0.5 * (eps_out + eps_in) / (eps_out - eps_in)` unchanged

### 5. RHS

No special treatment. RHS evaluates `laplace2d_grad(point_source, panel_point, normal)` at quadrature points. For singular panels the points are at GJ nodes; the RHS function `g(t)` is smooth so GJ nodes work fine.

### 6. Testing

Convergence test comparing:
- Current approach: GL-only with graded mesh
- New approach: GJ on innermost panel + graded mesh

Validate with Gauss's law: `total_flux + 1/eps_src ~ 1.0` using a point source near a corner where the singularity matters most. Demonstrate improved convergence rate.

## Files to Modify/Add

| File | Change |
|------|--------|
| `src/core/panels.jl` | Add `is_singular::Bool` and `singular_exponent::T` to `FlatPanel` |
| `src/utils/corner_singularity.jl` | New: compute `gamma` from theta ODE eigenvalue problem |
| `src/shape/box2d.jl` | GJ nodes/weights on innermost corner panels, pass exponent |
| `src/kernel/laplace2d.jl` | Near-field correction for singular panels |
| `src/solver/dielectric_box2d.jl` | Wire correction into LHS matvec |
| `test/solver/dielectric_box2d.jl` | Convergence test |

## Dependencies

- `FastGaussQuadrature.gaussjacobi` (already available)
- `HCubature.hquadrature` (already a dependency)
- `Roots.fzero` — need to add `Roots.jl` as a dependency (or use a simple Newton iteration to avoid the dep)

## References

- Design doc: `design/plans/2026-3-30-gauss-jocabi.md`
- Singularity theory: `/mnt/home/xgao1/work/four_index_integral_solver/codes/theory/cornervanbladel2d.jl`
- Multi-junction: `/mnt/home/xgao1/work/four_index_integral_solver/codes/theory/multicorner2dpowers.jl`
