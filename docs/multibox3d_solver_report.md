# BoundaryIntegral.jl: Multibox 3D Solver — Technical Report

## 1. Density Screening

Three screening modes are implemented in `src/core/sources.jl`:

**Sharp Screening (`SharpScreening`):**  
Applies a step-function permittivity: `ε(x) = ε_in` if the point is inside the box, `ε_out` otherwise. In multi-box cases, the first matching box wins.

**Soft Screening (`SoftMixPermittivity`, `SoftMixInversePermittivity`):**  
Uses a smooth error-function transition via `_smooth_screen_factor(dist, bandwidth) = (1 + erf(dist/bandwidth)) / 2`. The effective permittivity is a weighted average: `ε_eff = ε_in * S + ε_out * (1 - S)` where `S` is a product of six erf factors (one per box face).

**Application:**  
For all modes, the screened density at each source point is `ρ_screened[s] = ρ[s] / ε_local(x_s)`. For multi-box, a partition-of-unity formula sums contributions across all boxes.

---

## 2. Mesh Refinement

### Initial Panels
Each of the 6 box faces is divided using `best_grid_mn(Lab, Lda, alpha)`, which minimizes the number of panels while keeping the aspect ratio ≤ `alpha ≈ √2`.

### RHS-Based Adaptive Refinement
**Entry point:** `single_dielectric_box3d_rhs_adaptive()` in `src/shape/box3d_rhs_adaptive.jl`

**Loop (depth 0 to `max_depth`):**
1. Generate tensor-product quadrature points + 10×10 verification test points per panel.
2. Classify all targets as **near** (within `5h` of any source, via KDTree) or **far**.
3. Evaluate the RHS at all targets via hybrid FMM/TKM (see §4).
4. Test resolution per panel: compute barycentric Gauss-Legendre interpolation error at the 10×10 test points. If `max_error ≤ rhs_atol`, the panel is resolved.
5. Subdivide unresolved panels 2×2 and repeat.

### Edge/Corner Refinement
After RHS convergence, any panel touching a physical edge or corner with `max(Lab, Lda) > l_ec` is further subdivided 2×2, regardless of the RHS error.

### Panel Structure
`TempPanel3D{T}` (four corners + edge/corner Boolean flags) is used during refinement. Once resolved, panels are converted to `FlatPanel{T,3}` with full n_quad² Gauss-Legendre quadrature information.

---

## 3. Multi-Box Geometry and Interface Construction

**Implemented in:** `src/shape/box3d_multi.jl`

**Stage 1 — Shared Face Detection (`_detect_shared_faces_3d`):**  
For every pair of boxes, checks all 6×6 face combinations for coplanar overlap. Stores overlapping rectangles with their outward normals.

**Stage 2 — Face Subtraction (`_subtract_rects_from_face_3d`):**  
Projects shared regions onto the 2D plane of each external face, builds a grid status map, removes those rectangles from the external face, and records which edges/corners are "physical" (adjacent to a box boundary or a removed shared region).

**Stage 3 — Region Collection:**  
- External face regions → `(eps_in = eps[box], eps_out = eps_out)`
- Shared interface regions → `(eps_in = eps[box_hi], eps_out = eps[box_lo])`

Each region is refined independently by `rect_panel3d_rhs_adaptive_panels()`, then all results are concatenated into a single `DielectricInterface`.

---

## 4. Operator Construction

### LHS: Double-Layer Transpose + Diagonal
**File:** `src/solver/dielectric_box3d.jl`

The operator is: **Lhs = DT + diag(t_i)**

where `DT[i,j] = (n_i · (x_j - x_i)) / (4π |x_j - x_i|³) * w_j` and `t_i = 0.5 * (ε_out + ε_in) / (ε_out - ε_in)`.

The FMM-accelerated version (`lhs_dielectric_box3d_fmm3d`) returns a `LinearMap` using `lfmm3d` for the DT matrix-vector product, making GMRES feasible for large systems.

### RHS: Normal Gradient of Green's Function
For a volume source:

**Direct:** `Rhs[i] = -sum_s w_s * ρ_s * (n_i · (x_i - x_s)) / (4π |x_i - x_s|³) / ε_src`

**FMM-accelerated (`rhs_dielectric_box3d_fmm3d`):** Uses `lfmm3d` with `pgt=2` (gradient output) then contracts with normals.

**Hybrid (`rhs_dielectric_box3d_hybrid`):** Splits targets into near/far and routes accordingly (see §5).

---

## 5. FMM and Near-Field Acceleration

### Far-Field: FMM3D (`lfmm3d`)
- Tolerance: `fmm_tol = rhs_atol * 0.1` (10× tighter than target error)
- Returns gradients (`pgt=2`) at target points
- Kernel normalization: `1/r` (requires manual `1/(4π)` factor)

### Near-Field: TKM3D (`ltkm3dc`)
- Applied to targets within radius `5h` (KDTree classification)
- Wavenumber cutoff: `kmax = π/h` where `h` is the estimated nearest-neighbor source spacing
- Kernel normalization: `1/(4πr)` (no extra factor needed)
- Handles the analytically difficult near-singular interactions accurately

### Classification
`_classify_near_far_targets()` builds a KDTree over source positions and marks each target point (not panel) as near if any source falls within `5h`. This pointwise classification avoids forcing entire large panels into the expensive TKM path.

---

## 6. Overall Algorithm Flow

```
Input: box geometry, volume source (positions/weights/density), ε values, tolerances

1. SCREEN DENSITY
   ρ_eff[s] = ρ[s] / ε_local(x_s)   [sharp or soft mode]

2. DETECT GEOMETRY (multi-box only)
   - Find shared faces between box pairs
   - Subtract shared regions from external faces
   - Collect all face regions with (eps_in, eps_out) labels

3. ADAPTIVE MESH REFINEMENT (per region)
   - Initial coarse panels via best_grid_mn
   - Loop: evaluate RHS via FMM/TKM → test barycentric interpolation error → subdivide
   - Edge/corner post-refinement to size ≤ l_ec

4. ASSEMBLE OPERATORS
   - LHS = LinearMap: x ↦ DT_fmm(x) + diag(t) * x
   - RHS = n · ∇G * ρ_eff evaluated at all panel quadrature points (hybrid FMM/TKM)

5. SOLVE
   - GMRES: Lhs * σ = Rhs   [σ = surface charge density]

6. POST-PROCESS
   - Potential/field at observation points via FMM or direct summation
```

---

## 7. Key Design Choices

| Design Choice | Rationale |
|---|---|
| Adaptive mesh driven by RHS residual | Concentrates panels where the source field is hardest to represent |
| 5h near/far split with KDTree | Avoids TKM overhead for majority of far targets |
| FMM tolerance = 0.1 × RHS tolerance | Prevents FMM errors from dominating refinement decisions |
| Barycentric interpolation error test | Directly checks if tensor-product GL quadrature can represent the RHS on each panel |
| `LinearMap` for LHS | Enables matrix-free GMRES, avoiding O(N²) storage |
| Partition-of-unity for multi-box screening | Handles overlapping box boundaries without discontinuities in soft mode |
