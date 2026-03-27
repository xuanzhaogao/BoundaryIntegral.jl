# Box3D RHS FMM (Volume Source) Design

## Goal
Speed up `single_dielectric_box3d_rhs_adaptive` for `VolumeSource` by replacing per-panel RHS evaluation with a per-refinement-depth batched FMM evaluation of the RHS at panel test points.

## Architecture
The adaptive refinement in `rect_panel3d_rhs_adaptive_panels` will be extended to support a batched RHS evaluator. For volume sources, we will construct one FMM call per refinement depth, with all test points for candidate panels as targets and all volume source points as sources. The FMM tolerance will be `rhs_atol * 0.1`, per user instruction.

## Data Flow
1. Build initial rough panels.
2. For each refinement depth:
   - Collect all candidate panels at that depth.
   - Build test points for each panel (same grid used for error estimation).
   - Run `lfmm3d` once to compute gradients at all targets.
   - Map target gradients back to panel test points and compute error estimates.
   - Split panels that are not resolved; keep those that are resolved.
3. Apply existing edge/corner refinement and final discretization unchanged.

## Error Handling
If FMM is not available (non-Float64), fall back to the existing direct evaluation. Keep behavior identical for `PointSource` and non-volume source RHS functions.

## Testing
Add a targeted test to validate that the volume-source adaptive interface matches the direct (non-FMM) RHS within tolerance for a small problem. Use the existing box3d tests as a template.
