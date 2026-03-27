# Multi-Dielectric Box 3D Design Spec

## Summary

Add `multi_dielectric_box3d` to BoundaryIntegral.jl — a function that creates a `DielectricInterface` for a system of multiple axis-aligned 3D dielectric boxes. Users specify each box by its center and dimensions (Lx, Ly, Lz), and the function automatically detects shared faces between adjacent boxes, constructs panels with adaptive refinement, and assigns permittivities.

This is the 3D counterpart of the existing `multi_dielectric_box2d`.

## API

```julia
multi_dielectric_box3d(
    n_quad::Int,
    l_ec::T,
    boxes::Vector{NamedTuple{(:center, :Lx, :Ly, :Lz), Tuple{NTuple{3,T}, T, T, T}}},
    epses::Vector{T},
    eps_out::T = one(T);
    alpha::T = T(sqrt(2))
) -> DielectricInterface{FlatPanel{T,3}, T}
```

### Parameters

- `n_quad`: Gauss-Legendre quadrature order (per direction)
- `l_ec`: Maximum edge/corner panel size for adaptive refinement
- `boxes`: Vector of named tuples, each with `center::NTuple{3,T}`, `Lx::T`, `Ly::T`, `Lz::T`
- `epses`: Permittivity for each box (length must match `boxes`)
- `eps_out`: Background (vacuum) permittivity, default `1.0`
- `alpha`: Aspect ratio control for coarse grid, default `sqrt(2)`

### Usage Example

```julia
boxes = [
    (center=(0.0, 0.0, 0.0), Lx=1.0, Ly=1.0, Lz=1.0),
    (center=(1.0, 0.0, 0.0), Lx=1.0, Ly=1.0, Lz=1.0),
]
epses = [2.0, 4.0]
interface = multi_dielectric_box3d(4, 0.2, boxes, epses)

# Works with all existing solvers unchanged
lhs = Lhs_dielectric_box3d(interface)
rhs = Rhs_dielectric_box3d(interface, ps, eps_src)
x = solve_gmres(lhs, rhs, 1e-12, 1e-12)
```

## Constraints

- Boxes must be axis-aligned (no rotation).
- Boxes must not overlap in volume; they may only touch face-to-face.
- Shared faces must be co-planar and axis-aligned.

## Architecture

### Face Detection Algorithm

For axis-aligned boxes, each box has 6 faces. Shared face detection:

1. For each pair of boxes (i, j) where i < j, check all 36 face-face combinations.
2. Two faces can share area only if:
   - They lie on the same plane (same axis direction, same coordinate value)
   - They have opposite normals (one faces +x, the other faces -x, etc.)
3. Compute the rectangular overlap of the two co-planar faces via 2D rectangle intersection in the shared plane.
4. If overlap area > 0, that rectangular region is a **shared internal interface**.

### Face Subtraction

For each box's original face:
- Collect all shared regions that overlap with this face.
- Subtract shared regions from the original face to produce **remaining external regions**.
- Each remaining region becomes an external (box-to-vacuum) interface.

The subtraction of axis-aligned rectangles from an axis-aligned rectangle produces a set of axis-aligned rectangles. This is implemented by sorting overlap intervals along one axis and splitting into strips.

### Panel Construction

For each face region (shared or external):
- Convert the rectangular region to corner points (a, b, c, d) with the appropriate outward normal.
- Use existing `rect_panel3d_adaptive_panels` to create panels with edge/corner refinement controlled by `l_ec`.
- Edge/corner flags are set based on whether a panel edge coincides with the boundary of its face region.

### Normal Convention

Consistent with the 2D multi-box implementation:
- **Shared face** between box i and box j (i < j): Normal points from box j toward box i.
- **External face** of box i: Normal points outward from the box into vacuum.

### Permittivity Assignment

- **Shared face** (normal from box j to box i): `eps_in = epses[j]`, `eps_out = epses[i]`
- **External face** of box i: `eps_in = epses[i]`, `eps_out = eps_out` (background)

### Internal Helper Functions

1. `_box3d_faces(center, Lx, Ly, Lz)`: Generate the 6 faces of a box as (corners, normal) tuples, offset by center.
2. `_detect_shared_faces_3d(boxes)`: For all box pairs, find co-planar face overlaps. Returns shared face regions with box indices.
3. `_rect_overlap_3d(face1, face2)`: Given two co-planar axis-aligned rectangles, compute their rectangular intersection (if any).
4. `_subtract_rects_from_rect(face, shared_rects)`: Subtract shared rectangles from a face, returning the remaining external rectangles.

## Output

A single `DielectricInterface{FlatPanel{T,3}, T}` containing all panels (shared + external) with correct `eps_in` and `eps_out` vectors. This is fully compatible with all existing solvers:
- `Lhs_dielectric_box3d` / `Lhs_dielectric_box3d_fmm3d`
- `Rhs_dielectric_box3d` / `Rhs_dielectric_box3d_fmm3d` / `Rhs_dielectric_box3d_hybrid`
- `solve_lu`, `solve_gmres`

## File Changes

1. **`src/shape/box3d.jl`**: Add `multi_dielectric_box3d` and internal helpers.
2. **`src/BoundaryIntegral.jl`**: Add `multi_dielectric_box3d` to exports (line 40).
3. **`test/`**: Add tests for multi-box 3D geometry (face detection, panel counts, permittivity assignment, solver integration).

## Testing Strategy

1. **Single box equivalence**: `multi_dielectric_box3d` with one box should produce the same interface as `single_dielectric_box3d`.
2. **Two touching boxes**: Verify shared face is detected, panel count is correct, eps_in/eps_out are assigned correctly.
3. **L-shaped arrangement**: Three boxes in an L shape — verify correct interface topology.
4. **Solver integration**: Run a full solve (LHS + RHS + GMRES) on a two-box system with a point source and verify the solution is physically reasonable.
