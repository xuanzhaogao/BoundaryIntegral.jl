# Fix corner flags in multi-box face subtraction

**Goal:** Correctly identify physical corner points in `_subtract_rects_from_face_3d` so that panels at shared-region corners get proper refinement.

**Root cause:** Current code computes `corner = edge1 && edge2` (both adjacent edges must be physical). But a corner point can be physical even when neither adjacent edge is — e.g., when the diagonal cell is the shared region. The correct check is: a grid point is a physical corner if any of the (up to 4) cells meeting at that point is shared or out-of-bounds.

**File:** `src/shape/box3d_multi.jl` — modify `_subtract_rects_from_face_3d`

---

## Implementation

Inside `_subtract_rects_from_face_3d`, after building the `cell_shared` matrix and before the loop that builds `remaining`, add a grid-point classification:

### Step 1: Build physical corner grid

Add the following code after the `cell_shared` matrix is built:

```julia
# Classify grid points as physical corners.
# Grid point (gi, gj) is where cells (gi-1,gj-1), (gi,gj-1), (gi-1,gj), (gi,gj) meet.
# A grid point is physical if any surrounding cell is shared or out-of-bounds.
# (If all 4 surrounding cells are remaining, the surface is smooth there.)
physical_corner = Matrix{Bool}(undef, nu + 1, nv + 1)
for gi in 1:(nu + 1)
    for gj in 1:(nv + 1)
        has_nonremaining = false
        for (di, dj) in ((0, 0), (-1, 0), (0, -1), (-1, -1))
            ci, cj = gi + di, gj + dj
            if ci < 1 || ci > nu || cj < 1 || cj > nv
                has_nonremaining = true  # out-of-bounds = face boundary
                break
            elseif cell_shared[ci, cj]
                has_nonremaining = true  # shared cell
                break
            end
        end
        physical_corner[gi, gj] = has_nonremaining
    end
end
```

### Step 2: Use the grid for corner classification

Replace the current corner computation:
```julia
corner_a = edge_da && edge_ab
corner_b = edge_ab && edge_bc
corner_c = edge_bc && edge_cd
corner_d = edge_cd && edge_da
```

With:
```julia
# Corner is physical if the grid point at that corner is physical.
# Cell (i,j) has corners at grid points: a=(i,j), b=(i+1,j), c=(i+1,j+1), d=(i,j+1)
corner_a = physical_corner[i, j]
corner_b = physical_corner[i + 1, j]
corner_c = physical_corner[i + 1, j + 1]
corner_d = physical_corner[i, j + 1]
```

### Step 3: Update tests

In `test/shape/multi_box3d.jl`, update the center-hole test to verify corner behavior:

For the center-hole case (1x1 shared in center of 2x2 face), the shared region's 4 corners each touch one remaining cell diagonally. All corner points of all 8 remaining cells should be physical (each touches either a face boundary or the shared region).

Also add a test with a shared region on the edge of the face (not centered) to verify that interior grid points not touching any shared region get `corner=false`.

### Step 4: Run tests and commit

Run: `julia --project -e 'using Pkg; Pkg.test()'`
Expected: all tests pass.

Commit: `git add -A && git commit -m "fix: use grid-point adjacency for corner classification in face subtraction"`

---

## Verification

After the fix, for the 2x2x2 + 1x1x1 case:
- The shared region's 4 corner points should be marked as physical corners in all remaining sub-rectangles that touch them
- Face boundary corner points remain physical
- Interior grid points where all 4 cells are remaining should NOT be physical corners
