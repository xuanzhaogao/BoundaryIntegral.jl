# Multi-Dielectric Box 3D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `multi_dielectric_box3d` function that creates a `DielectricInterface` for multiple axis-aligned 3D dielectric boxes, automatically detecting shared faces and assigning permittivities.

**Architecture:** Single public function `multi_dielectric_box3d` in `src/shape/box3d.jl` with internal helpers for face generation, shared-face detection (co-planar axis-aligned rectangle overlap), and face subtraction. Returns a standard `DielectricInterface` compatible with all existing solvers.

**Tech Stack:** Julia, BoundaryIntegral.jl (FlatPanel, DielectricInterface, rect_panel3d_adaptive_panels, FastGaussQuadrature)

---

## File Structure

- **Modify:** `src/shape/box3d.jl` — add helper functions and `multi_dielectric_box3d`
- **Modify:** `src/BoundaryIntegral.jl` — add export for `multi_dielectric_box3d`
- **Create:** `test/shape/multi_box3d.jl` — tests for multi-box 3D
- **Modify:** `test/runtests.jl` — include the new test file

---

### Task 1: Internal helper — `_box3d_faces_at_center`

Generate 6 faces of a box centered at an arbitrary point.

**Files:**
- Modify: `src/shape/box3d.jl` (append after `_box3d_face_quads` around line 260)
- Create: `test/shape/multi_box3d.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the test file with a failing test**

Create `test/shape/multi_box3d.jl`:

```julia
using BoundaryIntegral
import BoundaryIntegral as BI
using LinearAlgebra
using Test

@testset "multi_box3d helpers" begin
    @testset "_box3d_faces_at_center" begin
        center = (1.0, 2.0, 3.0)
        Lx, Ly, Lz = 2.0, 4.0, 6.0
        faces = BI._box3d_faces_at_center(center, Lx, Ly, Lz)
        @test length(faces) == 6

        # Each face is (a, b, c, d, normal) where a,b,c,d are NTuple{3,Float64}
        # Check that face centers are offset by center
        for (a, b, c, d, normal) in faces
            face_center = (a .+ b .+ c .+ d) ./ 4
            # face center should differ from box center only along the normal axis
            diff = face_center .- center
            @test abs(dot(diff, normal)) > 0  # offset along normal
            # tangential components should be zero
            tangential = diff .- normal .* dot(diff, normal)
            @test norm(tangential) < 1e-14
        end

        # Check normals are unit vectors
        for (a, b, c, d, normal) in faces
            @test norm(normal) ≈ 1.0
        end

        # Check face areas: two faces of each size
        areas = Float64[]
        for (a, b, c, d, normal) in faces
            Lab = norm(b .- a)
            Lda = norm(a .- d)
            push!(areas, Lab * Lda)
        end
        sort!(areas)
        @test areas[1] ≈ areas[2] ≈ Lx * Ly  # z-faces
        @test areas[3] ≈ areas[4] ≈ Ly * Lz  # x-faces
        @test areas[5] ≈ areas[6] ≈ Lx * Lz  # y-faces
    end
end
```

- [ ] **Step 2: Add test file to runtests.jl**

In `test/runtests.jl`, add the following line inside the `@testset "BoundaryIntegral.jl"` block, after the existing `include("shape/box3d.jl")` block (after line 31), but outside the `if run_full` guard so it runs in the basic test suite:

```julia
    # multi-box 3d
    include("shape/multi_box3d.jl")
```

Add this line after the `end` on line 31 (closing the first `if run_full` block) and before the second `if run_full` block on line 34.

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'using Pkg; Pkg.test()'`
Expected: FAIL — `_box3d_faces_at_center` not defined

- [ ] **Step 4: Implement `_box3d_faces_at_center`**

In `src/shape/box3d.jl`, add after the `_box3d_face_quads` function (after line 260):

```julia
# Generate 6 faces of an axis-aligned box centered at `center` with dimensions Lx x Ly x Lz.
# Each face is returned as (a, b, c, d, normal) where a,b,c,d are corners in anti-clockwise order
# and normal points outward.
function _box3d_faces_at_center(center::NTuple{3, T}, Lx::T, Ly::T, Lz::T) where T
    cx, cy, cz = center
    hx, hy, hz = Lx / 2, Ly / 2, Lz / 2
    t0 = zero(T)
    t1 = one(T)

    # vertices (same ordering as _box3d_geometry, but offset by center)
    v1 = (cx + hx, cy + hy, cz + hz)
    v2 = (cx - hx, cy + hy, cz + hz)
    v3 = (cx - hx, cy - hy, cz + hz)
    v4 = (cx + hx, cy - hy, cz + hz)
    v5 = (cx + hx, cy + hy, cz - hz)
    v6 = (cx - hx, cy + hy, cz - hz)
    v7 = (cx - hx, cy - hy, cz - hz)
    v8 = (cx + hx, cy - hy, cz - hz)

    faces = Tuple{NTuple{3,T}, NTuple{3,T}, NTuple{3,T}, NTuple{3,T}, NTuple{3,T}}[
        (v1, v2, v3, v4, ( t0,  t0,  t1)),  # z = +hz (top)
        (v5, v8, v7, v6, ( t0,  t0, -t1)),  # z = -hz (bottom)
        (v8, v5, v1, v4, ( t1,  t0,  t0)),  # x = +hx (right)
        (v7, v3, v2, v6, (-t1,  t0,  t0)),  # x = -hx (left)
        (v6, v2, v1, v5, ( t0,  t1,  t0)),  # y = +hy (front)
        (v7, v8, v4, v3, ( t0, -t1,  t0)),  # y = -hy (back)
    ]

    return faces
end
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'using Pkg; Pkg.test()'`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/shape/box3d.jl test/shape/multi_box3d.jl test/runtests.jl
git commit -m "feat: add _box3d_faces_at_center helper for multi-box 3D"
```

---

### Task 2: Internal helper — `_rect_overlap_3d`

Compute the rectangular intersection of two co-planar axis-aligned rectangles in 3D.

**Files:**
- Modify: `src/shape/box3d.jl`
- Modify: `test/shape/multi_box3d.jl`

- [ ] **Step 1: Write failing test**

Append to `test/shape/multi_box3d.jl`:

```julia
@testset "_rect_overlap_3d" begin
    # Two faces sharing a full face on x=0.5 plane
    # Box1 at origin, Box2 at (1,0,0), both unit cubes
    # Box1 has face at x=+0.5, Box2 has face at x=-0.5
    face1_a = (0.5, -0.5, -0.5)
    face1_b = (0.5,  0.5, -0.5)
    face1_c = (0.5,  0.5,  0.5)
    face1_d = (0.5, -0.5,  0.5)
    face1_n = (1.0, 0.0, 0.0)

    face2_a = (0.5, -0.5, -0.5)
    face2_b = (0.5,  0.5, -0.5)
    face2_c = (0.5,  0.5,  0.5)
    face2_d = (0.5, -0.5,  0.5)
    face2_n = (-1.0, 0.0, 0.0)

    has_overlap, region = BI._rect_overlap_3d(
        face1_a, face1_b, face1_c, face1_d, face1_n,
        face2_a, face2_b, face2_c, face2_d, face2_n
    )
    @test has_overlap
    # The overlap region should be the full face
    oa, ob, oc, od = region
    Lab = norm(ob .- oa)
    Lda = norm(oa .- od)
    @test Lab * Lda ≈ 1.0  # 1x1 face

    # Non-coplanar faces should not overlap
    face3_a = (0.0, -0.5, 0.5)
    face3_b = (0.0,  0.5, 0.5)
    face3_c = (1.0,  0.5, 0.5)
    face3_d = (1.0, -0.5, 0.5)
    face3_n = (0.0, 0.0, 1.0)

    has_overlap2, _ = BI._rect_overlap_3d(
        face1_a, face1_b, face1_c, face1_d, face1_n,
        face3_a, face3_b, face3_c, face3_d, face3_n
    )
    @test !has_overlap2

    # Partial overlap: box1 unit cube at origin, box2 unit cube at (0, 0.5, 0)
    # They share part of the y=+0.5 face of box1 and y=-0.5 face of box2 is at y=0.0
    # Actually box2's y=-0.5 face is at y=0.5-0.5=0.0, box1's y=+0.5 face is at y=0.5
    # These are NOT co-planar, so no overlap. Let me fix:
    # box1 at origin, box2 at (0, 1.0, 0) - they touch at y=0.5
    face_b1_yp_a = (0.5, 0.5, -0.5)  # box1 y=+0.5 face  (but this face has normal (0,1,0))
    face_b1_yp_b = (-0.5, 0.5, -0.5)
    face_b1_yp_c = (-0.5, 0.5, 0.5)
    face_b1_yp_d = (0.5, 0.5, 0.5)
    face_b1_yp_n = (0.0, 1.0, 0.0)

    face_b2_ym_a = (-0.5, 0.5, -0.5)  # box2 at (0,1,0) y=-0.5 face at y=0.5
    face_b2_ym_b = (0.5, 0.5, -0.5)
    face_b2_ym_c = (0.5, 0.5, 0.5)
    face_b2_ym_d = (-0.5, 0.5, 0.5)
    face_b2_ym_n = (0.0, -1.0, 0.0)

    has_overlap3, region3 = BI._rect_overlap_3d(
        face_b1_yp_a, face_b1_yp_b, face_b1_yp_c, face_b1_yp_d, face_b1_yp_n,
        face_b2_ym_a, face_b2_ym_b, face_b2_ym_c, face_b2_ym_d, face_b2_ym_n
    )
    @test has_overlap3
    oa3, ob3, oc3, od3 = region3
    @test norm(ob3 .- oa3) * norm(oa3 .- od3) ≈ 1.0
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'using Pkg; Pkg.test()'`
Expected: FAIL — `_rect_overlap_3d` not defined

- [ ] **Step 3: Implement `_rect_overlap_3d`**

In `src/shape/box3d.jl`, add after `_box3d_faces_at_center`:

```julia
# Check if two axis-aligned 3D rectangular faces overlap.
# Requirements: both faces must be co-planar (same plane) and have opposite normals.
# Returns (has_overlap::Bool, region) where region is (a, b, c, d) corners of the overlap rectangle.
# The overlap rectangle's normal is taken from face1.
function _rect_overlap_3d(
    a1::NTuple{3,T}, b1::NTuple{3,T}, c1::NTuple{3,T}, d1::NTuple{3,T}, n1::NTuple{3,T},
    a2::NTuple{3,T}, b2::NTuple{3,T}, c2::NTuple{3,T}, d2::NTuple{3,T}, n2::NTuple{3,T};
    tol::T = T(1e-10)
) where T
    # Check opposite normals
    if norm(n1 .+ n2) > tol
        return false, nothing
    end

    # Check co-planarity: all corners of face2 must lie on the plane of face1
    # plane equation: dot(n1, p - a1) = 0
    for p in (a2, b2, c2, d2)
        if abs(dot(n1, p .- a1)) > tol
            return false, nothing
        end
    end

    # Project onto 2D: find the two tangential axes
    # For axis-aligned faces, the normal is along one axis
    abs_n = (abs(n1[1]), abs(n1[2]), abs(n1[3]))
    if abs_n[1] > 0.5  # normal along x
        ax1, ax2 = 2, 3  # project onto y-z plane
    elseif abs_n[2] > 0.5  # normal along y
        ax1, ax2 = 1, 3  # project onto x-z plane
    else  # normal along z
        ax1, ax2 = 1, 2  # project onto x-y plane
    end

    # Get bounding intervals of each face in the 2D projection
    corners1 = (a1, b1, c1, d1)
    corners2 = (a2, b2, c2, d2)

    min1_u = minimum(c[ax1] for c in corners1)
    max1_u = maximum(c[ax1] for c in corners1)
    min1_v = minimum(c[ax2] for c in corners1)
    max1_v = maximum(c[ax2] for c in corners1)

    min2_u = minimum(c[ax1] for c in corners2)
    max2_u = maximum(c[ax1] for c in corners2)
    min2_v = minimum(c[ax2] for c in corners2)
    max2_v = maximum(c[ax2] for c in corners2)

    # Compute overlap interval
    lo_u = max(min1_u, min2_u)
    hi_u = min(max1_u, max2_u)
    lo_v = max(min1_v, min2_v)
    hi_v = min(max1_v, max2_v)

    if hi_u - lo_u < tol || hi_v - lo_v < tol
        return false, nothing
    end

    # Reconstruct 3D corners of overlap rectangle
    # The coordinate along the normal axis is the same for all points on the plane
    plane_coord = a1[findfirst(x -> x > 0.5, abs_n)]

    function make_point(u::T, v::T)
        p = zeros(T, 3)
        p[ax1] = u
        p[ax2] = v
        p[findfirst(x -> x > 0.5, abs_n)] = plane_coord
        return NTuple{3,T}(Tuple(p))
    end

    # Corners in anti-clockwise order when viewed from the direction of n1
    # We need to figure out the winding. Use the same convention as face1.
    # face1 goes a1 -> b1 -> c1 -> d1 anti-clockwise.
    # edge a1->b1 direction
    e_ab = (b1[ax1] - a1[ax1], b1[ax2] - a1[ax2])
    # edge a1->d1 direction
    e_ad = (d1[ax1] - a1[ax1], d1[ax2] - a1[ax2])

    # Determine corner ordering based on face1's winding
    # a is the corner with the minimum values in the directions of a->b and a->d from face1
    u_vals = [lo_u, hi_u]
    v_vals = [lo_v, hi_v]

    # Choose a corner: the one closest to a1 in the projected plane
    u_start = (e_ab[1] >= 0) ? lo_u : hi_u
    u_end   = (e_ab[1] >= 0) ? hi_u : lo_u
    v_start = (e_ad[2] >= 0) ? lo_v : hi_v
    v_end   = (e_ad[2] >= 0) ? hi_v : lo_v

    # But actually for axis-aligned rects, let's just pick a consistent winding
    # matching the face1 winding direction
    if abs(e_ab[1]) > tol  # a->b is along ax1
        oa = make_point(u_start, v_start)
        ob = make_point(u_end, v_start)
        oc = make_point(u_end, v_end)
        od = make_point(u_start, v_end)
    else  # a->b is along ax2
        oa = make_point(u_start, v_start)
        ob = make_point(u_start, v_end)
        oc = make_point(u_end, v_end)
        od = make_point(u_end, v_start)
    end

    return true, (oa, ob, oc, od)
end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'using Pkg; Pkg.test()'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/shape/box3d.jl test/shape/multi_box3d.jl
git commit -m "feat: add _rect_overlap_3d helper for co-planar face intersection"
```

---

### Task 3: Internal helper — `_detect_shared_faces_3d`

For all pairs of boxes, find co-planar face overlaps.

**Files:**
- Modify: `src/shape/box3d.jl`
- Modify: `test/shape/multi_box3d.jl`

- [ ] **Step 1: Write failing test**

Append to `test/shape/multi_box3d.jl`:

```julia
@testset "_detect_shared_faces_3d" begin
    # Two unit cubes touching at x=0.5
    boxes = [
        (center=(0.0, 0.0, 0.0), Lx=1.0, Ly=1.0, Lz=1.0),
        (center=(1.0, 0.0, 0.0), Lx=1.0, Ly=1.0, Lz=1.0),
    ]
    shared = BI._detect_shared_faces_3d(boxes)
    @test length(shared) == 1  # one shared face

    # Each entry is (region, id_lo, id_hi, normal_from_hi_to_lo)
    region, id1, id2, normal = shared[1]
    @test id1 < id2
    a, b, c, d = region
    area = norm(b .- a) * norm(a .- d)
    @test area ≈ 1.0  # full 1x1 face

    # Three boxes in a line
    boxes3 = [
        (center=(0.0, 0.0, 0.0), Lx=1.0, Ly=1.0, Lz=1.0),
        (center=(1.0, 0.0, 0.0), Lx=1.0, Ly=1.0, Lz=1.0),
        (center=(2.0, 0.0, 0.0), Lx=1.0, Ly=1.0, Lz=1.0),
    ]
    shared3 = BI._detect_shared_faces_3d(boxes3)
    @test length(shared3) == 2  # box1-box2, box2-box3

    # Non-touching boxes
    boxes_far = [
        (center=(0.0, 0.0, 0.0), Lx=1.0, Ly=1.0, Lz=1.0),
        (center=(5.0, 0.0, 0.0), Lx=1.0, Ly=1.0, Lz=1.0),
    ]
    shared_far = BI._detect_shared_faces_3d(boxes_far)
    @test length(shared_far) == 0
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'using Pkg; Pkg.test()'`
Expected: FAIL — `_detect_shared_faces_3d` not defined

- [ ] **Step 3: Implement `_detect_shared_faces_3d`**

In `src/shape/box3d.jl`, add after `_rect_overlap_3d`:

```julia
# Detect all shared faces between pairs of axis-aligned boxes.
# Returns a vector of (region, id_lo, id_hi, normal) where:
#   region = (a, b, c, d) corners of the shared rectangle
#   id_lo < id_hi are box indices
#   normal points from box id_hi toward box id_lo
function _detect_shared_faces_3d(boxes::Vector{<:NamedTuple}) where T
    T_val = typeof(boxes[1].Lx)
    shared = Tuple{NTuple{4, NTuple{3, T_val}}, Int, Int, NTuple{3, T_val}}[]

    n_boxes = length(boxes)
    for i in 1:n_boxes
        faces_i = _box3d_faces_at_center(boxes[i].center, boxes[i].Lx, boxes[i].Ly, boxes[i].Lz)
        for j in (i+1):n_boxes
            faces_j = _box3d_faces_at_center(boxes[j].center, boxes[j].Lx, boxes[j].Ly, boxes[j].Lz)
            for (a1, b1, c1, d1, n1) in faces_i
                for (a2, b2, c2, d2, n2) in faces_j
                    has_overlap, region = _rect_overlap_3d(a1, b1, c1, d1, n1, a2, b2, c2, d2, n2)
                    if has_overlap
                        # normal points from higher-id box (j) to lower-id box (i)
                        # n1 is the outward normal of box i's face, which points toward box j
                        # So the normal from j to i is -n1... but actually n1 points outward from box i.
                        # Convention: normal from box j toward box i = n2 (outward of j, but n2 = -n1)
                        # Actually: n1 points outward from box i (toward j), n2 = -n1 points outward from j (toward i)
                        # We want normal from j to i, which is -n1 = n2's direction but normalized.
                        # Since n2 ≈ -n1, the normal from j→i is -n1.
                        push!(shared, (region, i, j, (.-n1)))
                    end
                end
            end
        end
    end

    return shared
end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'using Pkg; Pkg.test()'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/shape/box3d.jl test/shape/multi_box3d.jl
git commit -m "feat: add _detect_shared_faces_3d for multi-box face detection"
```

---

### Task 4: Internal helper — `_subtract_rects_from_face_3d`

Subtract shared rectangles from a face to get the remaining external regions.

**Files:**
- Modify: `src/shape/box3d.jl`
- Modify: `test/shape/multi_box3d.jl`

- [ ] **Step 1: Write failing test**

Append to `test/shape/multi_box3d.jl`:

```julia
@testset "_subtract_rects_from_face_3d" begin
    # Face on x=0.5 plane, 1x1 face from y=-0.5..0.5, z=-0.5..0.5
    face_a = (0.5, -0.5, -0.5)
    face_b = (0.5,  0.5, -0.5)
    face_c = (0.5,  0.5,  0.5)
    face_d = (0.5, -0.5,  0.5)
    face_n = (1.0, 0.0, 0.0)

    # No shared regions: should return the whole face
    remaining = BI._subtract_rects_from_face_3d(face_a, face_b, face_c, face_d, face_n, NTuple{4, NTuple{3, Float64}}[])
    @test length(remaining) == 1
    total_area = sum(norm(r[2] .- r[1]) * norm(r[1] .- r[4]) for r in remaining)
    @test total_area ≈ 1.0

    # Full face shared: should return empty
    shared_full = [(face_a, face_b, face_c, face_d)]
    remaining_full = BI._subtract_rects_from_face_3d(face_a, face_b, face_c, face_d, face_n, shared_full)
    @test isempty(remaining_full)

    # Half face shared: bottom half z=-0.5..0.0
    shared_half = [((0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.0), (0.5, -0.5, 0.0))]
    remaining_half = BI._subtract_rects_from_face_3d(face_a, face_b, face_c, face_d, face_n, shared_half)
    total_area_remaining = sum(norm(r[2] .- r[1]) * norm(r[1] .- r[4]) for r in remaining_half)
    @test total_area_remaining ≈ 0.5
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'using Pkg; Pkg.test()'`
Expected: FAIL — `_subtract_rects_from_face_3d` not defined

- [ ] **Step 3: Implement `_subtract_rects_from_face_3d`**

In `src/shape/box3d.jl`, add after `_detect_shared_faces_3d`:

```julia
# Subtract a set of axis-aligned rectangles from an axis-aligned face in 3D.
# All rectangles must lie on the same plane as the face.
# Returns a vector of (a, b, c, d) remaining rectangular regions with normal = face normal.
function _subtract_rects_from_face_3d(
    a::NTuple{3,T}, b::NTuple{3,T}, c::NTuple{3,T}, d::NTuple{3,T}, normal::NTuple{3,T},
    shared::Vector{<:NTuple{4, NTuple{3,T}}};
    tol::T = T(1e-10)
) where T
    if isempty(shared)
        return [(a, b, c, d)]
    end

    abs_n = (abs(normal[1]), abs(normal[2]), abs(normal[3]))
    if abs_n[1] > 0.5
        ax1, ax2 = 2, 3
    elseif abs_n[2] > 0.5
        ax1, ax2 = 1, 3
    else
        ax1, ax2 = 1, 2
    end
    normal_ax = findfirst(x -> x > 0.5, abs_n)
    plane_coord = a[normal_ax]

    # Face bounding box in 2D
    corners_face = (a, b, c, d)
    face_u_min = minimum(p[ax1] for p in corners_face)
    face_u_max = maximum(p[ax1] for p in corners_face)
    face_v_min = minimum(p[ax2] for p in corners_face)
    face_v_max = maximum(p[ax2] for p in corners_face)

    # Collect all u and v coordinates from face and shared regions
    u_coords = T[face_u_min, face_u_max]
    v_coords = T[face_v_min, face_v_max]
    for rect in shared
        for corner in rect
            push!(u_coords, corner[ax1])
            push!(v_coords, corner[ax2])
        end
    end
    sort!(unique!(u_coords))
    sort!(unique!(v_coords))

    # Filter to within face bounds
    filter!(u -> face_u_min - tol <= u <= face_u_max + tol, u_coords)
    filter!(v -> face_v_min - tol <= v <= face_v_max + tol, v_coords)

    function make_point_3d(u::T, v::T)
        p = zeros(T, 3)
        p[ax1] = u
        p[ax2] = v
        p[normal_ax] = plane_coord
        return NTuple{3,T}(Tuple(p))
    end

    # Check if a 2D cell center is inside any shared rectangle
    function is_shared(u_mid::T, v_mid::T)
        for rect in shared
            rect_u_min = minimum(p[ax1] for p in rect)
            rect_u_max = maximum(p[ax1] for p in rect)
            rect_v_min = minimum(p[ax2] for p in rect)
            rect_v_max = maximum(p[ax2] for p in rect)
            if rect_u_min - tol <= u_mid <= rect_u_max + tol &&
               rect_v_min - tol <= v_mid <= rect_v_max + tol
                return true
            end
        end
        return false
    end

    remaining = NTuple{4, NTuple{3,T}}[]

    for i in 1:(length(u_coords) - 1)
        for j in 1:(length(v_coords) - 1)
            u_lo, u_hi = u_coords[i], u_coords[i + 1]
            v_lo, v_hi = v_coords[j], v_coords[j + 1]

            if u_hi - u_lo < tol || v_hi - v_lo < tol
                continue
            end

            u_mid = (u_lo + u_hi) / 2
            v_mid = (v_lo + v_hi) / 2

            if !is_shared(u_mid, v_mid)
                # Build corners with same winding as original face
                ra = make_point_3d(u_lo, v_lo)
                rb = make_point_3d(u_hi, v_lo)
                rc = make_point_3d(u_hi, v_hi)
                rd = make_point_3d(u_lo, v_hi)
                push!(remaining, (ra, rb, rc, rd))
            end
        end
    end

    return remaining
end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'using Pkg; Pkg.test()'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/shape/box3d.jl test/shape/multi_box3d.jl
git commit -m "feat: add _subtract_rects_from_face_3d for face region subtraction"
```

---

### Task 5: Main function — `multi_dielectric_box3d`

Compose the helpers into the public API function.

**Files:**
- Modify: `src/shape/box3d.jl`
- Modify: `src/BoundaryIntegral.jl` (line 40, add export)
- Modify: `test/shape/multi_box3d.jl`

- [ ] **Step 1: Write failing test**

Append to `test/shape/multi_box3d.jl`:

```julia
@testset "multi_dielectric_box3d" begin
    @testset "single box equivalence" begin
        Lx, Ly, Lz = 1.0, 1.0, 1.0
        n_quad = 4
        l_ec = 0.3
        eps_in = 2.0
        eps_out = 1.0

        single = BI.single_dielectric_box3d(Lx, Ly, Lz, n_quad, l_ec, eps_in, eps_out, Float64)

        boxes = [(center=(0.0, 0.0, 0.0), Lx=Lx, Ly=Ly, Lz=Lz)]
        epses = [eps_in]
        multi = BI.multi_dielectric_box3d(n_quad, l_ec, boxes, epses, eps_out)

        @test length(multi.panels) == length(single.panels)
        @test BI.num_points(multi) == BI.num_points(single)
        @test all(multi.eps_in .== eps_in)
        @test all(multi.eps_out .== eps_out)
    end

    @testset "two touching boxes" begin
        boxes = [
            (center=(0.0, 0.0, 0.0), Lx=1.0, Ly=1.0, Lz=1.0),
            (center=(1.0, 0.0, 0.0), Lx=1.0, Ly=1.0, Lz=1.0),
        ]
        epses = [2.0, 4.0]
        eps_out = 1.0
        interface = BI.multi_dielectric_box3d(4, 0.3, boxes, epses, eps_out)

        @test length(interface.panels) > 0
        @test BI.num_points(interface) > 0

        # Check that eps values are correct
        unique_eps_pairs = Set{Tuple{Float64, Float64}}()
        for i in 1:length(interface.panels)
            push!(unique_eps_pairs, (interface.eps_in[i], interface.eps_out[i]))
        end
        # Should have: (2.0, 1.0) for box1-vacuum, (4.0, 1.0) for box2-vacuum,
        # and (4.0, 2.0) for the shared face (normal from box2 to box1)
        @test (2.0, 1.0) in unique_eps_pairs  # box1 external
        @test (4.0, 1.0) in unique_eps_pairs  # box2 external
        @test (4.0, 2.0) in unique_eps_pairs  # shared face

        # Total surface area check:
        # Two unit cubes touching: total exposed area = 2*6 - 2*1 = 10 external + 1 shared = 11 face-areas
        total_weight = sum(BI.all_weights(interface))
        @test total_weight ≈ 11.0 atol=0.01
    end

    @testset "three boxes L-shape" begin
        boxes = [
            (center=(0.0, 0.0, 0.0), Lx=1.0, Ly=1.0, Lz=1.0),
            (center=(1.0, 0.0, 0.0), Lx=1.0, Ly=1.0, Lz=1.0),
            (center=(0.0, 1.0, 0.0), Lx=1.0, Ly=1.0, Lz=1.0),
        ]
        epses = [2.0, 3.0, 4.0]
        interface = BI.multi_dielectric_box3d(4, 0.3, boxes, epses)

        @test length(interface.panels) > 0

        # 3 cubes, 2 shared faces, total weight = 3*6 - 2*2 + 2 = 16
        # Actually: 3 cubes have 18 face-areas total. 2 shared faces remove 2 external face-areas
        # and add 2 shared face-areas (which are the same area).
        # Total area integrated = 18 - 2*2 + 2 = 16... let me recalculate:
        # Each shared face replaces 2 external faces (one from each box) with 1 shared face.
        # So total face-areas = 18 - 2*1 = 16 (each shared face removes 1 net face since
        # 2 external become 1 shared). Wait:
        # - 3 boxes * 6 faces = 18 face-units
        # - 2 shared faces: each shared face means 2 external faces are replaced by 1 shared face
        # But the shared face is still a face with panels, so total face count = 18 - 2 = 16
        # Total area = 16 * 1.0 = 16.0
        total_weight = sum(BI.all_weights(interface))
        @test total_weight ≈ 16.0 atol=0.01
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'using Pkg; Pkg.test()'`
Expected: FAIL — `multi_dielectric_box3d` not defined

- [ ] **Step 3: Add export to BoundaryIntegral.jl**

In `src/BoundaryIntegral.jl`, change line 40 from:
```julia
export single_dielectric_box3d
```
to:
```julia
export single_dielectric_box3d, multi_dielectric_box3d
```

- [ ] **Step 4: Implement `multi_dielectric_box3d`**

In `src/shape/box3d.jl`, add after `_subtract_rects_from_face_3d`:

```julia
function multi_dielectric_box3d(
    n_quad::Int, l_ec::T,
    boxes::Vector{<:NamedTuple},
    epses::Vector{T},
    eps_out::T = one(T);
    alpha::T = T(sqrt(T(2)))
) where T
    @assert length(boxes) == length(epses) "Number of boxes must match number of permittivities"
    @assert length(boxes) >= 1 "At least one box is required"

    ns, ws = gausslegendre(n_quad)
    ns = T.(ns)
    ws = T.(ws)

    n_boxes = length(boxes)

    # Step 1: Detect all shared faces
    shared_faces = _detect_shared_faces_3d(boxes)

    # Build lookup: for each box and face, which shared regions apply
    # shared_faces entries: (region, id_lo, id_hi, normal)

    # Step 2: For each box, generate faces, subtract shared regions, build panels
    panels_vec = Vector{FlatPanel{T, 3}}()
    eps_in_vec = Vector{T}()
    eps_out_vec = Vector{T}()

    for box_id in 1:n_boxes
        box = boxes[box_id]
        faces = _box3d_faces_at_center(box.center, box.Lx, box.Ly, box.Lz)

        for (fa, fb, fc, fd, fn) in faces
            # Find shared regions that overlap with this face
            face_shared = NTuple{4, NTuple{3, T}}[]
            for (region, id_lo, id_hi, sn) in shared_faces
                if box_id == id_lo || box_id == id_hi
                    # Check if this shared region lies on this face
                    has_ov, ov_region = _rect_overlap_3d(fa, fb, fc, fd, fn,
                        region[1], region[2], region[3], region[4], (.-fn))
                    if has_ov
                        push!(face_shared, ov_region)
                    end
                end
            end

            # Get remaining external regions
            remaining = _subtract_rects_from_face_3d(fa, fb, fc, fd, fn, face_shared)

            # Build panels for remaining external regions
            for (ra, rb, rc, rd) in remaining
                is_edge = (true, true, true, true)
                is_corner = (true, true, true, true)
                new_panels = rect_panel3d_adaptive_panels(ra, rb, rc, rd, ns, ws, fn, is_edge, is_corner, alpha, l_ec)
                append!(panels_vec, new_panels)
                append!(eps_in_vec, fill(epses[box_id], length(new_panels)))
                append!(eps_out_vec, fill(eps_out, length(new_panels)))
            end
        end
    end

    # Step 3: Build panels for shared faces
    for (region, id_lo, id_hi, normal) in shared_faces
        a, b, c, d = region
        is_edge = (true, true, true, true)
        is_corner = (true, true, true, true)
        new_panels = rect_panel3d_adaptive_panels(a, b, c, d, ns, ws, normal, is_edge, is_corner, alpha, l_ec)
        append!(panels_vec, new_panels)
        # Convention (matching 2D): normal points from id_hi → id_lo
        # eps_in = medium the normal comes FROM = epses[id_hi]
        # eps_out = medium the normal points TO = epses[id_lo]
        append!(eps_in_vec, fill(epses[id_hi], length(new_panels)))
        append!(eps_out_vec, fill(epses[id_lo], length(new_panels)))
    end

    return DielectricInterface(panels_vec, eps_in_vec, eps_out_vec)
end
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'using Pkg; Pkg.test()'`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/shape/box3d.jl src/BoundaryIntegral.jl test/shape/multi_box3d.jl
git commit -m "feat: add multi_dielectric_box3d for multi-box 3D dielectric systems"
```

---

### Task 6: Solver integration test

Verify the multi-box interface works with existing solvers.

**Files:**
- Modify: `test/shape/multi_box3d.jl`

- [ ] **Step 1: Write solver integration test**

Append to `test/shape/multi_box3d.jl`:

```julia
@testset "multi_dielectric_box3d solver integration" begin
    boxes = [
        (center=(0.0, 0.0, 0.0), Lx=1.0, Ly=1.0, Lz=1.0),
        (center=(1.0, 0.0, 0.0), Lx=1.0, Ly=1.0, Lz=1.0),
    ]
    epses = [2.0, 4.0]
    eps_out = 1.0
    interface = BI.multi_dielectric_box3d(4, 0.3, boxes, epses, eps_out)

    # Point source outside both boxes
    ps = BI.PointSource((3.0, 0.0, 0.0), 1.0)
    eps_src = eps_out

    # Build LHS and RHS
    lhs = BI.Lhs_dielectric_box3d(interface)
    rhs = BI.Rhs_dielectric_box3d(interface, ps, eps_src)

    @test size(lhs, 1) == BI.num_points(interface)
    @test size(lhs, 2) == BI.num_points(interface)
    @test length(rhs) == BI.num_points(interface)

    # Solve
    sigma = lhs \ rhs
    @test length(sigma) == BI.num_points(interface)
    @test all(isfinite.(sigma))
end
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cd /mnt/home/xgao1/codes/BoundaryIntegral.jl && julia --project -e 'using Pkg; Pkg.test()'`
Expected: PASS (solvers work on any `DielectricInterface`)

- [ ] **Step 3: Commit**

```bash
git add test/shape/multi_box3d.jl
git commit -m "test: add solver integration test for multi_dielectric_box3d"
```
