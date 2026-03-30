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

    faces = Tuple{NTuple{3, T}, NTuple{3, T}, NTuple{3, T}, NTuple{3, T}, NTuple{3, T}}[
        (v1, v2, v3, v4, ( t0,  t0,  t1)),  # z = +hz (top)
        (v5, v8, v7, v6, ( t0,  t0, -t1)),  # z = -hz (bottom)
        (v8, v5, v1, v4, ( t1,  t0,  t0)),  # x = +hx (right)
        (v7, v3, v2, v6, (-t1,  t0,  t0)),  # x = -hx (left)
        (v6, v2, v1, v5, ( t0,  t1,  t0)),  # y = +hy (front)
        (v7, v8, v4, v3, ( t0, -t1,  t0)),  # y = -hy (back)
    ]

    return faces
end

# Check if two axis-aligned 3D rectangular faces overlap.
# Requirements: both faces must be co-planar (same plane) and have opposite normals.
# Returns (has_overlap::Bool, region) where region is (a, b, c, d) corners of the overlap rectangle.
# The overlap rectangle's normal is taken from face1.
function _rect_overlap_3d(
    a1::NTuple{3, T}, b1::NTuple{3, T}, c1::NTuple{3, T}, d1::NTuple{3, T}, n1::NTuple{3, T},
    a2::NTuple{3, T}, b2::NTuple{3, T}, c2::NTuple{3, T}, d2::NTuple{3, T}, n2::NTuple{3, T};
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
    normal_ax = findfirst(x -> x > 0.5, abs_n)
    plane_coord = a1[normal_ax]

    function make_point(u::T, v::T)
        p = zeros(T, 3)
        p[ax1] = u
        p[ax2] = v
        p[normal_ax] = plane_coord
        return NTuple{3, T}(Tuple(p))
    end

    # Corners in anti-clockwise order when viewed from the direction of n1
    # We need to figure out the winding. Use the same convention as face1.
    # face1 goes a1 -> b1 -> c1 -> d1 anti-clockwise.
    # edge a1->b1 direction
    e_ab = (b1[ax1] - a1[ax1], b1[ax2] - a1[ax2])
    # edge a1->d1 direction
    e_ad = (d1[ax1] - a1[ax1], d1[ax2] - a1[ax2])

    # Determine corner ordering based on face1's winding
    if abs(e_ab[1]) > tol  # a->b is along ax1
        u_start = (e_ab[1] >= 0) ? lo_u : hi_u
        u_end   = (e_ab[1] >= 0) ? hi_u : lo_u
        v_start = (e_ad[2] >= 0) ? lo_v : hi_v
        v_end   = (e_ad[2] >= 0) ? hi_v : lo_v
        oa = make_point(u_start, v_start)
        ob = make_point(u_end, v_start)
        oc = make_point(u_end, v_end)
        od = make_point(u_start, v_end)
    else  # a->b is along ax2
        u_start = (e_ad[1] >= 0) ? lo_u : hi_u
        u_end   = (e_ad[1] >= 0) ? hi_u : lo_u
        v_start = (e_ab[2] >= 0) ? lo_v : hi_v
        v_end   = (e_ab[2] >= 0) ? hi_v : lo_v
        oa = make_point(u_start, v_start)
        ob = make_point(u_start, v_end)
        oc = make_point(u_end, v_end)
        od = make_point(u_end, v_start)
    end

    return true, (oa, ob, oc, od)
end

# Detect all shared faces between pairs of axis-aligned boxes.
# Returns a vector of (region, id_lo, id_hi, normal) where:
#   region = (a, b, c, d) corners of the shared rectangle
#   id_lo < id_hi are box indices
#   normal points from box id_hi toward box id_lo
function _detect_shared_faces_3d(boxes::Vector{<:NamedTuple})
    T = typeof(boxes[1].Lx)
    shared = Tuple{NTuple{4, NTuple{3, T}}, Int, Int, NTuple{3, T}}[]

    n_boxes = length(boxes)
    for i in 1:n_boxes
        faces_i = _box3d_faces_at_center(boxes[i].center, boxes[i].Lx, boxes[i].Ly, boxes[i].Lz)
        for j in (i + 1):n_boxes
            faces_j = _box3d_faces_at_center(boxes[j].center, boxes[j].Lx, boxes[j].Ly, boxes[j].Lz)
            for (a1, b1, c1, d1, n1) in faces_i
                for (a2, b2, c2, d2, n2) in faces_j
                    has_overlap, region = _rect_overlap_3d(a1, b1, c1, d1, n1, a2, b2, c2, d2, n2)
                    if has_overlap
                        # normal points from higher-id box (j) to lower-id box (i)
                        # n1 is the outward normal of box i's face, which points toward box j.
                        push!(shared, (region, i, j, (-n1[1], -n1[2], -n1[3])))
                    end
                end
            end
        end
    end

    return shared
end

# Subtract a set of axis-aligned rectangles from an axis-aligned face in 3D.
# All rectangles must lie on the same plane as the face.
# Returns a vector of (a, b, c, d, is_edge, is_corner) where is_edge and is_corner
# are NTuple{4, Bool} indicating which edges/corners lie on the original face boundary.
# Edge order: (ab, bc, cd, da). Corner order: (a, b, c, d).
function _subtract_rects_from_face_3d(
    a::NTuple{3, T}, b::NTuple{3, T}, c::NTuple{3, T}, d::NTuple{3, T}, normal::NTuple{3, T},
    shared::Vector{<:NTuple{4, NTuple{3, T}}};
    tol::T = T(1e-10)
) where T
    if isempty(shared)
        return [(a, b, c, d, (true, true, true, true), (true, true, true, true))]
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
        return NTuple{3, T}(Tuple(p))
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

    # Build a grid status map: true = shared, false = remaining
    nu = length(u_coords) - 1
    nv = length(v_coords) - 1
    cell_shared = Matrix{Bool}(undef, nu, nv)
    for i in 1:nu
        for j in 1:nv
            u_lo, u_hi = u_coords[i], u_coords[i + 1]
            v_lo, v_hi = v_coords[j], v_coords[j + 1]
            if u_hi - u_lo < tol || v_hi - v_lo < tol
                cell_shared[i, j] = true  # degenerate cell, treat as shared
            else
                cell_shared[i, j] = is_shared((u_lo + u_hi) / 2, (v_lo + v_hi) / 2)
            end
        end
    end

    # A grid point is physical if any surrounding cell is shared or out-of-bounds.
    physical_corner = Matrix{Bool}(undef, nu + 1, nv + 1)
    for gi in 1:(nu + 1)
        for gj in 1:(nv + 1)
            has_nonremaining = false
            for (di, dj) in ((0, 0), (-1, 0), (0, -1), (-1, -1))
                ci, cj = gi + di, gj + dj
                if ci < 1 || ci > nu || cj < 1 || cj > nv
                    has_nonremaining = true
                    break
                elseif cell_shared[ci, cj]
                    has_nonremaining = true
                    break
                end
            end
            physical_corner[gi, gj] = has_nonremaining
        end
    end

    remaining = Tuple{NTuple{3,T}, NTuple{3,T}, NTuple{3,T}, NTuple{3,T}, NTuple{4,Bool}, NTuple{4,Bool}}[]

    for i in 1:nu
        for j in 1:nv
            if cell_shared[i, j]
                continue
            end

            u_lo, u_hi = u_coords[i], u_coords[i + 1]
            v_lo, v_hi = v_coords[j], v_coords[j + 1]

            ra = make_point_3d(u_lo, v_lo)
            rb = make_point_3d(u_hi, v_lo)
            rc = make_point_3d(u_hi, v_hi)
            rd = make_point_3d(u_lo, v_hi)

            # An edge is physical if the neighbor on the other side is either:
            #   - out of bounds (face boundary), or
            #   - a shared (removed) cell (interface with another box)
            # An edge is NOT physical if the neighbor is another remaining cell
            # (just an internal grid line on the same smooth surface).
            edge_ab = (j == 1)  || cell_shared[i, j - 1]   # neighbor below
            edge_bc = (i == nu) || cell_shared[i + 1, j]    # neighbor right
            edge_cd = (j == nv) || cell_shared[i, j + 1]    # neighbor above
            edge_da = (i == 1)  || cell_shared[i - 1, j]    # neighbor left

            corner_a = physical_corner[i, j]
            corner_b = physical_corner[i + 1, j]
            corner_c = physical_corner[i + 1, j + 1]
            corner_d = physical_corner[i, j + 1]

            push!(remaining, (ra, rb, rc, rd,
                (edge_ab, edge_bc, edge_cd, edge_da),
                (corner_a, corner_b, corner_c, corner_d)))
        end
    end

    return remaining
end

# Compute all face regions for a multi-box system.
# Returns a vector of (a, b, c, d, normal, eps_in, eps_out, is_edge, is_corner) for each face region
# (both external and shared).
function _multi_box3d_face_regions(
    boxes::Vector{<:NamedTuple},
    epses::Vector{T},
    eps_out::T,
) where T
    shared_faces = _detect_shared_faces_3d(boxes)
    n_boxes = length(boxes)

    regions = Tuple{NTuple{3,T}, NTuple{3,T}, NTuple{3,T}, NTuple{3,T}, NTuple{3,T}, T, T, NTuple{4,Bool}, NTuple{4,Bool}}[]

    for box_id in 1:n_boxes
        box = boxes[box_id]
        faces = _box3d_faces_at_center(box.center, box.Lx, box.Ly, box.Lz)

        for (fa, fb, fc, fd, fn) in faces
            face_shared = NTuple{4, NTuple{3, T}}[]
            for (region, id_lo, id_hi, normal) in shared_faces
                if box_id == id_lo || box_id == id_hi
                    has_ov, ov_region = _rect_overlap_3d(
                        fa, fb, fc, fd, fn,
                        region[1], region[2], region[3], region[4], (-fn[1], -fn[2], -fn[3]),
                    )
                    if has_ov
                        push!(face_shared, ov_region)
                    end
                end
            end

            remaining = _subtract_rects_from_face_3d(fa, fb, fc, fd, fn, face_shared)
            for (ra, rb, rc, rd, ie, ic) in remaining
                push!(regions, (ra, rb, rc, rd, fn, epses[box_id], eps_out, ie, ic))
            end
        end
    end

    # Shared faces: all edges are physical boundaries (interface between two media)
    for (region, id_lo, id_hi, normal) in shared_faces
        a, b, c, d = region
        push!(regions, (a, b, c, d, normal, epses[id_hi], epses[id_lo],
            (true, true, true, true), (true, true, true, true)))
    end

    return regions
end

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

    regions = _multi_box3d_face_regions(boxes, epses, eps_out)

    panels_vec = Vector{FlatPanel{T, 3}}()
    eps_in_vec = Vector{T}()
    eps_out_vec = Vector{T}()

    for (ra, rb, rc, rd, fn, ei, eo, ie, ic) in regions
        new_panels = rect_panel3d_adaptive_panels(ra, rb, rc, rd, ns, ws, fn, ie, ic, alpha, l_ec)
        append!(panels_vec, new_panels)
        append!(eps_in_vec, fill(ei, length(new_panels)))
        append!(eps_out_vec, fill(eo, length(new_panels)))
    end

    return DielectricInterface(panels_vec, eps_in_vec, eps_out_vec)
end

function multi_dielectric_box3d_rhs_adaptive(
    n_quad::Int, l_ec::T,
    boxes::Vector{<:NamedTuple},
    epses::Vector{T},
    rhs::Function,
    rhs_atol::T,
    eps_out::T = one(T);
    max_depth::Int = 128,
    alpha::T = T(sqrt(T(2))),
) where T
    @assert length(boxes) == length(epses) "Number of boxes must match number of permittivities"
    @assert length(boxes) >= 1 "At least one box is required"

    regions = _multi_box3d_face_regions(boxes, epses, eps_out)

    panels_vec = Vector{FlatPanel{T, 3}}()
    eps_in_vec = Vector{T}()
    eps_out_vec = Vector{T}()

    for (ra, rb, rc, rd, fn, ei, eo, ie, ic) in regions
        new_panels = rect_panel3d_rhs_adaptive_panels(
            ra, rb, rc, rd, n_quad, rhs, fn, ie, ic, alpha, l_ec, rhs_atol, max_depth;
        )
        append!(panels_vec, new_panels)
        append!(eps_in_vec, fill(ei, length(new_panels)))
        append!(eps_out_vec, fill(eo, length(new_panels)))
    end

    return DielectricInterface(panels_vec, eps_in_vec, eps_out_vec)
end

function multi_dielectric_box3d_rhs_adaptive(
    n_quad::Int, l_ec::T,
    boxes::Vector{<:NamedTuple},
    epses::Vector{T},
    ps::PointSource{T, 3},
    eps_src::T,
    rhs_atol::T,
    eps_out::T = one(T);
    max_depth::Int = 128,
    alpha::T = T(sqrt(T(2))),
) where T
    rhs(p, n) = -ps.charge * laplace3d_grad(ps.point, p, n) / eps_src
    return multi_dielectric_box3d_rhs_adaptive(
        n_quad, l_ec, boxes, epses, rhs, rhs_atol, eps_out;
        max_depth = max_depth, alpha = alpha,
    )
end

function multi_dielectric_box3d_rhs_adaptive(
    n_quad::Int, l_ec::T,
    boxes::Vector{<:NamedTuple},
    epses::Vector{T},
    vs::VolumeSource{T, 3},
    rhs_atol::T;
    eps_out::T = one(T),
    max_depth::Int = 128,
    alpha::T = T(sqrt(T(2))),
    tkm_kmax::Union{Nothing, T} = nothing,
) where T
    screened_vs = screened_volume_source(boxes, epses, eps_out, vs, SharpScreening())
    return multi_dielectric_box3d_rhs_adaptive(
        n_quad, l_ec, boxes, epses, screened_vs, one(T), rhs_atol, eps_out;
        max_depth = max_depth, alpha = alpha, tkm_kmax = tkm_kmax,
    )
end

function multi_dielectric_box3d_rhs_adaptive(
    n_quad::Int, l_ec::T,
    boxes::Vector{<:NamedTuple},
    epses::Vector{T},
    vs::VolumeSource{T, 3},
    eps_src::T,
    rhs_atol::T,
    eps_out::T = one(T);
    max_depth::Int = 128,
    alpha::T = T(sqrt(T(2))),
    tkm_kmax::Union{Nothing, T} = nothing,
) where T
    @assert length(boxes) == length(epses) "Number of boxes must match number of permittivities"
    @assert length(boxes) >= 1 "At least one box is required"

    regions = _multi_box3d_face_regions(boxes, epses, eps_out)
    n_regions = length(regions)

    ns, ws = gausslegendre(n_quad)
    ns = T.(ns)
    ws = T.(ws)
    max_depth = max(max_depth, 0)
    fmm_tol = rhs_atol * T(0.1)
    h = _estimate_source_spacing(vs)
    resolved_tkm_kmax = isnothing(tkm_kmax) ? _estimate_tkm3dc_kmax(h) : tkm_kmax
    resolved_tkm_kmax > zero(T) || throw(ArgumentError("tkm_kmax must be positive"))

    @info "multi_dielectric_box3d rhs adaptive: $n_regions face regions, $(length(vs.density)) source points"

    # Initialize panels at depth 0 for each region
    panels_by_depth = [[TempPanel3D{T}[] for _ in 0:max_depth] for _ in 1:n_regions]
    for (r, (ra, rb, rc, rd, fn, ei, eo, ie, ic)) in enumerate(regions)
        Lab = norm(rb .- ra)
        Lda = norm(ra .- rd)
        n_divide_x, n_divide_y = best_grid_mn(Lab, Lda, alpha)
        rough = divide_temp_panel3d(
            TempPanel3D(ra, rb, rc, rd, ic[1], ic[2], ic[3], ic[4], ie[1], ie[2], ie[3], ie[4], fn),
            n_divide_x, n_divide_y,
        )
        for tpl in rough
            push!(panels_by_depth[r][1], tpl)
        end
    end

    fine_panels = [TempPanel3D{T}[] for _ in 1:n_regions]

    # Process all regions together at each depth — one FMM call per depth level
    for depth in 0:max_depth
        all_panels = TempPanel3D{T}[]
        counts = Int[]
        for r in 1:n_regions
            cnt = length(panels_by_depth[r][depth + 1])
            append!(all_panels, panels_by_depth[r][depth + 1])
            push!(counts, cnt)
        end

        isempty(all_panels) && continue

        if depth >= max_depth
            for r in 1:n_regions
                append!(fine_panels[r], panels_by_depth[r][depth + 1])
            end
            continue
        end

        @info "  depth $depth, panels: $(length(all_panels))"
        resolved = _rhs_panel3d_resolved_volume_fmm(
            all_panels, vs, eps_src, ns, ws, rhs_atol, fmm_tol, h, resolved_tkm_kmax,
        )

        offset = 0
        for r in 1:n_regions
            for k in 1:counts[r]
                i = offset + k
                tpl = all_panels[i]
                if resolved[i]
                    push!(fine_panels[r], tpl)
                else
                    for child in divide_temp_panel3d(tpl, 2, 2)
                        push!(panels_by_depth[r][depth + 2], child)
                    end
                end
            end
            offset += counts[r]
        end
    end

    panels_vec = Vector{FlatPanel{T, 3}}()
    eps_in_vec = Vector{T}()
    eps_out_vec = Vector{T}()

    for (r, (ra, rb, rc, rd, fn, ei, eo, ie, ic)) in enumerate(regions)
        rough_ec = copy(fine_panels[r])
        refined = TempPanel3D{T}[]
        while !isempty(rough_ec)
            tpl = popfirst!(rough_ec)
            has_ec = tpl.is_a_corner || tpl.is_b_corner || tpl.is_c_corner || tpl.is_d_corner ||
                tpl.is_ab_edge || tpl.is_bc_edge || tpl.is_cd_edge || tpl.is_da_edge
            L_ab = norm(tpl.b .- tpl.a)
            L_da = norm(tpl.a .- tpl.d)
            if has_ec && max(L_ab, L_da) > l_ec
                append!(rough_ec, divide_temp_panel3d(tpl, 2, 2))
            else
                push!(refined, tpl)
            end
        end

        for tpl in refined
            is_edge_flag = tpl.is_ab_edge || tpl.is_bc_edge || tpl.is_cd_edge || tpl.is_da_edge ||
                tpl.is_a_corner || tpl.is_b_corner || tpl.is_c_corner || tpl.is_d_corner
            push!(panels_vec, rect_panel3d_discretize(tpl.a, tpl.b, tpl.c, tpl.d, ns, ws, tpl.normal; is_edge = is_edge_flag))
        end
        append!(eps_in_vec, fill(ei, length(refined)))
        append!(eps_out_vec, fill(eo, length(refined)))
    end

    return DielectricInterface(panels_vec, eps_in_vec, eps_out_vec)
end
