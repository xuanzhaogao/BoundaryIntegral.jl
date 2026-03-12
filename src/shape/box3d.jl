# temporary panal for 3d rectangle surface panel generation
struct TempPanel3D{T}
    a::NTuple{3, T}
    b::NTuple{3, T}
    c::NTuple{3, T}
    d::NTuple{3, T}

    is_a_corner::Bool
    is_b_corner::Bool
    is_c_corner::Bool
    is_d_corner::Bool

    is_ab_edge::Bool
    is_bc_edge::Bool
    is_cd_edge::Bool
    is_da_edge::Bool

    normal::NTuple{3, T}
end

# mesh a rectangle surface panel with tensor product Gauss-Legendre quadrature points
function rect_panel3d_discretize(
    a::NTuple{3, T},
    b::NTuple{3, T},
    c::NTuple{3, T},
    d::NTuple{3, T},
    ns::Vector{T},
    ws::Vector{T},
    normal::NTuple{3, T};
    is_edge::Bool = false,
) where T

    # check edge lengths
    Lab = norm(b .- a)
    Lbc = norm(c .- b)
    Lcd = norm(d .- c)
    Lda = norm(a .- d)
    @assert (Lab ≈ Lcd) && (Lbc ≈ Lda) "Edges of the square are not equal"

    # check perpendicularity
    @assert (abs(dot(normal, b .- a)) < 1e-10) && (abs(dot(normal, c .- b)) < 1e-10) && (abs(dot(normal, d .- c)) < 1e-10) && (abs(dot(normal, a .- d)) < 1e-10) "Normal is not perpendicular to the edges"
    @assert (abs(dot(b .- a, c .- b)) < 1e-10) && (abs(dot(c .- b, d .- c)) < 1e-10) && (abs(dot(d .- c, a .- d)) < 1e-10) && (abs(dot(a .- d, b .- a)) < 1e-10) "Edges are not perpendicular"

    cc = (a .+ b .+ c .+ d) ./ 4

    points = Vector{NTuple{3, T}}()
    for i in 1:length(ns)
        for j in 1:length(ns)
            p = cc .+ (b .- a) .* (ns[i] / 2) .+ (d .- a) .* (ns[j] / 2)
            push!(points, p)
        end
    end    

    weights = Vector{T}()
    for i in 1:length(ns)
        for j in 1:length(ns)
            push!(weights, ws[i] * ws[j] * Lab * Lbc / 4)
        end
    end
    
    corners = [a, b, c, d]

    return FlatPanel(normal, corners, is_edge, length(ns), ns, ws, points, weights)
end

function divide_temp_panel3d(tpl::TempPanel3D{T}, n_divide_x::Int, n_divide_y::Int) where T
    # @assert n_divide_x >= 2 "n_divide_x must be greater than or equal to 2"
    # @assert n_divide_y >= 2 "n_divide_y must be greater than or equal to 2"

    panels = Vector{TempPanel3D{T}}(undef, n_divide_x * n_divide_y) # the panels are arranged in a row-major order

    for i in 1:n_divide_x
        for j in 1:n_divide_y
            u0 = (i - 1) / n_divide_x
            u1 = i / n_divide_x
            v0 = (j - 1) / n_divide_y
            v1 = j / n_divide_y

            a_ij = tpl.a .+ (tpl.b .- tpl.a) .* u0 .+ (tpl.d .- tpl.a) .* v0
            b_ij = tpl.a .+ (tpl.b .- tpl.a) .* u1 .+ (tpl.d .- tpl.a) .* v0
            c_ij = tpl.a .+ (tpl.b .- tpl.a) .* u1 .+ (tpl.d .- tpl.a) .* v1
            d_ij = tpl.a .+ (tpl.b .- tpl.a) .* u0 .+ (tpl.d .- tpl.a) .* v1

            is_a_corner = (i == 1 && j == 1) ? tpl.is_a_corner : false
            is_b_corner = (i == n_divide_x && j == 1) ? tpl.is_b_corner : false
            is_c_corner = (i == n_divide_x && j == n_divide_y) ? tpl.is_c_corner : false
            is_d_corner = (i == 1 && j == n_divide_y) ? tpl.is_d_corner : false

            is_ab_edge = (j == 1) ? tpl.is_ab_edge : false
            is_bc_edge = (i == n_divide_x) ? tpl.is_bc_edge : false
            is_cd_edge = (j == n_divide_y) ? tpl.is_cd_edge : false
            is_da_edge = (i == 1) ? tpl.is_da_edge : false

            panels[(i - 1) * n_divide_y + j] = TempPanel3D(a_ij, b_ij, c_ij, d_ij, is_a_corner, is_b_corner, is_c_corner, is_d_corner, is_ab_edge, is_bc_edge, is_cd_edge, is_da_edge, tpl.normal)
        end
    end
    return panels
end

# alpha controls the coarse grid aspect ratio; l_ec is the maximum length of an edge/corner panel
function rect_panel3d_adaptive_panels(a::NTuple{3, T}, b::NTuple{3, T}, c::NTuple{3, T}, d::NTuple{3, T}, ns::Vector{T}, ws::Vector{T}, normal::NTuple{3, T}, is_edge::NTuple{4, Bool}, is_corner::NTuple{4, Bool}, alpha::T, l_ec::T) where T
    Lab = norm(b .- a)
    Lda = norm(a .- d)
    n_divide_x, n_divide_y = best_grid_mn(Lab, Lda, alpha)

    rough = divide_temp_panel3d(
        TempPanel3D(a, b, c, d, is_corner[1], is_corner[2], is_corner[3], is_corner[4], is_edge[1], is_edge[2], is_edge[3], is_edge[4], normal),
        n_divide_x,
        n_divide_y,
    )

    fine = TempPanel3D{T}[]
    while !isempty(rough)
        tpl = popfirst!(rough)
        has_ec = tpl.is_a_corner || tpl.is_b_corner || tpl.is_c_corner || tpl.is_d_corner ||
            tpl.is_ab_edge || tpl.is_bc_edge || tpl.is_cd_edge || tpl.is_da_edge
        L_ab = norm(tpl.b .- tpl.a)
        L_da = norm(tpl.a .- tpl.d)
        if has_ec && max(L_ab, L_da) > l_ec
            append!(rough, divide_temp_panel3d(tpl, 2, 2))
        else
            push!(fine, tpl)
        end
    end

    panels = Vector{FlatPanel{T, 3}}()
    for tpl in fine
        is_edge = tpl.is_ab_edge || tpl.is_bc_edge || tpl.is_cd_edge || tpl.is_da_edge || tpl.is_a_corner || tpl.is_b_corner || tpl.is_c_corner || tpl.is_d_corner
        push!(panels, rect_panel3d_discretize(tpl.a, tpl.b, tpl.c, tpl.d, ns, ws, tpl.normal; is_edge = is_edge))
    end

    return panels
end

function rhs_panel3d_integral(tpl::TempPanel3D{T}, rhs::Function, n_quad::Int) where T
    ns, ws = gausslegendre(n_quad)
    a, b, c, d = tpl.a, tpl.b, tpl.c, tpl.d
    cc = (a .+ b .+ c .+ d) ./ 4
    bma = b .- a
    dma = d .- a
    Lx = norm(bma)
    Ly = norm(dma)
    scale = Lx * Ly / 4

    val = zero(T)
    for k in 1:n_quad
        x = ns[k] / 2
        for l in 1:n_quad
            y = ns[l] / 2
            p = cc .+ bma .* x .+ dma .* y
            val += ws[k] * ws[l] * T(rhs(p, tpl.normal)) * scale
        end
    end
    return val
end

function rhs_panel3d_resolved(tpl::TempPanel3D{T}, rhs::Function, n_quad::Int, atol::T) where T
    ns, ws = gausslegendre(n_quad)
    a, b, c, d = tpl.a, tpl.b, tpl.c, tpl.d
    cc = (a .+ b .+ c .+ d) ./ 4
    bma = b .- a
    dma = d .- a

    vals = Matrix{T}(undef, n_quad, n_quad)
    for i in 1:n_quad
        u = ns[i]
        for j in 1:n_quad
            v = ns[j]
            p = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
            vals[i, j] = T(rhs(p, tpl.normal))
        end
    end

    λ = gl_barycentric_weights(ns, ws)
    n_pts = 10
    xs = range(-one(T), one(T); length = n_pts)
    ys = range(-one(T), one(T); length = n_pts)

    err = zero(T)
    max_ref = zero(T)
    for u in xs
        rx = T.(barycentric_row(ns, λ, u))
        for v in ys
            ry = T.(barycentric_row(ns, λ, v))
            approx = zero(T)
            for i in 1:n_quad
                for j in 1:n_quad
                    approx += vals[i, j] * rx[i] * ry[j]
                end
            end
            p = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
            exact = T(rhs(p, tpl.normal))
            err = max(err, abs(exact - approx))
            max_ref = max(max_ref, abs(exact))
        end
    end

    return err <= atol
end

function _volume_source_fmm_sources(vs::VolumeSource{T, 3}) where T
    n_sources = length(vs.density)
    sources = Matrix{T}(undef, 3, n_sources)
    charges = Vector{T}(undef, n_sources)
    @inbounds for s in 1:n_sources
        sources[1, s] = vs.positions[1, s]
        sources[2, s] = vs.positions[2, s]
        sources[3, s] = vs.positions[3, s]
        charges[s] = vs.weights[s] * vs.density[s]
    end
    return sources, charges
end

function _box3d_geometry(Lx::T, Ly::T, Lz::T) where T
    t1 = one(T)
    t0 = zero(T)

    vertices = NTuple{3, T}[
        ( Lx / 2,  Ly / 2,  Lz / 2),  # 1
        (-Lx / 2,  Ly / 2,  Lz / 2),  # 2
        (-Lx / 2, -Ly / 2,  Lz / 2),  # 3
        ( Lx / 2, -Ly / 2,  Lz / 2),  # 4
        ( Lx / 2,  Ly / 2, -Lz / 2),  # 5
        (-Lx / 2,  Ly / 2, -Lz / 2),  # 6
        (-Lx / 2, -Ly / 2, -Lz / 2),  # 7
        ( Lx / 2, -Ly / 2, -Lz / 2),  # 8
    ]

    faces = NTuple{4, Int}[
        (1, 2, 3, 4),  # z = +Lz/2
        (5, 8, 7, 6),  # z = -Lz/2
        (8, 5, 1, 4),  # x = +Lx/2
        (7, 3, 2, 6),  # x = -Lx/2
        (6, 2, 1, 5),  # y = +Ly/2
        (7, 8, 4, 3),  # y = -Ly/2
    ]

    normals = NTuple{3, T}[
        ( t0,  t0,  t1),
        ( t0,  t0, -t1),
        ( t1,  t0,  t0),
        (-t1,  t0,  t0),
        ( t0,  t1,  t0),
        ( t0, -t1,  t0),
    ]

    return vertices, faces, normals
end

function _box3d_face_quads(Lx::T, Ly::T, Lz::T) where T
    vertices, faces, normals = _box3d_geometry(Lx, Ly, Lz)
    quads = NTuple{4, NTuple{3, T}}[]
    face_normals = NTuple{3, T}[]
    for i in 1:6
        face = faces[i]
        push!(quads, (vertices[face[1]], vertices[face[2]], vertices[face[3]], vertices[face[4]]))
        push!(face_normals, normals[i])
    end
    return quads, face_normals
end

function _box3d_rhs_adaptive_initial_panels(Lx::T, Ly::T, Lz::T, alpha::T) where T
    vertices, faces, normals = _box3d_geometry(Lx, Ly, Lz)

    panels = TempPanel3D{T}[]
    is_edge = (true, true, true, true)
    is_corner = (true, true, true, true)
    for i in 1:6
        face = faces[i]
        a, b, c, d = vertices[face[1]], vertices[face[2]], vertices[face[3]], vertices[face[4]]
        normal = normals[i]
        Lab = norm(b .- a)
        Lda = norm(a .- d)
        n_divide_x, n_divide_y = best_grid_mn(Lab, Lda, alpha)
        append!(panels, divide_temp_panel3d(
            TempPanel3D(a, b, c, d, is_corner[1], is_corner[2], is_corner[3], is_corner[4],
                is_edge[1], is_edge[2], is_edge[3], is_edge[4], normal),
            n_divide_x,
            n_divide_y,
        ))
    end

    return panels
end

function _estimate_source_spacing(vs::VolumeSource{T, 3}) where T
    n = size(vs.positions, 2)
    if n <= 1
        n == 0 && return zero(T)
        # Fallback for sparse sources: infer a characteristic spacing from quadrature volume.
        return cbrt(max(abs(vs.weights[1]), eps(T)))
    end
    tree = KDTree(vs.positions)
    h = typemax(T)
    for i in 1:n
        idxs, dists = knn(tree, view(vs.positions, :, i), 2, true)
        h = min(h, T(dists[2]))
    end
    if !isfinite(h) || h <= zero(T)
        return cbrt(max(mean(abs.(vs.weights)), eps(T)))
    end
    return h
end

@inline function _estimate_tkm3dc_kmax(h::T) where T
    h > zero(T) || throw(ArgumentError("source spacing must be positive"))
    return T(π) / h
end

function _estimate_tkm3dc_kmax(vs::VolumeSource{T, 3}) where T
    return _estimate_tkm3dc_kmax(_estimate_source_spacing(vs))
end

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

function _classify_near_far_targets(targets::Matrix{T}, vs::VolumeSource{T, 3}, h::T, h_factor::T = T(5)) where T
    n_targets = size(targets, 2)
    is_near = fill(false, n_targets)
    n_sources = size(vs.positions, 2)
    n_sources == 0 && return is_near

    tree = KDTree(vs.positions)
    radius = h * h_factor

    for i in 1:n_targets
        idxs = inrange(tree, view(targets, :, i), radius)
        if !isempty(idxs)
            is_near[i] = true
        end
    end
    return is_near
end

@inline function _rhs_from_grad(normal::AbstractVector{T}, grad::AbstractVector{T}, eps_src::T, kernel_scale::T) where T
    return - dot(normal, grad) / (kernel_scale * eps_src)
end

function _rhs_volume_targets_hybrid(
    sources::Matrix{T},
    charges::Vector{T},
    targets::Matrix{T},
    normals::Matrix{T},
    eps_src::T,
    fmm_tol::T,
    tkm_kmax::T,
    is_near::Vector{Bool},
) where T
    n_targets = size(targets, 2)
    @assert size(normals, 2) == n_targets
    @assert length(is_near) == n_targets

    near_idxs = Int[]
    far_idxs = Int[]
    for i in 1:n_targets
        if is_near[i]
            push!(near_idxs, i)
        else
            push!(far_idxs, i)
        end
    end

    rhs_vals = Vector{T}(undef, n_targets)

    n_far = length(far_idxs)
    if n_far > 0
        far_targets = targets[:, far_idxs]
        vals_far = lfmm3d(fmm_tol, sources, charges = charges, targets = far_targets, pgt = 2)
        grad_far = vals_far.gradtarg
        for (k, i) in enumerate(far_idxs)
            # FMM3D gradient uses the 1/r kernel normalization.
            rhs_vals[i] = _rhs_from_grad(view(normals, :, i), view(grad_far, :, k), eps_src, T(4π))
        end
    end

    n_near = length(near_idxs)
    if n_near > 0
        near_targets = targets[:, near_idxs]
        vals_near = ltkm3dc(fmm_tol, sources; charges = charges, targets = near_targets, pgt = 2, kmax = tkm_kmax)
        vals_near.ier == 0 || error("ltkm3dc target evaluation failed with ier=$(vals_near.ier)")
        grad_near = vals_near.gradtarg
        for (k, i) in enumerate(near_idxs)
            # TKM3D gradient already uses the free-space 1/(4πr) normalization.
            rhs_vals[i] = _rhs_from_grad(view(normals, :, i), view(grad_near, :, k), eps_src, one(T))
        end
    end

    return rhs_vals, n_near, n_far
end

function _rhs_panel3d_resolved_volume_fmm(
    panels::Vector{TempPanel3D{T}},
    vs::VolumeSource{T, 3},
    eps_src::T,
    ns::Vector{T},
    ws::Vector{T},
    atol::T,
    fmm_tol::T,
    h::T,
    tkm_kmax::T,
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

    is_near = _classify_near_far_panels(panels, vs, h)

    # Promote panel-level near/far tags to target-level flags.
    is_near_target = Vector{Bool}(undef, n_targets)
    for p in 1:n_panels
        panel_range = ((p - 1) * n_per_panel + 1):(p * n_per_panel)
        fill!(view(is_near_target, panel_range), is_near[p])
    end

    rhs_vals, n_near, n_far = _rhs_volume_targets_hybrid(
        sources,
        charges,
        targets,
        normals,
        eps_src,
        fmm_tol,
        tkm_kmax,
        is_near_target,
    )

    @info "    rhs panel hybrid evaluation, source points: $(length(charges)), near targets: $n_near, far targets: $n_far"

    # Check resolution per panel
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

function rhs_panel3d_quad_order(tpl::TempPanel3D{T}, rhs::Function, n_quad_min::Int, n_quad_max::Int, atol::T) where T
    @assert n_quad_min >= 1 "n_quad_min must be >= 1"
    @assert n_quad_max >= n_quad_min "n_quad_max must be >= n_quad_min"
    for n_quad in n_quad_min:n_quad_max
        if rhs_panel3d_resolved(tpl, rhs, n_quad, atol)
            return n_quad
        end
    end
    return n_quad_max
end

function rect_panel3d_rhs_adaptive_panels(
    a::NTuple{3, T},
    b::NTuple{3, T},
    c::NTuple{3, T},
    d::NTuple{3, T},
    n_quad::Int,
    rhs::Function,
    normal::NTuple{3, T},
    is_edge::NTuple{4, Bool},
    is_corner::NTuple{4, Bool},
    alpha::T,
    l_ec::T,
    rhs_atol::T,
    max_depth::Int;
) where T
    ns, ws = gausslegendre(n_quad)
    Lab = norm(b .- a)
    Lda = norm(a .- d)
    n_divide_x, n_divide_y = best_grid_mn(Lab, Lda, alpha)
    rough = divide_temp_panel3d(
        TempPanel3D(a, b, c, d, is_corner[1], is_corner[2], is_corner[3], is_corner[4], is_edge[1], is_edge[2], is_edge[3], is_edge[4], normal),
        n_divide_x,
        n_divide_y,
    )
    stack = [(tpl, 0) for tpl in rough]
    fine = TempPanel3D{T}[]

    while !isempty(stack)
        tpl, depth = pop!(stack)
        if depth >= max_depth || rhs_panel3d_resolved(tpl, rhs, n_quad, rhs_atol)
            push!(fine, tpl)
        else
            for child in divide_temp_panel3d(tpl, 2, 2)
                push!(stack, (child, depth + 1))
            end
        end
    end

    rough_ec = copy(fine)
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

    panels = Vector{FlatPanel{T, 3}}()
    for tpl in refined
        is_edge = tpl.is_ab_edge || tpl.is_bc_edge || tpl.is_cd_edge || tpl.is_da_edge ||
            tpl.is_a_corner || tpl.is_b_corner || tpl.is_c_corner || tpl.is_d_corner
        push!(panels, rect_panel3d_discretize(tpl.a, tpl.b, tpl.c, tpl.d, ns, ws, tpl.normal; is_edge = is_edge))
    end
    return panels
end

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
    tkm_kmax::T;
) where T
    ns, ws = gausslegendre(n_quad)
    Lab = norm(b .- a)
    Lda = norm(a .- d)
    n_divide_x, n_divide_y = best_grid_mn(Lab, Lda, alpha)
    rough = divide_temp_panel3d(
        TempPanel3D(a, b, c, d, is_corner[1], is_corner[2], is_corner[3], is_corner[4], is_edge[1], is_edge[2], is_edge[3], is_edge[4], normal),
        n_divide_x,
        n_divide_y,
    )

    max_depth = max(max_depth, 0)
    fmm_tol = rhs_atol * T(0.1)
    h = _estimate_source_spacing(vs)

    panels_by_depth = [TempPanel3D{T}[] for _ in 0:max_depth]
    for tpl in rough
        push!(panels_by_depth[1], tpl)
    end

    fine = TempPanel3D{T}[]

    for depth in 0:max_depth
        current = panels_by_depth[depth + 1]
        isempty(current) && continue

        if depth >= max_depth
            append!(fine, current)
            continue
        end

        resolved = _rhs_panel3d_resolved_volume_fmm(current, vs, eps_src, ns, ws, rhs_atol, fmm_tol, h, tkm_kmax)
        for i in eachindex(current)
            tpl = current[i]
            if resolved[i]
                push!(fine, tpl)
            else
                for child in divide_temp_panel3d(tpl, 2, 2)
                    push!(panels_by_depth[depth + 2], child)
                end
            end
        end
    end

    rough_ec = copy(fine)
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

    panels = Vector{FlatPanel{T, 3}}()
    for tpl in refined
        is_edge = tpl.is_ab_edge || tpl.is_bc_edge || tpl.is_cd_edge || tpl.is_da_edge ||
            tpl.is_a_corner || tpl.is_b_corner || tpl.is_c_corner || tpl.is_d_corner
        push!(panels, rect_panel3d_discretize(tpl.a, tpl.b, tpl.c, tpl.d, ns, ws, tpl.normal; is_edge = is_edge))
    end
    return panels
end

function rect_panel3d_rhs_adaptive_panels_varquad(
    a::NTuple{3, T},
    b::NTuple{3, T},
    c::NTuple{3, T},
    d::NTuple{3, T},
    n_quad_max::Int,
    rhs::Function,
    normal::NTuple{3, T},
    is_edge::NTuple{4, Bool},
    is_corner::NTuple{4, Bool},
    alpha::T,
    l_ec::T,
    rhs_atol::T,
    max_depth::Int;
    n_quad_min::Int = 2,
) where T
    Lab = norm(b .- a)
    Lda = norm(a .- d)
    n_divide_x, n_divide_y = best_grid_mn(Lab, Lda, alpha)
    rough = divide_temp_panel3d(
        TempPanel3D(a, b, c, d, is_corner[1], is_corner[2], is_corner[3], is_corner[4], is_edge[1], is_edge[2], is_edge[3], is_edge[4], normal),
        n_divide_x,
        n_divide_y,
    )
    stack = [(tpl, 0) for tpl in rough]
    fine = TempPanel3D{T}[]

    while !isempty(stack)
        tpl, depth = pop!(stack)
        if depth >= max_depth || rhs_panel3d_resolved(tpl, rhs, n_quad_max, rhs_atol)
            push!(fine, tpl)
        else
            for child in divide_temp_panel3d(tpl, 2, 2)
                push!(stack, (child, depth + 1))
            end
        end
    end

    rough_ec = copy(fine)
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

    quad_cache = Dict{Int, Tuple{Vector{T}, Vector{T}}}()
    panels = Vector{FlatPanel{T, 3}}()
    for tpl in refined
        n_quad = rhs_panel3d_quad_order(tpl, rhs, n_quad_min, n_quad_max, rhs_atol)
        ns, ws = get!(quad_cache, n_quad) do
            gausslegendre(n_quad)
        end
        is_edge = tpl.is_ab_edge || tpl.is_bc_edge || tpl.is_cd_edge || tpl.is_da_edge ||
            tpl.is_a_corner || tpl.is_b_corner || tpl.is_c_corner || tpl.is_d_corner
        push!(panels, rect_panel3d_discretize(tpl.a, tpl.b, tpl.c, tpl.d, ns, ws, tpl.normal; is_edge = is_edge))
    end
    return panels
end

function single_dielectric_box3d(Lx::T, Ly::T, Lz::T, n_quad::Int, l_ec::T, eps_in::T, eps_out::T, ::Type{T} = Float64; alpha::T = sqrt(T(2))) where T
    ns, ws = gausslegendre(n_quad)
    quads, normals = _box3d_face_quads(Lx, Ly, Lz)

    panels = Vector{FlatPanel{T, 3}}()
    for i in eachindex(quads)
        a, b, c, d = quads[i]
        normal = normals[i]
        append!(panels, rect_panel3d_adaptive_panels(
            a, b, c, d,
            ns,
            ws,
            normal,
            (true, true, true, true),
            (true, true, true, true),
            alpha,
            l_ec,
        ))
    end

    return DielectricInterface(panels, fill(eps_in, length(panels)), fill(eps_out, length(panels)))
end

function single_dielectric_box3d_rhs_adaptive(
    Lx::T,
    Ly::T,
    Lz::T,
    n_quad::Int,
    rhs::Function,
    l_ec::T,
    rhs_atol::T,
    eps_in::T,
    eps_out::T,
    ::Type{T} = Float64;
    max_depth::Int = 8,
    alpha::T = sqrt(T(2)),
) where T
    quads, normals = _box3d_face_quads(Lx, Ly, Lz)

    panels = Vector{FlatPanel{T, 3}}()
    for i in eachindex(quads)
        a, b, c, d = quads[i]
        normal = normals[i]
        append!(panels, rect_panel3d_rhs_adaptive_panels(
            a,
            b,
            c,
            d,
            n_quad,
            rhs,
            normal,
            (true, true, true, true),
            (true, true, true, true),
            alpha,
            l_ec,
            rhs_atol,
            max_depth;
        ))
    end

    return DielectricInterface(panels, fill(eps_in, length(panels)), fill(eps_out, length(panels)))
end

function single_dielectric_box3d_rhs_adaptive_varquad(
    Lx::T,
    Ly::T,
    Lz::T,
    n_quad_max::Int,
    rhs::Function,
    l_ec::T,
    rhs_atol::T,
    eps_in::T,
    eps_out::T,
    ::Type{T} = Float64;
    max_depth::Int = 8,
    alpha::T = sqrt(T(2)),
    n_quad_min::Int = 2,
) where T
    quads, normals = _box3d_face_quads(Lx, Ly, Lz)

    panels = Vector{FlatPanel{T, 3}}()
    for i in eachindex(quads)
        a, b, c, d = quads[i]
        normal = normals[i]
        append!(panels, rect_panel3d_rhs_adaptive_panels_varquad(
            a,
            b,
            c,
            d,
            n_quad_max,
            rhs,
            normal,
            (true, true, true, true),
            (true, true, true, true),
            alpha,
            l_ec,
            rhs_atol,
            max_depth;
            n_quad_min = n_quad_min,
        ))
    end

    return DielectricInterface(panels, fill(eps_in, length(panels)), fill(eps_out, length(panels)))
end

function single_dielectric_box3d_rhs_adaptive(
    Lx::T,
    Ly::T,
    Lz::T,
    n_quad::Int,
    ps::PointSource{T, 3},
    eps_src::T,
    l_ec::T,
    rhs_atol::T,
    eps_in::T,
    eps_out::T,
    ::Type{T} = Float64;
    max_depth::Int = 100,
    alpha::T = sqrt(T(2)),
) where T
    rhs(p, n) = -ps.charge * laplace3d_grad(ps.point, p, n) / eps_src
    return single_dielectric_box3d_rhs_adaptive(
        Lx,
        Ly,
        Lz,
        n_quad,
        rhs,
        l_ec,
        rhs_atol,
        eps_in,
        eps_out,
        T;
        max_depth = max_depth,
        alpha = alpha,
    )
end

function single_dielectric_box3d_rhs_adaptive(
    Lx::T,
    Ly::T,
    Lz::T,
    n_quad::Int,
    vs::VolumeSource{T, 3},
    eps_src::T,
    l_ec::T,
    rhs_atol::T,
    eps_in::T,
    eps_out::T,
    ::Type{T} = Float64;
    max_depth::Int = 100,
    alpha::T = sqrt(T(2)),
    tkm_kmax::Union{Nothing, T} = nothing,
) where T
    ns, ws = gausslegendre(n_quad)
    fmm_tol = rhs_atol * T(0.1)
    h = _estimate_source_spacing(vs)
    resolved_tkm_kmax = isnothing(tkm_kmax) ? _estimate_tkm3dc_kmax(h) : tkm_kmax
    resolved_tkm_kmax > zero(T) || throw(ArgumentError("tkm_kmax must be positive"))

    @info "box3d volume source rhs adaptive panel generation, source points: $(length(vs.density))"

    # rhs refinement
    solved = TempPanel3D{T}[]
    unsolved = _box3d_rhs_adaptive_initial_panels(Lx, Ly, Lz, alpha)
    depth = 0
    while !isempty(unsolved) && depth < max_depth
        @info "  depth $depth, unsolved panels: $(length(unsolved))"
        resolved = _rhs_panel3d_resolved_volume_fmm(unsolved, vs, eps_src, ns, ws, rhs_atol, fmm_tol, h, resolved_tkm_kmax)
        next_unsolved = TempPanel3D{T}[]
        for i in eachindex(unsolved)
            tpl = unsolved[i]
            if resolved[i]
                push!(solved, tpl)
            else
                append!(next_unsolved, divide_temp_panel3d(tpl, 2, 2))
            end
        end
        unsolved = next_unsolved
        depth += 1
    end

    append!(solved, unsolved)

    # edge and corner refinement
    rough_ec = copy(solved)
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

    # add tensor product gl points to panels
    panels = Vector{FlatPanel{T, 3}}()
    for tpl in refined
        is_edge = tpl.is_ab_edge || tpl.is_bc_edge || tpl.is_cd_edge || tpl.is_da_edge ||
            tpl.is_a_corner || tpl.is_b_corner || tpl.is_c_corner || tpl.is_d_corner
        push!(panels, rect_panel3d_discretize(tpl.a, tpl.b, tpl.c, tpl.d, ns, ws, tpl.normal; is_edge = is_edge))
    end

    return DielectricInterface(panels, fill(eps_in, length(panels)), fill(eps_out, length(panels)))
end

function single_dielectric_box3d_rhs_adaptive_varquad(
    Lx::T,
    Ly::T,
    Lz::T,
    n_quad_max::Int,
    ps::PointSource{T, 3},
    eps_src::T,
    l_ec::T,
    rhs_atol::T,
    eps_in::T,
    eps_out::T,
    ::Type{T} = Float64;
    max_depth::Int = 100,
    alpha::T = sqrt(T(2)),
    n_quad_min::Int = 2,
) where T
    rhs(p, n) = -ps.charge * laplace3d_grad(ps.point, p, n) / eps_src
    return single_dielectric_box3d_rhs_adaptive_varquad(
        Lx,
        Ly,
        Lz,
        n_quad_max,
        rhs,
        l_ec,
        rhs_atol,
        eps_in,
        eps_out,
        T;
        max_depth = max_depth,
        alpha = alpha,
        n_quad_min = n_quad_min,
    )
end
