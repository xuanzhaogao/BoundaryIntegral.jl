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
    xs_src, ys_src, zs_src = vs.axes
    weights = vs.weights
    density = vs.density
    nx, ny, nz = length(xs_src), length(ys_src), length(zs_src)
    n_sources = nx * ny * nz
    sources = Matrix{T}(undef, 3, n_sources)
    charges = Vector{T}(undef, n_sources)
    s = 0
    for ix in 1:nx, iy in 1:ny, iz in 1:nz
        s += 1
        sources[1, s] = xs_src[ix]
        sources[2, s] = ys_src[iy]
        sources[3, s] = zs_src[iz]
        charges[s] = weights[ix, iy, iz] * density[ix, iy, iz]
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

function _rhs_panel3d_resolved_volume_fmm(
    panels::Vector{TempPanel3D{T}},
    vs::VolumeSource{T, 3},
    eps_src::T,
    ns::Vector{T},
    ws::Vector{T},
    atol::T,
    fmm_tol::T,
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
    n_targets = n_panels * (n_quad * n_quad + n_test)

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

    vals = lfmm3d(fmm_tol, sources, charges = charges, targets = targets, pgt = 2)
    grad = vals.gradtarg

    rhs_vals = Vector{T}(undef, n_targets)
    for i in 1:n_targets
        rhs_vals[i] = -dot(normals[:, i], grad[:, i]) / (4π * eps_src)
    end

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

    max_depth = max(max_depth, 0)
    fmm_tol = rhs_atol * T(0.1)

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

        resolved = _rhs_panel3d_resolved_volume_fmm(current, vs, eps_src, ns, ws, rhs_atol, fmm_tol)
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
) where T
    ns, ws = gausslegendre(n_quad)
    fmm_tol = rhs_atol * T(0.1)

    solved = TempPanel3D{T}[]
    unsolved = _box3d_rhs_adaptive_initial_panels(Lx, Ly, Lz, alpha)
    depth = 0
    while !isempty(unsolved) && depth < max_depth
        resolved = _rhs_panel3d_resolved_volume_fmm(unsolved, vs, eps_src, ns, ws, rhs_atol, fmm_tol)
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

# function square_surface_adaptive_panels(a::NTuple{3, T}, b::NTuple{3, T}, c::NTuple{3, T}, d::NTuple{3, T}, ns::Vector{T}, ws::Vector{T}, normal::NTuple{3, T}, is_edge::NTuple{4, Bool}, is_corner::NTuple{4, Bool}, n_adapt_edge::Int, n_adapt_corner::Int) where T

#     @assert n_adapt_edge <= n_adapt_corner "n_adapt_edge must be less than or equal to n_adapt_corner"

#     squares = Vector{NTuple{4, NTuple{3, T}}}()

#     # first handle the edges
#     _square_surfaces!(squares, (a, b, c, d), is_edge, is_corner, n_adapt_edge, n_adapt_corner)

#     panels = Vector{Panel{T, 3}}()
#     for square in squares
#         push!(panels, square_surface_uniform_panel(square..., ns, ws, normal))
#     end

#     return panels
# end

# function square_surface_adaptive_panels(a::NTuple{3, T}, b::NTuple{3, T}, c::NTuple{3, T}, d::NTuple{3, T}, n_quad_max::Int, n_quad_min::Int, normal::NTuple{3, T}, is_edge::NTuple{4, Bool}, is_corner::NTuple{4, Bool}, n_adapt_edge::Int, n_adapt_corner::Int) where T

#     @assert n_adapt_edge <= n_adapt_corner "n_adapt_edge must be less than or equal to n_adapt_corner"


#     if n_adapt_edge == 0
#         ns, ws = gausslegendre(n_quad_min)
#         return [square_surface_uniform_panel(a, b, c, d, ns, ws, normal)]
#     end

#     squares = Vector{NTuple{4, NTuple{3, T}}}()

#     # first handle the edges
#     _square_surfaces!(squares, (a, b, c, d), is_edge, is_corner, n_adapt_edge, n_adapt_corner)

#     L_ab = norm(b .- a)
#     panels = Vector{Panel{T, 3}}()

#     L_min = L_ab
#     for square in squares
#         L_min = min(L_min, norm(square[2] .- square[1]))
#     end

#     for square in squares
#         L = norm(square[2] .- square[1])
#         r = (L_ab == L_min) ? 1 : (L - L_min) / (L_ab - L_min)
#         n_quad = ceil(Int, n_quad_min + (n_quad_max - n_quad_min) * r)
#         ns, ws = gausslegendre(n_quad)
#         push!(panels, square_surface_uniform_panel(square..., ns, ws, normal))
#     end

#     return panels
# end

# function _square_surfaces!(squares::Vector{NTuple{4, NTuple{3, T}}}, abcd::NTuple{4, NTuple{3, T}}, is_edge::NTuple{4, Bool}, is_corner::NTuple{4, Bool}, edge_depth::Int, corner_depth::Int) where T

#     if (!any(is_edge) && !any(is_corner)) || (edge_depth == 0 && corner_depth == 0)
#         push!(squares, abcd)
#         return squares
#     end

#     a, b, c, d = abcd
#     h_ab = (a .+ b) ./ 2
#     h_bc = (b .+ c) ./ 2
#     h_cd = (c .+ d) ./ 2
#     h_da = (d .+ a) ./ 2
#     c_abcd = (a .+ b .+ c .+ d) ./ 4

#     lb = (a, h_ab, c_abcd, h_da)
#     rb = (h_ab, b, h_bc, c_abcd)
#     rt = (c_abcd, h_bc, c, h_cd)
#     lt = (h_da, c_abcd, h_cd, d)

#     sub_squares = (lb, rb, rt, lt)

#     if (edge_depth == 0) 
#         if (corner_depth == 0 || !any(is_corner))
#             push!(squares, abcd)
#         else
#             for i in 1:4
#                 if !is_corner[i] 
#                     push!(squares, sub_squares[i])
#                 else
#                     dummy_is_edge = (false, false, false, false) # no further edge refinement needed
#                     new_is_corner_mut = [false, false, false, false]
#                     new_is_corner_mut[i] = true
#                     _square_surfaces!(squares, sub_squares[i], dummy_is_edge, Tuple(new_is_corner_mut), 0, corner_depth - 1)
#                 end
#             end
#         end
#     else
#         for i in 1:4
#             new_is_corner_mut = [false, false, false, false]
#             new_is_corner_mut[i] = is_corner[i]
#             new_is_corner = Tuple(new_is_corner_mut)

#             j = mod1(i - 1, 4)
#             new_is_edge_mut = [false, false, false, false]
#             new_is_edge_mut[i] = is_edge[i]
#             new_is_edge_mut[j] = is_edge[j]
#             new_is_edge = Tuple(new_is_edge_mut)

#             _square_surfaces!(squares, sub_squares[i], new_is_edge, new_is_corner, edge_depth - 1, corner_depth - 1)
#         end
#     end

#     return squares
# end

# # generate a cubic box with left front bottom corner at lfd and right upper back corner at rbu
# function cubic_box3d(lfd::NTuple{3, T}, rbu::NTuple{3, T}) where T
#     xL, yF, zD = lfd  # left, front, down
#     xR, yB, zU = rbu  # right, back, up

#     @assert xR > xL && yB > yF && zU > zD "rbu must be the opposite corner of lfd (xR>xL, yB>yF, zU>zD)"

#     # 1: (right, front, up)
#     # 2: (left,  front, up)
#     # 3: (left,  back,  up)
#     # 4: (right, back,  up)
#     # 5: (right, front, down)
#     # 6: (left,  front, down)
#     # 7: (left,  back,  down)
#     # 8: (right, back,  down)
#     vertices = NTuple{3,T}[
#         (xR, yF, zU),  # 1
#         (xL, yF, zU),  # 2
#         (xL, yB, zU),  # 3
#         (xR, yB, zU),  # 4
#         (xR, yF, zD),  # 5
#         (xL, yF, zD),  # 6
#         (xL, yB, zD),  # 7
#         (xR, yB, zD),  # 8
#     ]

#     faces = NTuple{4,Int}[
#         (1, 2, 3, 4),  # z = zU  (top),    n = (0, 0, +1)
#         (5, 8, 7, 6),  # z = zD  (bottom), n = (0, 0, -1)
#         (8, 5, 1, 4),  # x = xR  (right),  n = (+1, 0, 0)
#         (7, 3, 2, 6),  # x = xL  (left),   n = (-1, 0, 0)
#         (6, 2, 1, 5),  # y = yF  (front),  n = (0, +1, 0)
#         (7, 8, 4, 3),  # y = yB  (back),   n = (0, -1, 0)
#     ]

#     z0 = zero(T)
#     o  = one(T)

#     normals = NTuple{3,T}[
#         ( z0,  z0,  o ),  # top    (+z)
#         ( z0,  z0, -o ),  # bottom (-z)
#         (  o,  z0,  z0),  # right  (+x)
#         ( -o,  z0,  z0),  # left   (-x)
#         ( z0,  o,  z0),   # front  (+y)
#         ( z0, -o,  z0),   # back   (-y)
#     ]

#     return vertices, faces, normals
# end


# function single_box3d(Lx::T, Ly::T, Lz::T, nx::Int, ny::Int, nz::Int, n_quad_max::Int, n_quad_min::Int, n_adapt_edge::Int, n_adapt_corner::Int, ::Type{T} = Float64) where T

#     vertices = [
#         ( Lx / 2,  Ly / 2,  Lz / 2),  # 1: A
#         (-Lx / 2,  Ly / 2,  Lz / 2),  # 2: B
#         (-Lx / 2, -Ly / 2,  Lz / 2),  # 3: C
#         ( Lx / 2, -Ly / 2,  Lz / 2),  # 4: D
#         ( Lx / 2,  Ly / 2, -Lz / 2),  # 5: E
#         (-Lx / 2,  Ly / 2, -Lz / 2),  # 6: F
#         (-Lx / 2, -Ly / 2, -Lz / 2),  # 7: G
#         ( Lx / 2, -Ly / 2, -Lz / 2),  # 8: H
#     ]

#     faces = [
#         (1, 2, 3, 4),  # z = +1  (A B C D)
#         (5, 8, 7, 6),  # z = -1  (E H G F)
#         (8, 5, 1, 4),  # x = +1  (H E A D)
#         (7, 3, 2, 6),  # x = -1  (G C B F)
#         (6, 2, 1, 5),  # y = +1  (F B A E)
#         (7, 8, 4, 3),  # y = -1  (G H D C)
#     ]

#     n_boxes = [
#         (nx, ny),
#         (nx, ny),
#         (ny, nz),
#         (nz, ny),
#         (nz, nx),
#         (nx, nz)
#     ]

#     t1 = one(T)
#     t0 = zero(T)
#     normals = [
#         ( t0,  t0,  t1),  # (1,2,3,4)
#         ( t0,  t0, -t1),  # (5,8,7,6)
#         ( t1,  t0,  t0),  # (8,5,1,4)
#         (-t1,  t0,  t0),  # (7,3,2,6)
#         ( t0,  t1,  t0),  # (6,2,1,5)
#         ( t0, -t1,  t0),  # (7,8,4,3)
#     ]

#     panels = Vector{Panel{T, 3}}()
#     for i in 1:6
#         face = faces[i]
#         n1, n2 = n_boxes[i]
#         a, b, c, d = [vertices[j] for j in face]
#         r_ab = b .- a
#         r_ad = d .- a
#         d_ab = r_ab ./ n1
#         d_ad = r_ad ./ n2
#         normal = normals[i]
#         for face_idx in 1:n1
#             for face_idy in 1:n2
#                 af = a .+ d_ab .* (face_idx - 1) .+ d_ad .* (face_idy - 1)
#                 bf = af .+ d_ab
#                 cf = af .+ d_ab .+ d_ad
#                 df = af .+ d_ad

#                 is_edge_mut = [false, false, false, false]
#                 is_corner_mut = [false, false, false, false]

#                 is_edge_mut[4] = (face_idx == 1)
#                 is_edge_mut[2] = (face_idx == n1)
#                 is_edge_mut[1] = (face_idy == 1)
#                 is_edge_mut[3] = (face_idy == n2)

#                 is_corner_mut[1] = is_edge_mut[1] && is_edge_mut[4]
#                 is_corner_mut[2] = is_edge_mut[1] && is_edge_mut[2]
#                 is_corner_mut[3] = is_edge_mut[2] && is_edge_mut[3]
#                 is_corner_mut[4] = is_edge_mut[3] && is_edge_mut[4]

#                 new_panels = square_surface_adaptive_panels(af, bf, cf, df, n_quad_max, n_quad_min, normal, Tuple(is_edge_mut), Tuple(is_corner_mut), n_adapt_edge, n_adapt_corner)
#                 append!(panels, new_panels)
#             end
#         end
#     end

#     return Interface(length(panels), panels)
# end

# function dielectric_box3d(eps_box::T, eps_out::T, n_boxes::Int, n_quad_max::Int, n_quad_min::Int, n_adapt_edge::Int, n_adapt_corner::Int, ::Type{T} = Float64) where T
#     box = single_box3d(2.0, 2.0, 2.0, n_boxes, n_boxes, n_boxes, n_quad_max, n_quad_min, n_adapt_edge, n_adapt_corner, T)
#     return DielectricInterfaces(1, [(box, eps_box, eps_out)])
# end

# function dielectric_arbitrary_box3d(eps_box::T, eps_out::T, Lx::T, Ly::T, Lz::T, nx::Int, ny::Int, nz::Int, n_quad_max::Int, n_quad_min::Int, n_adapt_edge::Int, n_adapt_corner::Int, ::Type{T} = Float64) where T
#     box = single_box3d(Lx, Ly, Lz, nx, ny, nz, n_quad_max, n_quad_min, n_adapt_edge, n_adapt_corner, T)
#     return DielectricInterfaces(1, [(box, eps_box, eps_out)])
# end

# function dielectric_double_box3d(eps_box1::T, eps_box2::T, eps_out::T, n_quad_max::Int, n_quad_min::Int, n_adapt_edge::Int, n_adapt_corner::Int, ::Type{T} = Float64) where T

#     interfaces = Vector{Tuple{Interface{T, 3}, T, T}}()

#     # box1 at left, (0, -1, 0) -> (1, 0, 1), box2 at right, (0, 0, 0) -> (1, 1, 1)
#     square_1 = [
#         ((1.0, -1.0, 1.0), (0.0, -1.0, 1.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)), 
#         ((1.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)), 
#         ((1.0, 0.0, 0.0), (1.0, -1.0, 0.0), (1.0, -1.0, 1.0), (1.0, 0.0, 1.0)), 
#         ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 1.0), (0.0, -1.0, 0.0)), 
#         ((0.0, -1.0, 0.0), (0.0, -1.0, 1.0), (1.0, -1.0, 1.0), (1.0, -1.0, 0.0)), 
#         ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0))]
#     square_2 = [
#         ((1.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0)), 
#         ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)), 
#         ((1.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)), 
#         ((0.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)), 
#         ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 0.0, 0.0)), 
#         ((0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0))]
    
#     normals = [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, - 1.0, 0.0), (0.0, 1.0, 0.0)]

#     surf_1 = [(square_1[i], normals[i]) for i in (1:5)]
#     surf_2 = [(square_2[i], normals[i]) for i in [1, 2, 3, 4, 6]]
#     surf_3 = (square_1[6], normals[5])

#     panels_1 = Vector{Panel{T, 3}}()
#     for surf in surf_1
#         vertices, normal = surf
#         new_panels = square_surface_adaptive_panels(vertices..., n_quad_max, n_quad_min, normal, (true, true, true, true), (true, true, true, true), n_adapt_edge, n_adapt_corner)
#         append!(panels_1, new_panels)
#     end
#     interface_1 = Interface(length(panels_1), panels_1)

#     panels_2 = Vector{Panel{T, 3}}()
#     for surf in surf_2
#         vertices, normal = surf
#         new_panels = square_surface_adaptive_panels(vertices..., n_quad_max, n_quad_min, normal, (true, true, true, true), (true, true, true, true), n_adapt_edge, n_adapt_corner)
#         append!(panels_2, new_panels)
#     end
#     interface_2 = Interface(length(panels_2), panels_2)

#     panels_3 = square_surface_adaptive_panels(surf_3[1][1], surf_3[1][2], surf_3[1][3], surf_3[1][4], n_quad_max, n_quad_min, surf_3[2], (true, true, true, true), (true, true, true, true), n_adapt_edge, n_adapt_corner)
#     interface_3 = Interface(length(panels_3), panels_3)

#     interfaces = [(interface_1, eps_box1, eps_out), (interface_2, eps_box2, eps_out), (interface_3, eps_box2, eps_box1)]

#     return DielectricInterfaces(length(interfaces), interfaces)
# end
