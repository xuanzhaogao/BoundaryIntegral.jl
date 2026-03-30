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
    max_depth::Int = 128,
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
    max_depth::Int = 128,
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
    max_depth::Int = 128,
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
    max_depth::Int = 128,
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
    max_depth::Int = 128,
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
