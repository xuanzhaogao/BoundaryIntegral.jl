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
        return cbrt(max(sum(abs, vs.weights) / length(vs.weights), eps(T)))
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

function _rhs_panel3d_refinement_targets(
    panels::Vector{TempPanel3D{T}},
    ns::AbstractVector{T},
    ws::AbstractVector{T};
    n_pts::Int = 10,
) where T
    length(ns) == length(ws) || throw(ArgumentError("ns and ws must have the same length"))
    n_pts >= 1 || throw(ArgumentError("n_pts must be >= 1"))

    n_panels = length(panels)
    n_quad = length(ns)
    n_test = n_pts * n_pts
    n_per_panel = n_quad * n_quad + n_test
    n_targets = n_panels * n_per_panel
    targets = Matrix{T}(undef, 3, n_targets)
    normals = Matrix{T}(undef, 3, n_targets)
    xs = range(-one(T), one(T); length = n_pts)
    ys = range(-one(T), one(T); length = n_pts)

    idx = 0
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
                idx += 1
                p = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
                targets[1, idx] = p[1]
                targets[2, idx] = p[2]
                targets[3, idx] = p[3]
                normals[1, idx] = normal[1]
                normals[2, idx] = normal[2]
                normals[3, idx] = normal[3]
            end
        end

        for u in xs
            for v in ys
                idx += 1
                p = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
                targets[1, idx] = p[1]
                targets[2, idx] = p[2]
                targets[3, idx] = p[3]
                normals[1, idx] = normal[1]
                normals[2, idx] = normal[2]
                normals[3, idx] = normal[3]
            end
        end
    end

    return targets, normals, n_per_panel
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
    targets, normals, n_per_panel = _rhs_panel3d_refinement_targets(panels, ns, ws; n_pts = n_pts)
    n_targets = size(targets, 2)

    sources, charges = _volume_source_fmm_sources(vs)
    is_near_target = _classify_near_far_targets(targets, vs, h)

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
