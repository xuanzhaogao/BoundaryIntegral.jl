function _DT_integrand_hcubature(
    ::Val{NQ}, ns::Vector{T}, bary_weights::Vector{T},
    cc::NTuple{3,T}, bma::NTuple{3,T}, dma::NTuple{3,T},
    scale::T, point_trg::NTuple{3,T}, normal_trg::NTuple{3,T}
) where {NQ, T}
    N = NQ * NQ
    rx = MVector{NQ, T}(undef)
    ry = MVector{NQ, T}(undef)
    vals = MVector{N, T}(undef)
    function integrand(x)
        u = x[1]
        v = x[2]
        barycentric_row!(rx, ns, bary_weights, u)
        barycentric_row!(ry, ns, bary_weights, v)
        p = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
        k = laplace3d_grad(p, point_trg, normal_trg) * scale
        idx = 1
        @inbounds for ii in 1:NQ
            for jj in 1:NQ
                vals[idx] = k * rx[ii] * ry[jj]
                idx += 1
            end
        end
        return SVector(vals)
    end
    return integrand
end

function laplace3d_DT_panel_hcubature(panel_src::FlatPanel{T, 3}, panel_trg::FlatPanel{T, 3}, atol::T) where T
    ns = panel_src.gl_xs
    bary_weights = panel_src.bary_weights
    a, b, c, d = panel_src.corners
    cc = (a .+ b .+ c .+ d) ./ 4
    bma = b .- a
    dma = d .- a
    Lx = norm(b .- a)
    Ly = norm(d .- a)
    scale = Lx * Ly / 4

    n_quad = panel_src.n_quad
    nq_val = Val(n_quad)
    np_trg = num_points(panel_trg)
    DT_exact = zeros(T, np_trg, n_quad * n_quad)
    points_trg = panel_trg.points
    normal_trg = panel_trg.normal
    lb = SVector{2,T}(-1, -1)
    ub = SVector{2,T}(1, 1)

    Base.Threads.@threads for ti in 1:np_trg
        point_trg = points_trg[ti]
        integrand = _DT_integrand_hcubature(nq_val, ns, bary_weights, cc, bma, dma, scale, point_trg, normal_trg)
        res, _ = hcubature(integrand, lb, ub; atol = atol)
        @views DT_exact[ti, :] .= res
    end

    return DT_exact
end

function _pot_integrand_hcubature(
    ::Val{NQ}, ns::Vector{T}, bary_weights::Vector{T},
    cc::NTuple{3,T}, bma::NTuple{3,T}, dma::NTuple{3,T},
    scale::T, target::NTuple{3,T}
) where {NQ, T}
    N = NQ * NQ
    rx = MVector{NQ, T}(undef)
    ry = MVector{NQ, T}(undef)
    vals = MVector{N, T}(undef)
    function integrand(x)
        u = x[1]
        v = x[2]
        barycentric_row!(rx, ns, bary_weights, u)
        barycentric_row!(ry, ns, bary_weights, v)
        p = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
        k = laplace3d_pot(p, target) * scale
        idx = 1
        @inbounds for ii in 1:NQ
            for jj in 1:NQ
                vals[idx] = k * rx[ii] * ry[jj]
                idx += 1
            end
        end
        return SVector(vals)
    end
    return integrand
end

"""
    laplace3d_pot_panel_hcubature(panel_src, targets, target_ids, atol; threaded = true)

Adaptive-quadrature potential of one source panel at `target_ids`. `threaded = false` runs the
target loop serially, for callers that are ALREADY inside a parallel region --
`laplace3d_pottrg_corrections_hcubature` parallelises over panels instead, so threading here
too would nest parallel regions.
"""
function laplace3d_pot_panel_hcubature(
    panel_src::FlatPanel{T, 3},
    targets::Matrix{T},
    target_ids::Vector{Int},
    atol::T;
    threaded::Bool = true,
) where T
    ns = panel_src.gl_xs
    bary_weights = panel_src.bary_weights
    a, b, c, d = panel_src.corners
    cc = (a .+ b .+ c .+ d) ./ 4
    bma = b .- a
    dma = d .- a
    Lx = norm(b .- a)
    Ly = norm(d .- a)
    scale = Lx * Ly / 4

    n_quad = panel_src.n_quad
    nq_val = Val(n_quad)
    n_src = n_quad * n_quad
    pot_exact = zeros(T, length(target_ids), n_src)
    lb = SVector{2,T}(-1, -1)
    ub = SVector{2,T}(1, 1)

    @inline function _one_target(ti)
        target_id = target_ids[ti]
        target = (targets[1, target_id], targets[2, target_id], targets[3, target_id])
        integrand = _pot_integrand_hcubature(nq_val, ns, bary_weights, cc, bma, dma, scale, target)
        res, _ = hcubature(integrand, lb, ub; atol = atol)
        @views pot_exact[ti, :] .= res
    end
    if threaded
        Base.Threads.@threads for ti in 1:length(target_ids)
            _one_target(ti)
        end
    else
        for ti in 1:length(target_ids)
            _one_target(ti)
        end
    end

    return pot_exact
end

function laplace3d_DT_corrections_hcubature(
    interface::DielectricInterface{P, T},
    neighbor_list::Dict{Tuple{Int, Int}, Int},
    atol::T,
) where {P <: AbstractPanel, T}

    cnt = zeros(Int, length(interface.panels))
    for i in 1:length(interface.panels)
        cnt[i] = length(interface.panels[i].points)
    end
    offsets = cumsum(vcat(0, cnt))
    total_n = offsets[end]

    rows = Int[]
    cols = Int[]
    vals = T[]

    for ((i, j), _) in neighbor_list
        panel_src = interface.panels[i]
        panel_trg = interface.panels[j]

        DT_exact = laplace3d_DT_panel_hcubature(panel_src, panel_trg, atol)
        DT_direct = laplace3d_DT_panel(panel_src, panel_trg)
        block = DT_exact - DT_direct

        row_range = (offsets[j] + 1):offsets[j + 1]
        col_range = (offsets[i] + 1):offsets[i + 1]

        for (r_local, r) in enumerate(row_range)
            for (c_local, c) in enumerate(col_range)
                v = block[r_local, c_local]
                iszero(v) && continue
                push!(rows, r)
                push!(cols, c)
                push!(vals, v)
            end
        end
    end

    return sparse(rows, cols, vals, total_n, total_n)
end

function laplace3d_pottrg_corrections_hcubature(
    interface::DielectricInterface{P, T},
    targets::Matrix{T},
    target_neighbor_list::Dict{Int, Vector{Int}},
    atol::T,
) where {P <: AbstractPanel, T}
    cnt = zeros(Int, length(interface.panels))
    for i in 1:length(interface.panels)
        cnt[i] = length(interface.panels[i].points)
    end
    offsets = cumsum(vcat(0, cnt))
    total_n = offsets[end]
    n_targets = size(targets, 2)

    # Parallelise over PANELS, not over one panel's targets.
    #
    # This loop used to be serial, with laplace3d_pot_panel_hcubature opening a @threads region
    # for each panel's near targets -- typically a handful. Every entry of the neighbour list
    # therefore paid a fork/join over the whole thread pool for a few iterations of work, and
    # that overhead GROWS with the thread count. Measured on the Sec. 5.2 evaluation benchmark
    # (7.17M targets): 21.6 s at 64 threads against 99.1 s at 96, while the FMM (0.95 s) and
    # NUFFT (1.25 s) components underneath were flat. One parallel region over the whole
    # neighbour list removes both the fork/join overhead and the serial push! accumulation.
    #
    # Buffers are indexed by CHUNK, not by threadid(): Threads.@threads may migrate tasks
    # between threads, so threadid() inside the loop body is not a safe buffer key. Chunking at
    # a multiple of nthreads keeps the dynamic scheduler balanced -- panels have very unequal
    # target counts -- while giving each chunk sole ownership of its buffer.
    entries = collect(target_neighbor_list)
    n_entries = length(entries)
    if n_entries == 0
        return sparse(Int[], Int[], T[], n_targets, total_n)
    end
    nchunks = min(n_entries, max(1, 4 * Base.Threads.nthreads()))
    chunks = collect(Iterators.partition(1:n_entries, cld(n_entries, nchunks)))

    rows_c = [Int[] for _ in 1:length(chunks)]
    cols_c = [Int[] for _ in 1:length(chunks)]
    vals_c = [T[] for _ in 1:length(chunks)]

    Base.Threads.@threads for ci in 1:length(chunks)
        rows, cols, vals = rows_c[ci], cols_c[ci], vals_c[ci]
        for e in chunks[ci]
            i, target_ids = entries[e]
            panel_src = interface.panels[i]
            col_range = (offsets[i] + 1):offsets[i + 1]

            # serial: we are already inside a parallel region
            pot_exact = laplace3d_pot_panel_hcubature(panel_src, targets, target_ids, atol;
                                                      threaded = false)

            for (t_local, t_global) in enumerate(target_ids)
                target = (targets[1, t_global], targets[2, t_global], targets[3, t_global])
                for (c_local, c_global) in enumerate(col_range)
                    direct = panel_src.weights[c_local] * laplace3d_pot(panel_src.points[c_local], target)
                    v = pot_exact[t_local, c_local] - direct
                    iszero(v) && continue
                    push!(rows, t_global)
                    push!(cols, c_global)
                    push!(vals, v)
                end
            end
        end
    end

    # (t_global, c_global) pairs are unique across panels -- distinct panels own disjoint
    # column ranges -- so sparse() sees no duplicates and the result does not depend on the
    # order in which the chunks are concatenated.
    return sparse(reduce(vcat, rows_c), reduce(vcat, cols_c), reduce(vcat, vals_c),
                  n_targets, total_n)
end

# direct evaluation of correction action (DT_exact - DT_direct) * sigma
function laplace3d_DT_corrections_hcubature_apply(
    interface::DielectricInterface{P, T},
    neighbor_list::Dict{Tuple{Int, Int}, Int},
    atol::T,
    sigma::Function,
) where {P <: AbstractPanel, T}
    cnt = zeros(Int, length(interface.panels))
    for i in 1:length(interface.panels)
        cnt[i] = length(interface.panels[i].points)
    end
    offsets = cumsum(vcat(0, cnt))
    total_n = offsets[end]

    out = zeros(T, total_n)

    for ((i, j), _) in neighbor_list
        panel_src = interface.panels[i]
        panel_trg = interface.panels[j]

        # precompute sigma at source quadrature nodes for the direct term
        sigma_src = Vector{T}(undef, length(panel_src.points))
        for k in 1:length(panel_src.points)
            sigma_src[k] = T(sigma(panel_src.points[k]))
        end

        a, b, c, d = panel_src.corners
        cc = (a .+ b .+ c .+ d) ./ 4
        bma = b .- a
        dma = d .- a
        Lx = norm(b .- a)
        Ly = norm(d .- a)
        scale = Lx * Ly / 4

        weights_src = panel_src.weights
        points_src = panel_src.points
        points_trg = panel_trg.points
        normal_trg = panel_trg.normal

        row_range = (offsets[j] + 1):offsets[j + 1]

        for (t_local, t_global) in enumerate(row_range)
            point_trg = points_trg[t_local]

            function integrand(x)
                u = x[1]
                v = x[2]
                p = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
                k = laplace3d_grad(p, point_trg, normal_trg) * scale
                return T(sigma(p)) * k
            end

            exact, _ = hcubature(integrand, T[-1, -1], T[1, 1]; atol = atol)

            direct = zero(T)
            @inbounds for s in 1:length(points_src)
                direct += sigma_src[s] * weights_src[s] * laplace3d_grad(points_src[s], point_trg, normal_trg)
            end

            out[t_global] += exact - direct
        end
    end

    return out
end

function laplace3d_DT_fmm3d_corrected_hcubature(
    interface::DielectricInterface{P, Float64},
    fmm_tol::Float64,
    hcubature_atol::Float64;
) where {P <: AbstractPanel}
    n_points = num_points(interface)
    D_base = laplace3d_DT_fmm3d(interface, fmm_tol)
    # hcubature integrates each near block adaptively, so only the near-pair
    # KEYS matter here; the upsample order (capped by max_order) is ignored.
    max_order = maximum(panel.n_quad for panel in interface.panels)
    (; upsample, adaptive) = build_neighbor_list(interface, max_order, hcubature_atol)
    @info "neighbor list: upsample=$(length(upsample)) adaptive=$(length(adaptive)) of $(length(interface.panels)^2)"
    corrections = laplace3d_DT_corrections_hcubature(interface, upsample, hcubature_atol)

    f = charges -> (D_base * charges) + (corrections * charges)
    return LinearMap{Float64}(f, n_points, n_points)
end

function laplace3d_pottrg_fmm3d_corrected_hcubature(
    interface::DielectricInterface{P, Float64},
    targets::Matrix{Float64},
    fmm_tol::Float64,
    hcubature_atol::Float64,
    range_factor::Float64;
) where {P <: AbstractPanel}
    @assert size(targets, 1) == 3

    n_points = num_points(interface)
    refined_interface = interface
    prolongation = sparse(1:n_points, 1:n_points, ones(Float64, n_points), n_points, n_points)
    if P <: FlatPanel{Float64, 3}
        panel_size_limit = minimum(_panel_max_length(panel) for panel in interface.panels)
        refined_interface, parent_ids, from_split = _refine_interface_for_targets(
            interface,
            targets,
            panel_size_limit;
            range_factor = range_factor,
        )
        prolongation = _refined_interface_prolongation(interface, refined_interface, parent_ids, from_split)
    end

    pot_base = laplace3d_pottrg_fmm3d(refined_interface, targets, fmm_tol)
    target_neighbor_list = build_target_neighbor_list(refined_interface, targets, false; range_factor = range_factor)

    @info "num of sources: $(num_points(interface)) → $(num_points(refined_interface))"
    if !isempty(target_neighbor_list)
        @info "num of hcub calculations: $(sum(length(v) for v in values(target_neighbor_list)))"
    end

    corrections = laplace3d_pottrg_corrections_hcubature(refined_interface, targets, target_neighbor_list, hcubature_atol)

    f = charges -> begin
        charges_refined = prolongation * charges
        return (pot_base * charges_refined) + (corrections * charges_refined)
    end
    return LinearMap{Float64}(f, size(targets, 2), n_points)
end
