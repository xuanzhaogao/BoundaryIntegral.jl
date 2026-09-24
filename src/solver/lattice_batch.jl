# src/solver/lattice_batch.jl
# Lattice-scale batches: frame-TRANSLATED orbital instances on a virtual global grid.
# Unlike a periodic circshift on the template grid, instances
# here carry an integer global-frame offset and never wrap — so >5 distinct cells per
# direction are representable and pair products are exact on frame intersections.

# Canonical BoxGeom type (dielectric box geometry). Defined here because lattice_batch.jl
# is the first includer that needs it; campaign/toml_input.jl references it downstream.
const BoxGeom = NamedTuple{(:center, :Lx, :Ly, :Lz),
    Tuple{NTuple{3,Float64}, Float64, Float64, Float64}}

"""
    lattice_grid_steps(datagrid, primvec, n::NTuple{3,Int}) -> NTuple{3,Int}

Integer grid-step offset of the lattice translation `n1·a1 + n2·a2 + n3·a3` (the
global-frame offset of a translated orbital instance). Errors if a lattice vector is
not grid-commensurate. The offset is interpreted as a frame translation, not a
circshift.
"""
lattice_grid_steps(datagrid, primvec::AbstractMatrix, n::NTuple{3,Int}) =
    _lattice_grid_shift(datagrid, primvec, n)

"""
    OrbitalInstance(id, template_id, steps)

One orbital of a lattice campaign: template `template_id` translated by the integer
global-frame offset `steps` (grid steps; see `lattice_grid_steps`).
"""
struct OrbitalInstance
    id::Int
    template_id::Int
    steps::NTuple{3,Int}
end

# Per-axis overlap of two frames of length n offset by integer steps si, sj.
# Returns the GLOBAL index range covered by both (local_i = g - si, local_j = g - sj),
# or `nothing` if disjoint.
function _frame_overlap(n::Int, si::Int, sj::Int)
    glo = max(si, sj) + 1
    ghi = min(si, sj) + n
    glo > ghi && return nothing
    return glo:ghi
end

"""
    LatticeBatch

A batch of pair densities on the union of their supports, indexed on the VIRTUAL GLOBAL
GRID (`gidx` = integer grid coordinates in the template frame). All columns share
`gidx`/`positions`/`weights` — required for the nd-batched FMM.
"""
struct LatticeBatch
    pair_ids::Vector{Tuple{Int,Int}}     # (orbital id i, orbital id j) per column
    gidx::Vector{NTuple{3,Int}}          # n shared global grid indices (sorted)
    positions::Matrix{Float64}           # 3 × n
    weights::Vector{Float64}             # n (uniform: |det cell| / (nx ny nz))
    densities::Matrix{Float64}           # n × K raw (unscreened) pair densities
    # Primitive sampling-cell basis of the virtual global grid, for Eq. (3.3)'s h.
    lattice_basis::Union{Nothing, NTuple{3, NTuple{3, Float64}}}
end

num_pairs(b::LatticeBatch) = length(b.pair_ids)

"""
    assemble_lattice_batch(templates, instances, pairs; support_rtol=1e-6) -> LatticeBatch

Assemble the pair densities `rho_ij = phi_i * phi_j` for an explicit pair list. Each
orbital is `templates[instance.template_id]` translated by `instance.steps`; products
are exact pointwise multiplies on the integer frame intersection. The batch lives on
the union of the per-pair supports, then truncated by the group envelope
(rss across columns ≥ `support_rtol` × its max — same rule as `assemble_rhs_group`).
All templates must share one grid geometry.
"""
function assemble_lattice_batch(templates::AbstractVector,
        instances::AbstractDict{Int,OrbitalInstance},
        pairs::Vector{Tuple{Int,Int}}; support_rtol::Real = 1e-6)
    isempty(pairs) && error("empty batch")
    t1 = templates[1]
    for t in templates
        datagrids_compatible(t1, t) || error("all templates must share one grid geometry")
    end
    nx, ny, nz = t1.nx, t1.ny, t1.nz
    K = length(pairs)

    row = Dict{NTuple{3,Int},Int}()
    gidx = NTuple{3,Int}[]
    cols = Vector{Vector{Tuple{Int,Float64}}}(undef, K)
    for (k, (i, j)) in enumerate(pairs)
        oi = instances[i]; oj = instances[j]
        vi = templates[oi.template_id].values
        vj = templates[oj.template_id].values
        si = oi.steps; sj = oj.steps
        rx = _frame_overlap(nx, si[1], sj[1])
        ry = _frame_overlap(ny, si[2], sj[2])
        rz = _frame_overlap(nz, si[3], sj[3])
        vals = Tuple{Int,Float64}[]
        if rx !== nothing && ry !== nothing && rz !== nothing
            for gz in rz, gy in ry, gx in rx          # gx innermost: values are i-fastest
                v = vi[gx - si[1], gy - si[2], gz - si[3]] *
                    vj[gx - sj[1], gy - sj[2], gz - sj[3]]
                v == 0.0 && continue
                g = (gx, gy, gz)
                r = get!(row, g) do
                    push!(gidx, g)
                    length(gidx)
                end
                push!(vals, (r, v))
            end
        end
        cols[k] = vals
    end

    n = length(gidx)
    densities = zeros(Float64, n, K)
    for k in 1:K, (r, v) in cols[k]
        densities[r, k] = v
    end

    # union-support truncation (same rule as assemble_rhs_group)
    keep = if support_rtol > 0 && n > 0
        env = vec(sqrt.(sum(abs2, densities; dims = 2)))
        m = maximum(env)
        m > 0 ? findall(>=(support_rtol * m), env) : collect(1:n)
    else
        collect(1:n)
    end
    gk = gidx[keep]
    perm = sortperm(gk)                                # deterministic order
    gk = gk[perm]
    dk = densities[keep, :][perm, :]

    m = length(gk)
    positions = Matrix{Float64}(undef, 3, m)
    for s in 1:m
        p = grid_point(t1, gk[s][1], gk[s][2], gk[s][3])   # affine: valid for any ints
        positions[1, s] = p[1]; positions[2, s] = p[2]; positions[3, s] = p[3]
    end
    At, Bt, Ct = true_cell_vectors(t1)
    w = abs(det(hcat(collect(At), collect(Bt), collect(Ct)))) / (nx * ny * nz)
    # primitive step vectors: the cell vectors divided by the grid counts
    lb = ((At[1] / nx, At[2] / nx, At[3] / nx),
          (Bt[1] / ny, Bt[2] / ny, Bt[3] / ny),
          (Ct[1] / nz, Ct[2] / nz, Ct[3] / nz))
    return LatticeBatch(copy(pairs), gk, positions, fill(w, m), dk, lb)
end

"""
    envelope_volume_source(b::LatticeBatch)

Per-point rss of the batch densities, as a VolumeSource (drives envelope refinement).
"""
function envelope_volume_source(b::LatticeBatch)
    n = size(b.densities, 1)
    env = Vector{Float64}(undef, n)
    @inbounds for s in 1:n
        acc = 0.0
        for k in 1:size(b.densities, 2)
            acc += b.densities[s, k]^2
        end
        env[s] = sqrt(acc)
    end
    return VolumeSource(copy(b.positions), copy(b.weights), env; lattice_basis = b.lattice_basis)
end

"""
    batch_volume_sources(b::LatticeBatch) -> Vector{VolumeSource{Float64,3}}

Split a LatticeBatch into the Vector{VolumeSource} core form (shared positions).
"""
function batch_volume_sources(b::LatticeBatch)
    return VolumeSource{Float64, 3}[
        VolumeSource(copy(b.positions), copy(b.weights), b.densities[:, k];
                     lattice_basis = b.lattice_basis)
        for k in 1:num_pairs(b)
    ]
end

"""
    evaluate_batch_potential(interface, Σ, sources, targets;
        lhs_tol, volume_tol, c_pad=5.0, range_factor=5.0) -> Φ (n_targets × K)

Total potential `Φ_a = u_inc[ρ_a] + u[σ_a]` of a solved batch at arbitrary targets.

- `u[σ_a]`: corrected layer-potential map (FMM + hcubature near correction) built ONCE
  for the target set, applied per column of Σ.
- `u_inc[rho_a]`: batch-level near/far split on the Section 3 padded near region
  B_pad (Eq. 3.9), `h_n = c_pad * ||A_rho||_2`. Near targets: TKM via
  PrecomputedVolumeField per source, on the paper's Fourier geometry
  (Eqs. 3.11, 3.16). Far targets: ONE `nd = K` point-charge FMM (potential/4pi)
  over the screened quadrature points.

`c_pad` is DIMENSIONLESS, a multiple of the lattice spacing; default 5.0.

All `sources` must share identical grid points (the batch convention); this is checked.

Conventions match `four_index_matrix`: TKM returns the `1/(4π|r|)`-normalised potential
and is used without further scaling; the scattered layer potential is the output of
`laplace3d_pottrg_fmm3d_corrected_hcubature` applied to each column of Σ.
"""
function evaluate_batch_potential(interface, Σ::AbstractMatrix,
        sources::Vector{<:VolumeSource{Float64, 3}}, targets::Matrix{Float64};
        lhs_tol::Float64, volume_tol::Float64, c_pad::Float64 = 5.0,
        range_factor::Float64 = 5.0,
        screen_boxes::Union{Nothing, Vector{<:NamedTuple}} = nothing,
        screen_epses::Union{Nothing, Vector{Float64}} = nothing,
        screen_eps_out::Float64 = 1.0)
    K = length(sources)
    nt = size(targets, 2)
    size(targets, 1) == 3 || throw(ArgumentError("targets must be 3 × n"))
    size(Σ, 2) == K || throw(DimensionMismatch("Σ columns ≠ number of sources"))

    # shared-positions contract: all sources must be on the same grid points
    for s in 2:K
        sources[s].positions == sources[1].positions ||
            throw(ArgumentError("evaluate_batch_potential requires all sources on identical grid points"))
    end

    # hoist screening: one pass over all sources. Default interface-based (single eps region);
    # with `screen_boxes` given, screen box-based (multi-region heterojunction interface).
    screened = screen_boxes === nothing ?
        [screened_volume_source(interface, vs, SharpScreening()) for vs in sources] :
        [screened_volume_source(screen_boxes, screen_epses, screen_eps_out, vs, SharpScreening()) for vs in sources]

    Φ = Matrix{Float64}(undef, nt, K)

    # scattered part: build the corrected map once, apply per column
    pottrg = laplace3d_pottrg_fmm3d_corrected_hcubature(interface, targets, lhs_tol, lhs_tol, range_factor)
    for a in 1:K
        Φ[:, a] = pottrg * Σ[:, a]
    end

    # incident part: Section 3 near/far split (Eq. 3.9) on the screened source.
    # screened_volume_source preserves positions, so the geometry is identical for
    # every column and is built once.
    geom = near_field_geometry(screened[1]; c_pad = c_pad)
    near_idx = findall(i -> in_near_region(geom, targets, i), 1:nt)
    far_idx = setdiff(1:nt, near_idx)

    # near targets: TKM via PrecomputedVolumeField, one column at a time.
    # ltkm3dc cannot be used here: it derives its own truncation radius from the
    # combined source+target box (TKM3D src/continuous.jl:133-146) and exposes no
    # hook for Eq. (3.11)'s L. Building one field per column and discarding it
    # keeps peak memory at a single coefficient array, matching the old call.
    if !isempty(near_idx)
        near_targets = targets[:, near_idx]
        for a in 1:K
            fld = PrecomputedVolumeField(screened[a];
                tol = volume_tol, c_pad = c_pad, compute_grad = false)
            Φ[near_idx, a] .+= volume_field_potential(fld, near_targets)
        end
    end

    # far targets: one nd=K batched point-charge FMM, potential / (4π)
    if !isempty(far_idx)
        far_targets = targets[:, far_idx]
        # screened sources share positions; build nd=K charge matrix
        sa1 = screened[1]
        n_src = length(sa1.weights)
        charges = Matrix{Float64}(undef, K, n_src)
        for a in 1:K
            sa = screened[a]
            @inbounds for s in 1:n_src
                charges[a, s] = sa.weights[s] * sa.density[s]
            end
        end
        vals = lfmm3d(volume_tol, sa1.positions;
            charges = charges, targets = far_targets, pgt = 1, nd = K)
        # vals.pottarg is (K, n_far) for nd=K
        pt = reshape(vals.pottarg, K, length(far_idx))
        for a in 1:K
            Φ[far_idx, a] .+= pt[a, :] ./ (4π)
        end
    end

    return Φ
end

"""
    solve_dielectric_lattice_batch(boxes, epses, eps_out, b::LatticeBatch; kw...)
        -> (; sigma, interface, sources, stats)

Solve an explicit-pair batch: ONE shared interface refined on the batch envelope,
then block GMRES over all pair densities.
"""
function solve_dielectric_lattice_batch(boxes::Vector{BoxGeom}, epses::Vector{Float64},
        eps_out::Float64, b::LatticeBatch;
        n_quad::Int, rhs_atol::Float64, l_ec::Float64,
        fmm_tol::Float64, up_tol::Float64 = fmm_tol, max_order::Int = 8,
        gmres_rtol::Float64, max_depth::Int = 128, itmax::Int = 500,
        precondition::Bool = true, correct_edges::Bool = true)
    env = envelope_volume_source(b)
    interface = multi_dielectric_box3d_rhs_adaptive(
        n_quad, l_ec, boxes, epses, env, rhs_atol;
        eps_out = eps_out, max_depth = max_depth)
    sources = batch_volume_sources(b)
    # box-based multi-region screening: the interface may span several eps regions
    # (heterojunction substrate), where interface-based source screening is undefined.
    Σ, stats = solve_dielectric_box3d_block(interface, sources;
        fmm_tol = fmm_tol, up_tol = up_tol, max_order = max_order,
        rtol = gmres_rtol, itmax = itmax, precondition = precondition,
        correct_edges = correct_edges,
        screen_boxes = boxes, screen_epses = epses, screen_eps_out = eps_out)
    return (; sigma = Σ, interface = interface, sources = sources, stats = stats)
end
