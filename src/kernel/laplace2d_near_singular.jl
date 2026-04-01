# Near-field correction for 2D Laplace kernel when source panel is singular (Gauss-Jacobi).
#
# Strategy: represent the smooth part φ of σ(t) = (1+t)^β φ(t) in the Jacobi polynomial
# basis P_m^{(0,β)} orthogonal w.r.t. weight (1+t)^β.  The conversion from nodal values
# to Jacobi coefficients is stable (no matrix inversion):
#
#   a_m = Σ_j φ(t_j) P_m(t_j) w_j^{GJ} / ||P_m||²
#
# Then A_near = F * C where
#   C[m,j] = P_{m-1}(t_j) w_j^{GJ} / ||P_{m-1}||²          (nodal → Jacobi, n×n)
#   F[k,m] = ∫₋₁¹ K(s(t), x_k) (1+t)^β P_{m-1}(t) (L/2) dt   (smooth integrand)
#
# HCubature integrates the smooth F[k,:] for all m simultaneously per target.

# ---------------------------------------------------------------------------
# Jacobi polynomial helpers for P_m^{(0, beta)} on [-1,1]
# ---------------------------------------------------------------------------

# Evaluate P_0, …, P_{n-1}^{(0,beta)} at t via three-term recurrence.
# Writes into pre-allocated vector v of length n.
function _jacobi_poly_vals!(v::AbstractVector{T}, beta::T, t::T) where T
    n = length(v)
    n == 0 && return v
    v[1] = one(T)
    n == 1 && return v
    v[2] = ((beta + 2) * t - beta) / 2
    for m in 1:(n - 2)
        # Recurrence: P_{m+1} = (a·t + b)·P_m - c·P_{m-1}
        # with α=0, β=beta, degree index m:
        abm = T(2m) + beta          # 2m + 0 + beta
        mp1 = T(m + 1)
        mbp1 = T(m) + beta + 1      # m + 0 + beta + 1
        a_c = (abm + 1) * (abm + 2) / (2 * mp1 * mbp1)
        b_c = -(beta^2) * (abm + 1) / (2 * mp1 * mbp1 * abm)
        c_c = T(m) * (T(m) + beta) * (abm + 2) / (mp1 * mbp1 * abm)
        v[m + 2] = (a_c * t + b_c) * v[m + 1] - c_c * v[m]
    end
    return v
end

# Squared norms: ||P_m^{(0,beta)}||² = 2^{beta+1} / (2m + beta + 1)
function _jacobi_norms_sq(beta::T, n::Int) where T
    [T(2)^(beta + 1) / (2 * T(m) + beta + 1) for m in 0:(n - 1)]
end

# ---------------------------------------------------------------------------

"""
    laplace2d_near_singular_block(panel_src, panel_trg; rtol, atol)

Compute `(A_near, A_direct)` for a singular source panel and an arbitrary target panel.

* `A_direct[k,j] = K(s_j, x_k) * w_j`   (direct sum, FMM-equivalent)
* `A_near        = F * C`                 (Jacobi basis, one hcubature call per target)

where `C` maps nodal values → Jacobi coefficients and `F[k,m]` integrates
`K(s(t), x_k) · (1+t)^β · P_{m-1}(t) · (L/2)` over `[-1,1]`.
"""
function laplace2d_near_singular_block(
    panel_src::FlatPanel{T, 2},
    panel_trg::FlatPanel{T, 2};
    rtol::Real = 1e-10,
    atol::Real = 1e-14,
) where T
    @assert panel_src.is_singular "source panel must be singular"

    n_src = num_points(panel_src)
    n_trg = num_points(panel_trg)

    a_pt, b_pt = panel_src.corners[1], panel_src.corners[2]
    mid   = (a_pt .+ b_pt) ./ 2
    half  = (b_pt .- a_pt) ./ 2
    Lhalf = norm(b_pt .- a_pt) / 2
    beta  = panel_src.singular_exponent   # weight exponent (1+t)^beta

    gj_xs = panel_src.gl_xs   # GJ nodes on [-1,1]
    gj_ws = panel_src.gl_ws   # GJ weights: Σ_j f(t_j) w_j ≈ ∫ f(t)(1+t)^beta dt

    # ------------------------------------------------------------------
    # Conversion matrix C (n_src × n_src):
    #   C[m, j] = P_{m-1}(t_j) * gj_ws[j] / ||P_{m-1}||²
    # No matrix inversion — stable projection onto orthogonal Jacobi basis.
    # ------------------------------------------------------------------
    P_mat = Matrix{T}(undef, n_src, n_src)   # P_mat[j, m] = P_{m-1}(gj_xs[j])
    pv    = Vector{T}(undef, n_src)
    for j in 1:n_src
        _jacobi_poly_vals!(pv, beta, T(gj_xs[j]))
        P_mat[j, :] .= pv
    end
    h = _jacobi_norms_sq(beta, n_src)              # h[m] = ||P_{m-1}||²
    # C = diag(1/h) * P_mat' * diag(gj_ws)
    C = (P_mat' .* reshape(gj_ws, 1, n_src)) ./ h  # n_src × n_src

    # ------------------------------------------------------------------
    # A_direct: direct kernel sum (same as FMM would compute)
    # ------------------------------------------------------------------
    is_self = (panel_src === panel_trg)
    A_direct = zeros(T, n_trg, n_src)
    for k in 1:n_trg
        trg_pt  = panel_trg.points[k]
        trg_nrm = panel_trg.normal
        for j in 1:n_src
            is_self && k == j && continue   # diagonal: FMM produces zero
            A_direct[k, j] = laplace2d_grad(panel_src.points[j], trg_pt, trg_nrm) *
                              panel_src.weights[j]
        end
    end

    # ------------------------------------------------------------------
    # F (n_trg × n_src): one hcubature call per target point, returning
    # a vector of n_src Jacobi-basis integrals simultaneously.
    # ------------------------------------------------------------------
    F  = zeros(T, n_trg, n_src)
    lb = SVector{1, T}(-1)
    ub = SVector{1, T}( 1)

    for k in 1:n_trg
        trg_pt  = panel_trg.points[k]
        trg_nrm = panel_trg.normal
        pv_k    = Vector{T}(undef, n_src)   # buffer for Jacobi evals in this closure
        result, _ = hcubature(
            x -> begin
                t    = T(x[1])
                s_pt = mid .+ t .* half
                K    = laplace2d_grad(s_pt, trg_pt, trg_nrm)
                _jacobi_poly_vals!(pv_k, beta, t)
                K * (1 + t)^beta * Lhalf .* pv_k
            end,
            lb, ub; rtol = T(rtol), atol = T(atol),
        )
        F[k, :] .= result
    end

    A_near = F * C

    return A_near, A_direct
end

"""
    laplace2d_near_singular_corrections(interface; range_factor=3.0, rtol=1e-10, atol=1e-14)

For every singular source panel in `interface`, find nearby target panels
(centre-to-centre distance < `range_factor × source_panel_length`) and
compute `(delta_A, src_range, trg_range)` correction tuples.

`delta_A = A_near - A_direct` is the matrix that should be *added* to the
FMM result to recover the accurate near-field interaction.
"""
function laplace2d_near_singular_corrections(
    interface::DielectricInterface{FlatPanel{T, 2}, T};
    range_factor::Real = 3.0,
    rtol::Real = 1e-10,
    atol::Real = 1e-14,
) where T
    panels   = interface.panels
    n_panels = length(panels)

    offsets = Vector{Int}(undef, n_panels)
    off = 0
    for i in 1:n_panels
        offsets[i] = off
        off += num_points(panels[i])
    end

    function panel_centre(p::FlatPanel{T, 2})
        a, b = p.corners[1], p.corners[2]
        return (a .+ b) ./ 2
    end
    function panel_length(p::FlatPanel{T, 2})
        a, b = p.corners[1], p.corners[2]
        return norm(b .- a)
    end

    corrections = Vector{Tuple{Matrix{T}, UnitRange{Int}, UnitRange{Int}}}()

    for i in 1:n_panels
        panels[i].is_singular || continue
        c_src     = panel_centre(panels[i])
        threshold = T(range_factor) * panel_length(panels[i])
        src_range = (offsets[i] + 1):(offsets[i] + num_points(panels[i]))

        for j in 1:n_panels
            norm(panel_centre(panels[j]) .- c_src) < threshold || continue
            trg_range = (offsets[j] + 1):(offsets[j] + num_points(panels[j]))
            A_near, A_direct = laplace2d_near_singular_block(
                panels[i], panels[j]; rtol = rtol, atol = atol)
            push!(corrections, (A_near .- A_direct, src_range, trg_range))
        end
    end

    return corrections
end
