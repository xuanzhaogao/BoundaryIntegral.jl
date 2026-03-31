# Near-field correction for 2D Laplace kernel when source panel is singular (Gauss-Jacobi).
#
# The GJ weights already absorb the (1+t)^exponent factor, so the FMM-equivalent
# "direct" interaction is  A_direct[k,j] = K(s_j, t_k) * w_j.
# For nearby targets the kernel varies too rapidly for FMM accuracy, so we
# recompute A_near by adaptive quadrature over the reference interval and
# return the correction  delta_A = A_near - A_direct.

"""
    laplace2d_near_singular_block(panel_src, panel_trg; rtol=1e-10, atol=1e-14)

Compute `(A_near, A_direct)` for a singular source panel and an arbitrary target panel.

* `A_direct[k,j] = K(s_j, t_k) * w_j`  (FMM-equivalent matrix)
* `A_near[k,j]  = ∫₋₁¹ K(s(t), t_k) · (1+t)^exponent · L_j(t) · (L/2) dt`

where `s(t)` maps reference `[-1,1]` to the physical source panel, `L_j` is the
j-th Lagrange basis on the GJ nodes, and `exponent = panel_src.singular_exponent`.
"""
function laplace2d_near_singular_block(
    panel_src::FlatPanel{T,2},
    panel_trg::FlatPanel{T,2};
    rtol::Real = 1e-10,
    atol::Real = 1e-14,
) where T
    @assert panel_src.is_singular "source panel must be singular"

    n_src = num_points(panel_src)
    n_trg = num_points(panel_trg)

    a, b = panel_src.corners[1], panel_src.corners[2]
    mid   = (a .+ b) ./ 2
    half  = (b .- a) ./ 2
    L     = norm(b .- a)
    Lhalf = L / 2
    exponent = panel_src.singular_exponent   # gamma - 1

    # Precompute Lagrange monomial coefficients on GJ reference nodes
    C = lagrange_mono_coeffs(panel_src.gl_xs)
    Lj_buf = Vector{T}(undef, n_src)

    # Detect self-interaction (same panel object)
    is_self = (panel_src === panel_trg)

    # --- A_direct ---
    A_direct = zeros(T, n_trg, n_src)
    for k in 1:n_trg
        trg_pt  = panel_trg.points[k]
        trg_nrm = panel_trg.normal
        for j in 1:n_src
            # Skip diagonal for self-interaction (FMM produces zero for src == trg)
            is_self && k == j && continue
            A_direct[k, j] = laplace2d_grad(panel_src.points[j], trg_pt, trg_nrm) * panel_src.weights[j]
        end
    end

    # --- A_near via adaptive quadrature ---
    A_near = Matrix{T}(undef, n_trg, n_src)
    for k in 1:n_trg
        trg_pt  = panel_trg.points[k]
        trg_nrm = panel_trg.normal
        for j in 1:n_src
            function integrand(t)
                # Map reference coord to physical point
                s_pt = mid .+ t .* half
                K = laplace2d_grad(s_pt, trg_pt, trg_nrm)
                # Evaluate j-th Lagrange basis at t
                eval_lagrange_horner!(Lj_buf, C, t)
                return K * (1 + t)^exponent * Lj_buf[j] * Lhalf
            end
            val, _ = hquadrature(integrand, T(-1), T(1); rtol = T(rtol), atol = T(atol))
            A_near[k, j] = val
        end
    end

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
    interface::DielectricInterface{FlatPanel{T,2},T};
    range_factor::Real = 3.0,
    rtol::Real = 1e-10,
    atol::Real = 1e-14,
) where T
    panels = interface.panels
    n_panels = length(panels)

    # Build global DOF offsets: panel i owns indices  offset[i]+1 : offset[i]+num_points(panels[i])
    offsets = Vector{Int}(undef, n_panels)
    off = 0
    for i in 1:n_panels
        offsets[i] = off
        off += num_points(panels[i])
    end

    # Panel centres and half-lengths
    function panel_centre(p::FlatPanel{T,2}) where T
        a, b = p.corners[1], p.corners[2]
        return (a .+ b) ./ 2
    end
    function panel_length(p::FlatPanel{T,2}) where T
        a, b = p.corners[1], p.corners[2]
        return norm(b .- a)
    end

    corrections = Vector{Tuple{Matrix{T}, UnitRange{Int}, UnitRange{Int}}}()

    for i in 1:n_panels
        panels[i].is_singular || continue
        c_src = panel_centre(panels[i])
        L_src = panel_length(panels[i])
        threshold = T(range_factor) * L_src

        src_range = (offsets[i] + 1):(offsets[i] + num_points(panels[i]))

        for j in 1:n_panels
            c_trg = panel_centre(panels[j])
            dist = norm(c_trg .- c_src)
            dist < threshold || continue

            trg_range = (offsets[j] + 1):(offsets[j] + num_points(panels[j]))
            A_near, A_direct = laplace2d_near_singular_block(panels[i], panels[j]; rtol = rtol, atol = atol)
            delta_A = A_near .- A_direct
            push!(corrections, (delta_A, src_range, trg_range))
        end
    end

    return corrections
end
