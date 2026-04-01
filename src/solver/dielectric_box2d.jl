function lhs_dielectric_box2d(interface::DielectricInterface{P, T}) where {P <: AbstractPanel, T}
    D_transpose = laplace2d_DT(interface)
    Lhs = D_transpose
    has_singular = any(p -> p.is_singular, interface.panels)
    if has_singular
        corrections = laplace2d_near_singular_corrections(interface)
        for (delta_A, src_range, trg_range) in corrections
            Lhs[trg_range, src_range] .+= delta_A
        end
    end
    offset = 0
    for i in 1:length(interface.panels)
        panel = interface.panels[i]
        eps_in = interface.eps_in[i]
        eps_out = interface.eps_out[i]
        n_pts = num_points(panel)
        t = 0.5 * (eps_out + eps_in) / (eps_out - eps_in)
        for j in 1:n_pts
            Lhs[offset + j, offset + j] += t
        end
        offset += n_pts
    end
    return Lhs
end
const Lhs_dielectric_box2d = lhs_dielectric_box2d # backward compat

function lhs_dielectric_box2d_fmm2d(interface::DielectricInterface{P, T}, tol::Float64 = 1e-12) where {P <: AbstractPanel, T}
    D_transpose = laplace2d_DT_fmm2d(interface, tol)
    has_singular = any(p -> p.is_singular, interface.panels)
    corrections = has_singular ? laplace2d_near_singular_corrections(interface) : nothing

    function g(x)
        Dx = D_transpose * x
        if corrections !== nothing
            for (delta_A, src_range, trg_range) in corrections
                Dx[trg_range] .+= delta_A * x[src_range]
            end
        end

        offset = 0
        for i in 1:length(interface.panels)
            panel = interface.panels[i]
            eps_in = interface.eps_in[i]
            eps_out = interface.eps_out[i]
            n_pts = num_points(panel)
            t = 0.5 * (eps_out + eps_in) / (eps_out - eps_in)
            for j in 1:n_pts
                Dx[offset + j] += t * x[offset + j]
            end
            offset += n_pts
        end

        return Dx
    end

    Lhs = LinearMap{T}(g, num_points(interface), num_points(interface))

    return Lhs
end
const Lhs_dielectric_box2d_fmm2d = lhs_dielectric_box2d_fmm2d # backward compat

function rhs_dielectric_box2d(interface::DielectricInterface{P, T}, ps::PointSource{T}, eps_src::T) where {P <: AbstractPanel, T}
    src = ps.point
    q = ps.charge
    n_points = num_points(interface)
    Rhs = zeros(T, n_points)
    for (i, point) in enumerate(eachpoint(interface))
        Rhs[i] = - q * laplace2d_grad(src, point.panel_point.point, point.panel_point.normal) / eps_src
    end
    return Rhs
end
const Rhs_dielectric_box2d = rhs_dielectric_box2d # backward compat
