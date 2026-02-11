function Lhs_dielectric_box3d(interface::DielectricInterface{P, T}) where {P <: AbstractPanel, T}
    D_transpose = laplace3d_DT(interface)
    Lhs = D_transpose
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

function Lhs_dielectric_box3d_fmm3d(interface::DielectricInterface{P, T}, tol::Float64 = 1e-6) where {P <: AbstractPanel, T}
    D_transpose = laplace3d_DT_fmm3d(interface, tol)

    function g(x)
        Dx = D_transpose * x

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

# linear operator for the corrected DT kernel
function Lhs_dielectric_box3d_fmm3d_corrected(
    interface::DielectricInterface{P, Float64},
    fmm_tol::Float64,
    up_tol::Float64,
    max_order::Int;
    include_edges_src::Bool = true,
    include_edges_trg::Bool = true,
) where {P <: AbstractPanel}
    n_points = num_points(interface)
    D_transpose = laplace3d_DT_fmm3d_corrected(
        interface,
        fmm_tol,
        up_tol,
        max_order;
        include_edges_src = include_edges_src,
        include_edges_trg = include_edges_trg,
    )

    function apply_operator(charges::AbstractVector{Float64})
        y = D_transpose * charges

        offset = 0
        for i in 1:length(interface.panels)
            panel = interface.panels[i]
            eps_in = interface.eps_in[i]
            eps_out = interface.eps_out[i]
            n_pts = num_points(panel)
            t = 0.5 * (eps_out + eps_in) / (eps_out - eps_in)
            for j in 1:n_pts
                y[offset + j] += t * charges[offset + j]
            end
            offset += n_pts
        end

        return y
    end

    return LinearMap{Float64}(apply_operator, n_points, n_points)
end


function Rhs_dielectric_box3d(interface::DielectricInterface{P, T}, ps::PointSource{T, 3}, eps_src::T) where {P <: AbstractPanel, T}
    src = ps.point
    q = ps.charge
    n_points = num_points(interface)
    Rhs = zeros(T, n_points)
    for (i, point) in enumerate(eachpoint(interface))
        Rhs[i] = - q * laplace3d_grad(src, point.panel_point.point, point.panel_point.normal) / eps_src
    end
    return Rhs
end

function Rhs_dielectric_box3d(interface::DielectricInterface{P, T}, vs::VolumeSource{T, 3}, eps_src::T) where {P <: AbstractPanel, T}
    xs, ys, zs = vs.axes
    weights = vs.weights
    density = vs.density
    n_points = num_points(interface)
    Rhs = zeros(T, n_points)
    for (i, point) in enumerate(eachpoint(interface))
        acc = zero(T)
        for ix in eachindex(xs), iy in eachindex(ys), iz in eachindex(zs)
            pos = (xs[ix], ys[iy], zs[iz])
            acc += weights[ix, iy, iz] * density[ix, iy, iz] *
                laplace3d_grad(pos, point.panel_point.point, point.panel_point.normal)
        end
        Rhs[i] = -acc / eps_src
    end
    return Rhs
end

function Rhs_dielectric_box3d_fmm3d(
    interface::DielectricInterface{P, Float64},
    vs::VolumeSource{Float64, 3},
    eps_src::Float64,
    thresh::Float64,
) where {P <: AbstractPanel}
    xs, ys, zs = vs.axes
    weights = vs.weights
    density = vs.density
    nx, ny, nz = length(xs), length(ys), length(zs)
    n_sources = nx * ny * nz
    sources = zeros(Float64, 3, n_sources)
    charges = zeros(Float64, n_sources)
    idx = 0
    for ix in 1:nx, iy in 1:ny, iz in 1:nz
        idx += 1
        sources[1, idx] = xs[ix]
        sources[2, idx] = ys[iy]
        sources[3, idx] = zs[iz]
        charges[idx] = weights[ix, iy, iz] * density[ix, iy, iz]
    end

    n_points = num_points(interface)
    targets = zeros(Float64, 3, n_points)
    normals = zeros(Float64, 3, n_points)
    for (i, point) in enumerate(eachpoint(interface))
        targets[1, i] = point.panel_point.point[1]
        targets[2, i] = point.panel_point.point[2]
        targets[3, i] = point.panel_point.point[3]
        normals[1, i] = point.panel_point.normal[1]
        normals[2, i] = point.panel_point.normal[2]
        normals[3, i] = point.panel_point.normal[3]
    end

    vals = lfmm3d(thresh, sources, charges = charges, targets = targets, pgt = 2)
    grad = vals.gradtarg
    Rhs = zeros(Float64, n_points)
    for i in 1:n_points
        Rhs[i] = dot(normals[:, i], grad[:, i]) / (4π * eps_src)
    end
    return Rhs
end
