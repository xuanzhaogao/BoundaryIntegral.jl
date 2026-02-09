# sources contribute to the right hand side of the problem
abstract type AbstractSource end

# simple Point Source
struct PointSource{T, D} <: AbstractSource
    point::NTuple{D, T}
    charge::T
end

Base.show(io::IO, s::PointSource{T, D}) where {T, D} = print(io, "PointSource at $(s.point) with charge $(s.charge) in $T")

# may extend to other types of sources later, not sure about the structure

function rhs_approx(interface::DielectricInterface{FlatPanel{T, 3}, T}, ps::PointSource{T, 3}, eps_src::T; tol::T = sqrt(eps(T))) where T
    rhs(p, n) = -ps.charge * laplace3d_grad(ps.point, p, n) / eps_src
    return rhs_approx(interface, rhs; tol = tol)
end


struct VolumeSource{T, D} <: AbstractSource
    points::Vector{NTuple{D, T}}
    weights::Vector{T}
    density::Vector{T}
end

function GaussianVolumeSource(center::NTuple{3, T}, σ::T, n::Int, tol::T) where T
    @assert n >= 1 "n must be >= 1"
    @assert σ > zero(T) "σ must be > 0"
    @assert tol > zero(T) && tol < one(T) "tol must be in (0, 1)"

    ns, ws = gausslegendre(n)
    points = Vector{NTuple{3, T}}(undef, n^3)
    weights = Vector{T}(undef, n^3)
    density = Vector{T}(undef, n^3)

    idx = 1
    two_sigma2 = T(2) * σ * σ
    support_r = sqrt(two_sigma2 * log(inv(tol)))
    norm_factor = inv((sqrt(T(2) * T(pi)) * σ)^3)
    for i in 1:n
        xi = center[1] + support_r * T(ns[i])
        wi = support_r * T(ws[i])
        for j in 1:n
            yj = center[2] + support_r * T(ns[j])
            wj = support_r * T(ws[j])
            for k in 1:n
                zk = center[3] + support_r * T(ns[k])
                wk = support_r * T(ws[k])
                points[idx] = (xi, yj, zk)
                weights[idx] = wi * wj * wk
                r2 = (xi - center[1])^2 + (yj - center[2])^2 + (zk - center[3])^2
                density[idx] = norm_factor * exp(-r2 / two_sigma2)
                idx += 1
            end
        end
    end

    return VolumeSource{T, 3}(points, weights, density)
end
