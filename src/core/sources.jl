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
