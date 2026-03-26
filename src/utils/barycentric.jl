# --------------------------
# 1D: barycentric weights for Gauss–Legendre nodes
# λ_i = (-1)^(n-i) * sqrt((1-x_i^2) w_i / 2)  (up to a global scaling)
# x must be sorted ascending and wG in the same order.
# --------------------------
function gl_barycentric_weights(x::AbstractVector{T}, wG::AbstractVector) where T
    n = length(x)
    TF = float(T)
    λ = similar(x, TF)
    for i in 1:n
        s = iseven(n - i) ? one(TF) : -one(TF)
        λ[i] = s * sqrt((1 - x[i]^2) * wG[i] / 2)
    end
    # optional normalization (does not change the interpolant)
    λ ./= maximum(abs, λ)
    return λ
end

function barycentric_row!(r::AbstractVector, x::AbstractVector, λ::AbstractVector, xq::Real)
    n = length(x)
    T = eltype(r)

    @inbounds for i in 1:n
        if isapprox(xq, x[i])
            fill!(r, zero(T))
            r[i] = one(T)
            return r
        end
    end

    denom = zero(T)
    @inbounds for i in 1:n
        t = T(λ[i]) / (T(xq) - T(x[i]))
        r[i] = t
        denom += t
    end
    inv_denom = one(T) / denom
    @inbounds for i in 1:n
        r[i] *= inv_denom
    end
    return r
end

function barycentric_row(x::AbstractVector, λ::AbstractVector, xq::Real)
    T = float(promote_type(eltype(x), eltype(λ), typeof(xq)))
    r = Vector{T}(undef, length(x))
    return barycentric_row!(r, x, λ, xq)
end

# --------------------------
# Lagrange-to-monomial coefficient matrix
# C[j, k] = coefficient of x^(k-1) in L_j(x)
# Computed as C = (V^T)^{-1} where V is the Vandermonde matrix V[i,k] = x_i^(k-1)
# --------------------------
function lagrange_mono_coeffs(x::AbstractVector{T}) where T
    n = length(x)
    TF = float(T)
    # Build Vandermonde matrix: V[i,k] = x[i]^(k-1)
    V = Matrix{TF}(undef, n, n)
    @inbounds for i in 1:n
        V[i, 1] = one(TF)
        for k in 2:n
            V[i, k] = V[i, k-1] * TF(x[i])
        end
    end
    # C = (V^{-1})^T, since L_j(x_i) = δ_{ij} means C * V^T = I
    C = Matrix{TF}(inv(V)')
    return C
end

# Evaluate all n Lagrange basis functions at xq using Horner's method
# with precomputed monomial coefficients C[j,k] = coeff of x^(k-1) in L_j
function eval_lagrange_horner!(r::AbstractVector, C::Matrix, xq::Real)
    n = size(C, 1)
    T = eltype(r)
    xq_T = T(xq)
    @inbounds for j in 1:n
        val = T(C[j, n])
        for k in (n-1):-1:1
            val = muladd(val, xq_T, T(C[j, k]))
        end
        r[j] = val
    end
    return r
end

function interp_matrix_1d_gl(x::AbstractVector, wG::AbstractVector, xq::AbstractVector)
    λ = gl_barycentric_weights(x, wG)
    m, n = length(xq), length(x)
    E = zeros(Float64, m, n)
    for j in 1:m
        E[j, :] .= barycentric_row(x, λ, xq[j])
    end
    return E
end

function interp_matrix_2d_gl_scattered(x::AbstractVector, wx::AbstractVector,
                                       y::AbstractVector, wy::AbstractVector,
                                       xt::AbstractVector, yt::AbstractVector)
    @assert length(xt) == length(yt)
    nx, ny = length(x), length(y)
    Nt = length(xt)

    λx = gl_barycentric_weights(x, wx)
    λy = gl_barycentric_weights(y, wy)

    A = zeros(Float64, Nt, nx * ny)
    for k in 1:Nt
        rx = barycentric_row(x, λx, xt[k])
        ry = barycentric_row(y, λy, yt[k])
        A[k, :] .= kron(ry, rx)   # vec(F) is column-major: i + (j-1)*nx
    end
    return A
end

function interp_matrix_2d_gl_tensor(x::AbstractVector, wx::AbstractVector,
                                    y::AbstractVector, wy::AbstractVector,
                                    Xout::AbstractVector, Yout::AbstractVector)
    Ex = interp_matrix_1d_gl(x, wx, Xout)
    Ey = interp_matrix_1d_gl(y, wy, Yout)
    return kron(Ey, Ex)
end