function legendre_basis_pl(ns::AbstractVector{T}, n_quad::Int) where T
    n_quad_up = length(ns)
    basis = Matrix{T}(undef, n_quad + 1, n_quad_up)
    for i in 0:n_quad
        for k in 1:n_quad_up
            basis[i + 1, k] = Pl(ns[k], i)
        end
    end
    return basis
end

function legendre_basis_recurrence(ns::AbstractVector{T}, n_quad::Int) where T
    n_quad_up = length(ns)
    basis = Matrix{T}(undef, n_quad + 1, n_quad_up)
    for k in 1:n_quad_up
        x = ns[k]
        basis[1, k] = one(T)
        if n_quad >= 1
            basis[2, k] = x
        end
        for i in 2:n_quad
            basis[i + 1, k] = ((2 * i - 1) * x * basis[i, k] - (i - 1) * basis[i - 1, k]) / i
        end
    end
    return basis
end

function int_laplace3d_grad(n_quad::Int, n_quad_up::Int, panel::FlatPanel{T, 3}, trg::NTuple{3, T}) where T
    ns, ws = gausslegendre(n_quad_up)
    a, b, c, d = panel.corners
    cc = (a .+ b .+ c .+ d) ./ 4
    bma = b .- a
    dma = d .- a
    Lx = norm(bma)
    Ly = norm(c .- a)
    scale = Lx * Ly / 4

    basis = legendre_basis_recurrence(ns, n_quad)

    weighted = Matrix{T}(undef, n_quad_up, n_quad_up)
    normal = panel.normal
    for k in 1:n_quad_up
        x = ns[k] / 2
        for l in 1:n_quad_up
            y = ns[l] / 2
            p = cc .+ bma .* x .+ dma .* y
            weighted[k, l] = ws[k] * ws[l] * laplace3d_grad(p, trg, normal) * scale
        end
    end

    temp = Matrix{T}(undef, n_quad + 1, n_quad_up)
    mul!(temp, basis, weighted)
    val = Matrix{T}(undef, n_quad + 1, n_quad + 1)
    mul!(val, temp, transpose(basis))
    return val
end

# check the upsampling order of quadrature points needed for a given panel and target point
function check_quad_order3d(panel::FlatPanel{T, 3}, trg::NTuple{3, T}, atol::T, max_order::Int) where T
    # number of original quadrature points in x and y directions
    # the range of basis functions are from P_1(x)P_1(y) to P_n(x)P_n(y)
    n_quad = panel.n_quad

    if n_quad >= max_order
        return n_quad
    end

    prev_val = int_laplace3d_grad(n_quad, n_quad, panel, trg)

    n_quad_up = n_quad
    while n_quad_up < max_order
        curr_val = int_laplace3d_grad(n_quad, 2 * n_quad_up, panel, trg)
        err = norm(curr_val - prev_val, Inf)
        if err < atol
            return n_quad_up
        end
        prev_val = curr_val
        n_quad_up *= 2
    end

    # @warn "self-convergence error is > $(atol) at order $(max_order)"
    return max_order
end
