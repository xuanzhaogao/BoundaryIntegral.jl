function legendre_p(n::Int, x)
    if n == 0
        return one(x)
    elseif n == 1
        return x
    end
    p_nm2 = one(x)
    p_nm1 = x
    for k in 2:n
        p_n = ((2 * k - 1) * x * p_nm1 - (k - 1) * p_nm2) / k
        p_nm2 = p_nm1
        p_nm1 = p_n
    end
    return p_nm1
end

function int_laplace3d_grad(n_quad::Int, n_quad_up::Int, panel::FlatPanel{T, 3}, trg::NTuple{3, T}) where T
    ns, ws = gausslegendre(n_quad_up)
    a, b, c, d = panel.corners
    cc = (a .+ b .+ c .+ d) ./ 4
    Lx = norm(b .- a)
    Ly = norm(c .- a)
    val = zeros(T, n_quad + 1, n_quad + 1)
    for i in 0:n_quad
        for j in 0:n_quad
            # f = (x, y) -> legendre_p(i, x) * legendre_p(j, y)
            f = (x, y) -> Pl(x, i) * Pl(y, j)
            val_ij = zero(T)
            for k in 1:n_quad_up
                for l in 1:n_quad_up
                    p = cc .+ (b .- a) .* (ns[k] / 2) .+ (d .- a) .* (ns[l] / 2)
                    val_ij += ws[k] * ws[l] * f(ns[k], ns[l]) * laplace3d_grad(p, trg, panel.normal) * Lx * Ly / 4
                end
            end
            val[i + 1, j + 1] = val_ij
        end
    end
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
        curr_val = int_laplace3d_grad(n_quad, n_quad_up + 1, panel, trg)
        err = norm(curr_val - prev_val, Inf)
        if err < atol
            return n_quad_up
        end
        prev_val = curr_val
        n_quad_up += 1
    end

    @warn "self-convergence error is > $(atol) at order $(max_order)"
    return max_order
end
