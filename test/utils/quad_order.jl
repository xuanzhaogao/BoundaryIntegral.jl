using BoundaryIntegral
using FastGaussQuadrature
using LinearAlgebra
import BoundaryIntegral as BI
using LegendrePolynomials
using Random
using Test

@testset "quad_order3d" begin
    ns, ws = gausslegendre(4)
    ns = Float64.(ns)
    ws = Float64.(ws)
    a = (-0.5, -0.5, 0.0)
    b = (0.5, -0.5, 0.0)
    c = (0.5, 0.5, 0.0)
    d = (-0.5, 0.5, 0.0)
    normal = (0.0, 0.0, 1.0)
    panel = BI.rect_panel3d_discretize(a, b, c, d, ns, ws, normal)
    trg = (0.1, -0.2, 0.25)

    function gaussian_integral_direct(nq::Int)
        xs, ws = gausslegendre(nq)
        xs = Float64.(xs)
        ws = Float64.(ws)
        cc = (a .+ b .+ c .+ d) ./ 4
        Lx = norm(b .- a)
        Ly = norm(d .- a)
        alpha = 2.0
        acc = 0.0
        for i in 1:nq
            for j in 1:nq
                p = cc .+ (b .- a) .* (xs[i] / 2) .+ (d .- a) .* (xs[j] / 2)
                rho = exp(-alpha * (p[1]^2 + p[2]^2 + p[3]^2))
                acc += ws[i] * ws[j] * rho * BI.laplace3d_pot(p, trg) * Lx * Ly / 4
            end
        end
        return acc
    end

    function gaussian_integral_upsampled(nq::Int)
        xs, ws = gausslegendre(nq)
        xs = Float64.(xs)
        ws = Float64.(ws)
        cc = (a .+ b .+ c .+ d) ./ 4
        Lx = norm(b .- a)
        Ly = norm(d .- a)
        alpha = 2.0

        ns0 = panel.gl_xs
        ws0 = panel.gl_ws
        n0 = length(ns0)
        rho0 = zeros(Float64, n0 * n0)
        idx = 1
        for j in 1:n0
            for i in 1:n0
                p = cc .+ (b .- a) .* (ns0[i] / 2) .+ (d .- a) .* (ns0[j] / 2)
                rho0[idx] = exp(-alpha * (p[1]^2 + p[2]^2 + p[3]^2))
                idx += 1
            end
        end

        E = BI.interp_matrix_2d_gl_tensor(ns0, ws0, ns0, ws0, xs, xs)
        rho_up = E * rho0

        acc = 0.0
        idx = 1
        for j in 1:nq
            for i in 1:nq
                p = cc .+ (b .- a) .* (xs[i] / 2) .+ (d .- a) .* (xs[j] / 2)
                acc += ws[i] * ws[j] * rho_up[idx] * BI.laplace3d_pot(p, trg) * Lx * Ly / 4
                idx += 1
            end
        end
        return acc
    end

    n_quad = panel.n_quad
    atol = 1e-6
    max_order = 16
    order = BI.check_quad_order3d(panel, trg, atol, max_order)
    @test order >= n_quad
    ref_val = gaussian_integral_direct(max_order)
    test_val = gaussian_integral_upsampled(order)
    @test abs(test_val - ref_val) <= 1e-4
end

@testset "quad_order3d_legendre_combo" begin
    function make_panel(n_quad, a, b, c, d)
        ns, ws = gausslegendre(n_quad)
        normal = (0.0, 0.0, 1.0)
        return BI.rect_panel3d_discretize(a, b, c, d, ns, ws, normal)
    end

    function combo_integral(panel, trg, n_quad_up, coeffs)
        ns, ws = gausslegendre(n_quad_up)
        a, b, c, d = panel.corners
        cc = (a .+ b .+ c .+ d) ./ 4
        Lx = norm(b .- a)
        Ly = norm(c .- a)

        acc = 0.0
        for k in 1:n_quad_up
            x = ns[k]
            for l in 1:n_quad_up
                y = ns[l]
                fxy = 0.0
                for i in 0:panel.n_quad
                    pix = Pl(x, i)
                    for j in 0:panel.n_quad
                        fxy += coeffs[i + 1, j + 1] * pix * Pl(y, j)
                    end
                end
                p = cc .+ (b .- a) .* (x / 2) .+ (d .- a) .* (y / 2)
                acc += ws[k] * ws[l] * fxy * BI.laplace3d_grad(p, trg, panel.normal) * Lx * Ly / 4
            end
        end
        return acc
    end

    panels = [
        ((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)),
        ((-0.6, -0.5, 0.0), (0.6, -0.5, 0.0), (0.6, 0.5, 0.0), (-0.6, 0.5, 0.0)),
    ]
    trgs = [
        (0.2, -0.15, 0.1),
        (0.3, 0.25, 0.5),
    ]

    rng = MersenneTwister(0)
    max_order = 128
    n_trials = 2

    for atol in (1e-4, 1e-6)
        err_tol = 2 * atol
        for (a, b, c, d) in panels
            for n_quad in (2, 4)
                panel = make_panel(n_quad, a, b, c, d)
                for trg in trgs
                    n_quad_up = BI.check_quad_order3d(panel, trg, atol, max_order)
                    n_ref = min(max_order * 2, max(n_quad_up * 4, n_quad_up))

                    M_up = BI.int_laplace3d_grad(panel.n_quad, n_quad_up, panel, trg)

                    errs = Float64[]
                    for _ in 1:n_trials
                        coeffs = randn(rng, panel.n_quad + 1, panel.n_quad + 1)
                        val_up = sum(M_up .* coeffs)
                        val_ref = combo_integral(panel, trg, n_ref, coeffs)
                        push!(errs, abs(val_up - val_ref))
                    end

                    @test maximum(errs) <= err_tol
                end
            end
        end
    end
end
