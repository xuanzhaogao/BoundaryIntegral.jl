using BoundaryIntegral
import BoundaryIntegral as BI
using Random
using Test

# @testset "box3d rhs adaptive" begin
#     rhs_const(p, n) = 1.0
#     interface = BI.single_dielectric_box3d_rhs_adaptive(
#         1.0,
#         1.0,
#         1.0,
#         2,
#         rhs_const,
#         2.0,
#         1e-8,
#         2.0,
#         1.0,
#         Float64;
#         max_depth = 3,
#     )
#     @test length(interface.panels) == 6

#     ps = BI.PointSource((0.1, 0.1, 0.1), 1.0)
#     interface_ps = BI.single_dielectric_box3d_rhs_adaptive(
#         1.0,
#         1.0,
#         1.0,
#         2,
#         ps,
#         1.0,
#         2.0,
#         1e-8,
#         2.0,
#         1.0,
#         Float64;
#         max_depth = 2,
#     )
#     @test length(interface_ps.panels) >= 6

#     rhs_poly(p, n) = p[1]^4 + p[2]^4 + p[3]^4
#     interface_refined = BI.single_dielectric_box3d_rhs_adaptive(
#         1.0,
#         1.0,
#         1.0,
#         2,
#         rhs_poly,
#         0.3,
#         1e-8,
#         2.0,
#         1.0,
#         Float64;
#         max_depth = 2,
#     )
#     @test length(interface_refined.panels) > 6

#     interface_refined_ec = BI.single_dielectric_box3d_rhs_adaptive(
#         1.0,
#         1.0,
#         1.0,
#         2,
#         rhs_const,
#         0.2,
#         1e-8,
#         2.0,
#         1.0,
#         Float64;
#         max_depth = 1,
#     )
#     @test length(interface_refined_ec.panels) > 6
# end

@testset "box3d rhs adaptive accuracy" begin
    rng = MersenneTwister(1234)
    ps = BI.PointSource((0.15, -0.12, 0.08), 1.0)
    eps_src = 2.0
    rhs(p, n) = -ps.charge * BI.laplace3d_grad(ps.point, p, n) / eps_src
    rhs_atol = 1e-4
    interface = BI.single_dielectric_box3d_rhs_adaptive(
        1.0,
        1.0,
        1.0,
        4,
        rhs,
        0.3,
        rhs_atol,
        2.0,
        1.0,
        Float64;
        max_depth = 4,
    )

    n_panels_sample = min(10, length(interface.panels))
    panel_indices = rand(rng, 1:length(interface.panels), n_panels_sample)
    rhs_panel = BI.rhs_approx(interface, ps, eps_src; tol = 1e-8)
    rhs_vals = [rhs(point.panel_point.point, point.panel_point.normal) for point in BI.eachpoint(interface)]
    rhs_panel_vals = BI.interface_approx(interface, rhs_vals; tol = 1e-8)
    n_points = 20
    for idx in panel_indices
        panel = interface.panels[idx]
        a, b, c, d = panel.corners
        cc = (a .+ b .+ c .+ d) ./ 4
        bma = b .- a
        dma = d .- a

        max_err = 0.0
        for _ in 1:n_points
            u = rand(rng) * 2 - 1
            v = rand(rng) * 2 - 1
            p = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
            exact = rhs(p, panel.normal)
            approx = rhs_panel(p)
            approx_vals = rhs_panel_vals(p)
            max_err = max(max_err, abs(exact - approx_vals))
            max_err = max(max_err, abs(exact - approx))
        end
        @test max_err <= rhs_atol
    end
end

@testset "box3d geometry helper" begin
    quads, normals = BI._box3d_face_quads(1.0, 2.0, 3.0)
    @test length(quads) == 6
    @test length(normals) == 6
    @test normals[1] == (0.0, 0.0, 1.0)
    @test normals[2] == (0.0, 0.0, -1.0)
end

@testset "volume source rhs adaptive uses batched eval" begin
    xs = [-0.25, 0.0, 0.25]
    ys = [-0.25, 0.0, 0.25]
    zs = [-0.25, 0.0, 0.25]
    weights = fill(1.0, 3, 3, 3)
    density = fill(1.0, 3, 3, 3)
    vs = BI.VolumeSource{Float64, 3}((xs, ys, zs), weights, density)
    eps_src = 1.0

    function rhs_volume(p, n)
        acc = 0.0
        for i in 1:length(xs), j in 1:length(ys), k in 1:length(zs)
            pos = (xs[i], ys[j], zs[k])
            acc += weights[i, j, k] * density[i, j, k] * BI.laplace3d_grad(pos, p, n)
        end
        return -acc / eps_src
    end

    a = (0.5, 0.5, 0.0)
    b = (-0.5, 0.5, 0.0)
    c = (-0.5, -0.5, 0.0)
    d = (0.5, -0.5, 0.0)
    normal = (0.0, 0.0, 1.0)
    tpl = BI.TempPanel3D(a, b, c, d, false, false, false, false, false, false, false, false, normal)

    n_quad = 3
    rhs_atol = 1e-6
    ns, ws = BI.gausslegendre(n_quad)

    resolved_direct = BI.rhs_panel3d_resolved(tpl, rhs_volume, n_quad, rhs_atol)
    resolved_fmm = BI._rhs_panel3d_resolved_volume_fmm([tpl], vs, eps_src, ns, ws, rhs_atol, rhs_atol * 0.1)[1]

    @test resolved_fmm || !resolved_direct
end

@testset "volume source rhs adaptive batches all faces" begin
    alpha = sqrt(2.0)
    n_divide_x, n_divide_y = BI.best_grid_mn(1.0, 1.0, alpha)
    panels = BI._box3d_rhs_adaptive_initial_panels(1.0, 1.0, 1.0, alpha)
    @test length(panels) == 6 * n_divide_x * n_divide_y
end

@testset "box3d rhs adaptive varquad accuracy" begin
    rng = MersenneTwister(4321)
    ps = BI.PointSource((0.12, -0.08, 0.05), 1.0)
    eps_src = 2.0
    rhs(p, n) = -ps.charge * BI.laplace3d_grad(ps.point, p, n) / eps_src
    rhs_atol = 1e-4
    n_quad_max = 6
    n_quad_min = 2
    interface = BI.single_dielectric_box3d_rhs_adaptive_varquad(
        1.0,
        1.0,
        1.0,
        n_quad_max,
        rhs,
        0.3,
        rhs_atol,
        2.0,
        1.0,
        Float64;
        max_depth = 4,
        n_quad_min = n_quad_min,
    )

    for panel in interface.panels
        @test panel.n_quad <= n_quad_max
        @test panel.n_quad >= n_quad_min
    end

    n_panels_sample = min(8, length(interface.panels))
    panel_indices = rand(rng, 1:length(interface.panels), n_panels_sample)
    rhs_panel = BI.rhs_approx(interface, ps, eps_src; tol = 1e-8)
    rhs_vals = [rhs(point.panel_point.point, point.panel_point.normal) for point in BI.eachpoint(interface)]
    rhs_panel_vals = BI.interface_approx(interface, rhs_vals; tol = 1e-8)
    n_points = 12
    for idx in panel_indices
        panel = interface.panels[idx]
        a, b, c, d = panel.corners
        cc = (a .+ b .+ c .+ d) ./ 4
        bma = b .- a
        dma = d .- a

        max_err = 0.0
        for _ in 1:n_points
            u = rand(rng) * 2 - 1
            v = rand(rng) * 2 - 1
            p = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
            exact = rhs(p, panel.normal)
            approx = rhs_panel(p)
            approx_vals = rhs_panel_vals(p)
            max_err = max(max_err, abs(exact - approx_vals))
            max_err = max(max_err, abs(exact - approx))
        end
        @test max_err <= rhs_atol
    end
end

@testset "box3d rhs adaptive volume source" begin
    rng = MersenneTwister(2024)
    xs = [4.5, 5.0]
    ys = [4.5, 5.0]
    zs = [4.5, 5.0]
    weights = fill(1.0, 2, 2, 2)
    density = fill(1.0, 2, 2, 2)
    vs = BI.VolumeSource{Float64, 3}((xs, ys, zs), weights, density)
    eps_src = 2.0

    function rhs_volume(p, n)
        acc = 0.0
        for i in 1:length(xs), j in 1:length(ys), k in 1:length(zs)
            pos = (xs[i], ys[j], zs[k])
            acc += weights[i, j, k] * density[i, j, k] * BI.laplace3d_grad(pos, p, n) / eps_src
        end
        return -acc
    end

    rhs_atol = 1e-4
    interface = BI.single_dielectric_box3d_rhs_adaptive(
        1.0,
        1.0,
        1.0,
        4,
        vs,
        eps_src,
        0.3,
        rhs_atol,
        2.0,
        1.0,
        Float64;
        max_depth = 4,
    )

    n_panels_sample = min(8, length(interface.panels))
    panel_indices = rand(rng, 1:length(interface.panels), n_panels_sample)
    rhs_vals = [rhs_volume(point.panel_point.point, point.panel_point.normal) for point in BI.eachpoint(interface)]
    rhs_panel_vals = BI.interface_approx(interface, rhs_vals; tol = 1e-8)
    n_points = 12
    for idx in panel_indices
        panel = interface.panels[idx]
        a, b, c, d = panel.corners
        cc = (a .+ b .+ c .+ d) ./ 4
        bma = b .- a
        dma = d .- a

        max_err = 0.0
        for _ in 1:n_points
            u = rand(rng) * 2 - 1
            v = rand(rng) * 2 - 1
            p = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
            exact = rhs_volume(p, panel.normal)
            approx_vals = rhs_panel_vals(p)
            max_err = max(max_err, abs(exact - approx_vals))
        end
        @test max_err <= rhs_atol
    end
end
