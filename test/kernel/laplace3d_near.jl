using BoundaryIntegral
import BoundaryIntegral as BI
using FastGaussQuadrature
using Test


@testset "laplace3d DT panel upsampled" begin
    ns, ws = gausslegendre(6)
    ns = Float64.(ns)
    ws = Float64.(ws)

    a = (-0.5, -0.5, 0.0)
    b = (0.5, -0.5, 0.0)
    c = (0.5, 0.5, 0.0)
    d = (-0.5, 0.5, 0.0)
    normal = (0.0, 0.0, 1.0)
    panel_src = BI.rect_panel3d_discretize(a, b, c, d, ns, ws, normal)

    a2 = (-0.6, -0.6, 0.4)
    b2 = (0.6, -0.6, 0.4)
    c2 = (0.6, 0.6, 0.4)
    d2 = (-0.6, 0.6, 0.4)
    panel_trg = BI.rect_panel3d_discretize(a2, b2, c2, d2, ns, ws, normal)

    function gaussian_density(points)
        rho = Vector{Float64}(undef, length(points))
        alpha = 0.5
        for i in 1:length(points)
            p = points[i]
            rho[i] = exp(-alpha * (p[1]^2 + p[2]^2 + p[3]^2))
        end
        return rho
    end

    n_ref = 48
    ns_ref, ws_ref = gausslegendre(n_ref)
    ns_ref = Float64.(ns_ref)
    ws_ref = Float64.(ws_ref)
    panel_ref = BI.rect_panel3d_discretize(a, b, c, d, ns_ref, ws_ref, normal)
    DT_ref = BI.laplace3d_DT_panel(panel_ref, panel_trg)
    rho_ref = gaussian_density(panel_ref.points)
    ref_val = DT_ref * rho_ref

    rho_src = gaussian_density(panel_src.points)
    trg_point = panel_trg.points[1]

    for atol in (1e-4, 1e-6, 1e-8)
        n_up = BI.check_quad_order3d(panel_src, trg_point, atol, 24)
        DT_up = BI.laplace3d_DT_panel_upsampled(panel_src, panel_trg, n_up)

        ns_up, ws_up = gausslegendre(n_up)
        ns_up = Float64.(ns_up)
        ws_up = Float64.(ws_up)
        panel_up = BI.rect_panel3d_discretize(a, b, c, d, ns_up, ws_up, normal)
        DT_direct = BI.laplace3d_DT_panel(panel_up, panel_trg)
        rho_up = gaussian_density(panel_up.points)

        err_up = norm(DT_up * rho_src - ref_val, Inf)
        err_direct = norm(DT_direct * rho_up - ref_val, Inf)

        @test err_up <= 40 * atol
        @test err_direct <= 40 * atol
    end
end

@testset "laplace3d DT panel direct integral" begin
    ns, ws = gausslegendre(5)
    ns = Float64.(ns)
    ws = Float64.(ws)

    a = (-0.4, -0.2, 0.1)
    b = (0.6, -0.2, 0.1)
    c = (0.6, 0.8, 0.1)
    d = (-0.4, 0.8, 0.1)
    normal_src = (0.0, 0.0, 1.0)
    panel_src = BI.rect_panel3d_discretize(a, b, c, d, ns, ws, normal_src)

    a2 = (-0.6, -0.4, 0.7)
    b2 = (0.7, -0.4, 0.7)
    c2 = (0.7, 0.9, 0.7)
    d2 = (-0.6, 0.9, 0.7)
    normal_trg = (0.0, 0.0, 1.0)
    panel_trg = BI.rect_panel3d_discretize(a2, b2, c2, d2, ns, ws, normal_trg)

    function density(points)
        rho = Vector{Float64}(undef, length(points))
        for i in 1:length(points)
            p = points[i]
            rho[i] = 0.3 + p[1] - 0.5 * p[2] + 0.2 * p[3]
        end
        return rho
    end

    rho_src = density(panel_src.points)
    DT_panel = BI.laplace3d_DT_panel(panel_src, panel_trg)
    vals_panel = DT_panel * rho_src

    np_trg = BI.num_points(panel_trg)
    vals_direct = zeros(Float64, np_trg)
    for (j, trg_point) in enumerate(BI.eachpoint(panel_trg))
        acc = 0.0
        for (i, src_point) in enumerate(BI.eachpoint(panel_src))
            acc += BI.laplace3d_grad(src_point.point, trg_point.point, trg_point.normal) * panel_src.weights[i] * rho_src[i]
        end
        vals_direct[j] = acc
    end

    @test norm(vals_panel - vals_direct, Inf) < 1e-12
end


@testset "build_neighbor_list" begin
    ns, ws = gausslegendre(2)
    ns = Float64.(ns)
    ws = Float64.(ws)

    normal = (0.0, 0.0, 1.0)
    p1 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, 0.5, 0.0),
        (-0.5, 0.5, 0.0),
        ns,
        ws,
        normal,
    )
    p2 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.2),
        (0.5, -0.5, 0.2),
        (0.5, 0.5, 0.2),
        (-0.5, 0.5, 0.2),
        ns,
        ws,
        normal,
    )
    p3 = BI.rect_panel3d_discretize(
        (1.0, 1.0, 1.0),
        (1.02, 1.0, 1.0),
        (1.02, 1.02, 1.0),
        (1.0, 1.02, 1.0),
        ns,
        ws,
        normal,
    )
    p4 = BI.rect_panel3d_discretize(
        (1.0, 1.0, 1.2),
        (1.02, 1.0, 1.2),
        (1.02, 1.02, 1.2),
        (1.0, 1.02, 1.2),
        ns,
        ws,
        normal,
    )

    interface = BI.DielectricInterface([p1, p2, p3, p4], fill(2.0, 4), fill(1.0, 4))

    max_order = 12
    atol = 1e-3
    neighbors = BI.build_neighbor_list(interface, max_order, atol, true, true)

    center_p2 = (p2.corners[1] .+ p2.corners[2] .+ p2.corners[3] .+ p2.corners[4]) ./ 4
    center_p1 = (p1.corners[1] .+ p1.corners[2] .+ p1.corners[3] .+ p1.corners[4]) ./ 4

    l1 = max(norm(p1.corners[1] .- p1.corners[2]), norm(p1.corners[2] .- p1.corners[3]))
    l2 = max(norm(p2.corners[1] .- p2.corners[2]), norm(p2.corners[2] .- p2.corners[3]))
    r1 = 5 * l1 / p1.n_quad
    r2 = 5 * l2 / p2.n_quad

    order_12 = p1.n_quad
    for pt in p2.points
        norm(pt .- center_p1) <= r1 || continue
        order_12 = max(order_12, BI.check_quad_order3d(p1, pt, atol, max_order))
    end

    order_21 = p2.n_quad
    for pt in p1.points
        norm(pt .- center_p2) <= r2 || continue
        order_21 = max(order_21, BI.check_quad_order3d(p2, pt, atol, max_order))
    end

    @test order_12 > p1.n_quad
    @test order_21 > p2.n_quad
    @test neighbors[(1, 2)] == order_12
    @test neighbors[(2, 1)] == order_21
    @test !haskey(neighbors, (3, 4))
    @test !haskey(neighbors, (4, 3))
end

@testset "build_neighbor_list edge filter" begin
    ns, ws = gausslegendre(2)
    ns = Float64.(ns)
    ws = Float64.(ws)

    normal = (0.0, 0.0, 1.0)
    p1 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, 0.5, 0.0),
        (-0.5, 0.5, 0.0),
        ns,
        ws,
        normal;
        is_edge = true,
    )
    p2 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.2),
        (0.5, -0.5, 0.2),
        (0.5, 0.5, 0.2),
        (-0.5, 0.5, 0.2),
        ns,
        ws,
        normal;
        is_edge = true,
    )

    interface = BI.DielectricInterface([p1, p2], fill(2.0, 2), fill(1.0, 2))
    max_order = 12
    atol = 1e-3

    neighbors_all = BI.build_neighbor_list(interface, max_order, atol, true, true)
    @test haskey(neighbors_all, (1, 2))
    @test haskey(neighbors_all, (2, 1))

    neighbors_skip = BI.build_neighbor_list(interface, max_order, atol, false, false)
    @test isempty(neighbors_skip)
end

@testset "build_neighbor_list same surface skip" begin
    ns, ws = gausslegendre(2)
    ns = Float64.(ns)
    ws = Float64.(ws)

    normal = (0.0, 0.0, 1.0)
    p1 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.0),
        (0.0, -0.5, 0.0),
        (0.0, 0.0, 0.0),
        (-0.5, 0.0, 0.0),
        ns,
        ws,
        normal,
    )
    p2 = BI.rect_panel3d_discretize(
        (0.1, 0.1, 0.0),
        (0.6, 0.1, 0.0),
        (0.6, 0.6, 0.0),
        (0.1, 0.6, 0.0),
        ns,
        ws,
        normal,
    )

    interface = BI.DielectricInterface([p1, p2], fill(2.0, 2), fill(1.0, 2))
    neighbors = BI.build_neighbor_list(interface, 12, 1e-3, true, true)
    @test isempty(neighbors)
end

@testset "build_neighbor_list varquad interface" begin
    rhs(p, n) = 1.0
    interface = BI.single_dielectric_box3d_rhs_adaptive_varquad(
        1.0,
        1.0,
        1.0,
        4,
        rhs,
        0.3,
        1e-6,
        2.0,
        1.0,
        Float64;
        max_depth = 2,
        n_quad_min = 2,
    )

    neighbors = BI.build_neighbor_list(interface, 1, 1e-6, true, true; distance_only = true, range_factor = 10.0)
    @test !isempty(neighbors)
    for ((i, _), n_up) in neighbors
        @test n_up == interface.panels[i].n_quad
    end
end

@testset "laplace3d_DT_correction_term" begin
    ns, ws = gausslegendre(2)
    ns = Float64.(ns)
    ws = Float64.(ws)

    normal = (0.0, 0.0, 1.0)
    p1 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, 0.5, 0.0),
        (-0.5, 0.5, 0.0),
        ns,
        ws,
        normal,
    )
    p2 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.2),
        (0.5, -0.5, 0.2),
        (0.5, 0.5, 0.2),
        (-0.5, 0.5, 0.2),
        ns,
        ws,
        normal,
    )
    p3 = BI.rect_panel3d_discretize(
        (1.0, 1.0, 1.0),
        (1.02, 1.0, 1.0),
        (1.02, 1.02, 1.0),
        (1.0, 1.02, 1.0),
        ns,
        ws,
        normal,
    )
    p4 = BI.rect_panel3d_discretize(
        (1.0, 1.0, 1.2),
        (1.02, 1.0, 1.2),
        (1.02, 1.02, 1.2),
        (1.0, 1.02, 1.2),
        ns,
        ws,
        normal,
    )

    interface = BI.DielectricInterface([p1, p2, p3, p4], fill(2.0, 4), fill(1.0, 4))
    neighbors = BI.build_neighbor_list(interface, 12, 1e-3, true, true)
    corrections = BI.laplace3d_DT_corrections(interface, neighbors)

    cnt = [length(p1.points), length(p2.points), length(p3.points), length(p4.points)]
    offsets = cumsum(vcat(0, cnt))
    total_n = offsets[end]

    rho = zeros(Float64, total_n)
    for i in 1:total_n
        rho[i] = sin(0.1 * i)
    end

    expected = zeros(Float64, total_n)

    n12 = neighbors[(1, 2)]
    n21 = neighbors[(2, 1)]

    block_12 = BI.laplace3d_DT_panel_upsampled(p1, p2, n12) - BI.laplace3d_DT_panel(p1, p2)
    block_21 = BI.laplace3d_DT_panel_upsampled(p2, p1, n21) - BI.laplace3d_DT_panel(p2, p1)

    src1 = (offsets[1] + 1):offsets[2]
    src2 = (offsets[2] + 1):offsets[3]
    trg1 = src1
    trg2 = src2

    expected[trg2] .+= block_12 * rho[src1]
    expected[trg1] .+= block_21 * rho[src2]

    got = corrections * rho

    @test norm(got - expected, Inf) < 1e-12
    @test norm(got[(offsets[3] + 1):offsets[5]], Inf) == 0.0
end

@testset "laplace3d_DT_panel_upsampled_ordering" begin
    ns, ws = gausslegendre(3)
    ns = Float64.(ns)
    ws = Float64.(ws)

    normal = (0.0, 0.0, 1.0)
    p1 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, 0.5, 0.0),
        (-0.5, 0.5, 0.0),
        ns,
        ws,
        normal,
    )
    p2 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.3),
        (0.5, -0.5, 0.3),
        (0.5, 0.5, 0.3),
        (-0.5, 0.5, 0.3),
        ns,
        ws,
        normal,
    )

    direct = BI.laplace3d_DT_panel(p1, p2)
    upsampled = BI.laplace3d_DT_panel_upsampled(p1, p2, p1.n_quad)

    @test norm(upsampled - direct, Inf) < 1e-12
end

@testset "laplace3d_DT_fmm3d_corrected" begin
    ns, ws = gausslegendre(2)
    ns = Float64.(ns)
    ws = Float64.(ws)

    normal = (0.0, 0.0, 1.0)
    p1 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, 0.5, 0.0),
        (-0.5, 0.5, 0.0),
        ns,
        ws,
        normal,
    )
    p2 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.2),
        (0.5, -0.5, 0.2),
        (0.5, 0.5, 0.2),
        (-0.5, 0.5, 0.2),
        ns,
        ws,
        normal,
    )
    p3 = BI.rect_panel3d_discretize(
        (1.0, 1.0, 1.0),
        (1.02, 1.0, 1.0),
        (1.02, 1.02, 1.0),
        (1.0, 1.02, 1.0),
        ns,
        ws,
        normal,
    )
    p4 = BI.rect_panel3d_discretize(
        (1.0, 1.0, 1.2),
        (1.02, 1.0, 1.2),
        (1.02, 1.02, 1.2),
        (1.0, 1.02, 1.2),
        ns,
        ws,
        normal,
    )

    interface = BI.DielectricInterface([p1, p2, p3, p4], fill(2.0, 4), fill(1.0, 4))

    tol = 1e-12
    max_order = 12
    corrected = BI.laplace3d_DT_fmm3d_corrected(interface, tol, tol, max_order)
    neighbors = BI.build_neighbor_list(interface, max_order, tol, true, true)
    corrections = BI.laplace3d_DT_corrections(interface, neighbors)
    base = BI.laplace3d_DT_fmm3d(interface, tol)

    n = BI.num_points(interface)
    rho = zeros(Float64, n)
    for i in 1:n
        rho[i] = cos(0.2 * i)
    end

    expected = base * rho + corrections * rho
    got = corrected * rho

    @test norm(got - expected, Inf) < 1e-10
end

@testset "laplace3d_DT_fmm3d_corrected_hcubature" begin
    ns, ws = gausslegendre(2)
    ns = Float64.(ns)
    ws = Float64.(ws)

    normal = (0.0, 0.0, 1.0)
    p1 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, 0.5, 0.0),
        (-0.5, 0.5, 0.0),
        ns,
        ws,
        normal,
    )
    p2 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.2),
        (0.5, -0.5, 0.2),
        (0.5, 0.5, 0.2),
        (-0.5, 0.5, 0.2),
        ns,
        ws,
        normal,
    )
    p3 = BI.rect_panel3d_discretize(
        (1.0, 1.0, 1.0),
        (1.02, 1.0, 1.0),
        (1.02, 1.02, 1.0),
        (1.0, 1.02, 1.0),
        ns,
        ws,
        normal,
    )
    p4 = BI.rect_panel3d_discretize(
        (1.0, 1.0, 1.2),
        (1.02, 1.0, 1.2),
        (1.02, 1.02, 1.2),
        (1.0, 1.02, 1.2),
        ns,
        ws,
        normal,
    )

    interface = BI.DielectricInterface([p1, p2, p3, p4], fill(2.0, 4), fill(1.0, 4))

    fmm_tol = 1e-8
    up_tol = 1e-8
    max_order = 256
    corrected_hcub = BI.laplace3d_DT_fmm3d_corrected_hcubature(interface, fmm_tol, up_tol, 5.0)
    corrected_up = BI.laplace3d_DT_fmm3d_corrected(interface, fmm_tol, up_tol, max_order)
    direct = BI.laplace3d_DT(interface)
    direct[diagind(direct)] .= 0.0

    n = BI.num_points(interface)
    rho = zeros(Float64, n)
    for i in 1:n
        rho[i] = cos(0.2 * i)
    end

    @test norm(corrected_hcub * rho - corrected_up * rho, Inf) < 1e-8
end

@testset "laplace3d_DT_corrections_hcubature_apply" begin
    ns, ws = gausslegendre(2)
    ns = Float64.(ns)
    ws = Float64.(ws)

    normal = (0.0, 0.0, 1.0)
    p1 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, 0.5, 0.0),
        (-0.5, 0.5, 0.0),
        ns,
        ws,
        normal,
    )
    p2 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.2),
        (0.5, -0.5, 0.2),
        (0.5, 0.5, 0.2),
        (-0.5, 0.5, 0.2),
        ns,
        ws,
        normal,
    )

    interface = BI.DielectricInterface([p1, p2], fill(2.0, 2), fill(1.0, 2))
    atol = 1e-8
    neighbors = BI.build_neighbor_list(interface, 1, atol, true, true; distance_only = true, range_factor = 5.0)
    corrections = BI.laplace3d_DT_corrections_hcubature(interface, neighbors, atol)

    sigma(p) = p[1] + 2 * p[2] - p[3]

    rho = zeros(Float64, BI.num_points(interface))
    for (i, point) in enumerate(BI.eachpoint(interface))
        rho[i] = sigma(point.panel_point.point)
    end

    expected = corrections * rho
    got = BI.laplace3d_DT_corrections_hcubature_apply(interface, neighbors, atol, sigma)

    @test norm(got - expected, Inf) < 1e-8
end

@testset "laplace3d_D_fmm3d_corrected" begin
    ns, ws = gausslegendre(2)
    ns = Float64.(ns)
    ws = Float64.(ws)

    normal = (0.0, 0.0, 1.0)
    p1 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, 0.5, 0.0),
        (-0.5, 0.5, 0.0),
        ns,
        ws,
        normal,
    )
    p2 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.2),
        (0.5, -0.5, 0.2),
        (0.5, 0.5, 0.2),
        (-0.5, 0.5, 0.2),
        ns,
        ws,
        normal,
    )
    p3 = BI.rect_panel3d_discretize(
        (1.0, 1.0, 1.0),
        (1.02, 1.0, 1.0),
        (1.02, 1.02, 1.0),
        (1.0, 1.02, 1.0),
        ns,
        ws,
        normal,
    )
    p4 = BI.rect_panel3d_discretize(
        (1.0, 1.0, 1.2),
        (1.02, 1.0, 1.2),
        (1.02, 1.02, 1.2),
        (1.0, 1.02, 1.2),
        ns,
        ws,
        normal,
    )

    interface = BI.DielectricInterface([p1, p2, p3, p4], fill(2.0, 4), fill(1.0, 4))

    tol = 1e-12
    max_order = 12
    corrected = BI.laplace3d_D_fmm3d_corrected(interface, tol, tol, max_order)
    neighbors = BI.build_neighbor_list(interface, max_order, tol, true, true)
    corrections = BI.laplace3d_D_corrections(interface, neighbors)
    base = BI.laplace3d_D_fmm3d(interface, tol)

    n = BI.num_points(interface)
    rho = zeros(Float64, n)
    for i in 1:n
        rho[i] = cos(0.2 * i)
    end

    expected = base * rho + corrections * rho
    got = corrected * rho

    @test norm(got - expected, Inf) < 1e-10
end
