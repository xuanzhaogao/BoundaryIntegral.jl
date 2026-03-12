using BoundaryIntegral
import BoundaryIntegral as BI
using LinearAlgebra
using Random
using Test

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
    h = BI._estimate_source_spacing(vs)
    tkm_kmax = BI._estimate_tkm3dc_kmax(h)

    resolved_direct = BI.rhs_panel3d_resolved(tpl, rhs_volume, n_quad, rhs_atol)
    resolved_fmm = BI._rhs_panel3d_resolved_volume_fmm([tpl], vs, eps_src, ns, ws, rhs_atol, rhs_atol * 0.1, h, tkm_kmax)[1]

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

@testset "hybrid FMM+TKM rhs evaluation" begin
    # Volume source near the z=+0.5 face triggers the near-field TKM path.
    center = (0.0, 0.0, 0.45)
    σ = 0.05
    vs = BI.GaussianVolumeSource(center, σ, 8, 1e-6)
    eps_src = 2.0

    interface = BI.single_dielectric_box3d_rhs_adaptive(
        1.0, 1.0, 1.0, 4, vs, eps_src, 0.3, 1e-4, 4.0, 1.0, Float64;
        max_depth = 3,
    )

    @test BI.num_points(interface) > 0
    @test length(interface.panels) > 0

    # Verify helper functions work
    h = BI._estimate_source_spacing(vs)
    @test h > 0.0

    panels_init = BI._box3d_rhs_adaptive_initial_panels(1.0, 1.0, 1.0, sqrt(2.0))
    is_near = BI._classify_near_far_panels(panels_init, vs, h)
    @test any(is_near)   # source is near z=+0.5 face
    @test !all(is_near)  # source is far from z=-0.5 face
end

@testset "hybrid rhs sparse-source near/far fallback" begin
    # Single-source VolumeSource should still produce a positive near-field radius.
    vs_sparse = BI.VolumeSource([(0.0, 0.0, 0.45)], [1e-3], [1.0])
    h = BI._estimate_source_spacing(vs_sparse)
    @test h > 0.0

    panels_init = BI._box3d_rhs_adaptive_initial_panels(1.0, 1.0, 1.0, sqrt(2.0))
    is_near = BI._classify_near_far_panels(panels_init, vs_sparse, h)
    @test any(is_near)
    @test !all(is_near)
end

@testset "hybrid rhs refinement uses pointwise target classification" begin
    center = (0.0, 0.0, 0.45)
    σ = 0.05
    vs = BI.GaussianVolumeSource(center, σ, 8, 1e-6)
    h = BI._estimate_source_spacing(vs)
    ns, ws = BI.gausslegendre(4)
    panels = BI._box3d_rhs_adaptive_initial_panels(4.0, 4.0, 1.0, sqrt(2.0))

    targets, normals, n_per_panel = BI._rhs_panel3d_refinement_targets(panels, ns, ws)

    @test size(targets, 1) == 3
    @test size(normals, 1) == 3
    @test size(targets, 2) == length(panels) * n_per_panel
    @test size(normals, 2) == size(targets, 2)

    is_near_panel = BI._classify_near_far_panels(panels, vs, h)
    is_near_target = BI._classify_near_far_targets(targets, vs, h)

    near_panel_ids = findall(is_near_panel)
    @test !isempty(near_panel_ids)
    near_counts = Int[]
    for panel_idx in near_panel_ids
        panel_range = ((panel_idx - 1) * n_per_panel + 1):(panel_idx * n_per_panel)
        n_near_targets = count(view(is_near_target, panel_range))
        push!(near_counts, n_near_targets)
    end
    @test any(0 < n_near_targets < n_per_panel for n_near_targets in near_counts)
end

@testset "hybrid rhs refinement routes only pointwise-near targets to TKM" begin
    center = (0.0, 0.0, 0.45)
    σ = 0.05
    vs = BI.GaussianVolumeSource(center, σ, 8, 1e-6)
    h = BI._estimate_source_spacing(vs)
    ns, ws = BI.gausslegendre(4)
    panels = BI._box3d_rhs_adaptive_initial_panels(4.0, 4.0, 1.0, sqrt(2.0))
    targets, normals, n_per_panel = BI._rhs_panel3d_refinement_targets(panels, ns, ws)
    expected_is_near = BI._classify_near_far_targets(targets, vs, h)
    expected_near = count(expected_is_near)
    expected_far = size(targets, 2) - expected_near
    tkm_kmax = BI._estimate_tkm3dc_kmax(h)

    @test 0 < expected_near < size(targets, 2)
    @test_logs (:info, Regex("near targets: $expected_near, far targets: $expected_far")) begin
        resolved = BI._rhs_panel3d_resolved_volume_fmm(panels, vs, 2.0, ns, ws, 1e-4, 1e-5, h, tkm_kmax)
        @test length(resolved) == length(panels)
    end
end

@testset "hybrid rhs backend scaling consistency" begin
    center = (0.0, 0.0, 0.05)
    σ = 0.08
    eps_src = 2.0
    vs = BI.GaussianVolumeSource(center, σ, 16, 1e-9)
    sources, charges = BI._volume_source_fmm_sources(vs)
    tkm_kmax = BI._estimate_tkm3dc_kmax(vs)

    xs = collect(range(-0.2, 0.2; length = 7))
    nt = length(xs)^2
    targets = Matrix{Float64}(undef, 3, nt)
    normals = Matrix{Float64}(undef, 3, nt)
    idx = 1
    for x in xs, y in xs
        targets[:, idx] .= (x, y, 0.0)
        normals[:, idx] .= (0.0, 0.0, 1.0)
        idx += 1
    end

    rhs_far, _, _ = BI._rhs_volume_targets_hybrid(
        sources, charges, targets, normals, eps_src, 1e-9, tkm_kmax, fill(false, nt),
    )
    rhs_near, _, _ = BI._rhs_volume_targets_hybrid(
        sources, charges, targets, normals, eps_src, 1e-9, tkm_kmax, fill(true, nt),
    )

    mean_abs_far = sum(abs.(rhs_far)) / nt
    mean_abs_near = sum(abs.(rhs_near)) / nt
    ratio = mean_abs_near / max(mean_abs_far, eps(Float64))
    rel_diff = norm(rhs_near .- rhs_far) / max(norm(rhs_far), eps(Float64))
    @test 0.7 <= ratio <= 1.3
    @test rel_diff <= 0.2
end

@testset "hybrid rhs convergence on gaussian-crossing plane" begin
    center = (0.0, 0.0, 0.05)
    σ = 0.1
    eps_src = 2.0
    vs = BI.GaussianVolumeSource(center, σ, 20, 1e-9)
    sources, charges = BI._volume_source_fmm_sources(vs)
    base_kmax = BI._estimate_tkm3dc_kmax(vs)

    xs = collect(range(-1.1, 1.1; length = 9))
    nt = length(xs)^2
    targets = Matrix{Float64}(undef, 3, nt)
    normals = Matrix{Float64}(undef, 3, nt)
    points = Vector{NTuple{3, Float64}}(undef, nt)
    idx = 1
    for x in xs, y in xs
        p = (x, y, 0.0)
        targets[:, idx] .= p
        normals[:, idx] .= (0.0, 0.0, 1.0)
        points[idx] = p
        idx += 1
    end

    # Direct quadrature reference for the same discrete Gaussian source:
    # this isolates hybrid-evaluation error from source discretization error.
    rhs_ref = Vector{Float64}(undef, nt)
    for i in 1:nt
        acc = 0.0
        p = points[i]
        for s in eachindex(vs.density)
            src = (vs.positions[1, s], vs.positions[2, s], vs.positions[3, s])
            acc += vs.weights[s] * vs.density[s] * BI.laplace3d_grad(src, p, (0.0, 0.0, 1.0))
        end
        rhs_ref[i] = acc / eps_src
    end

    h = BI._estimate_source_spacing(vs)
    is_near = BI._classify_near_far_targets(targets, vs, h)
    @test any(is_near)
    @test !all(is_near)

    rel_errors = Float64[]
    for kmax_factor in (0.5, 0.75, 1.0, 1.25)
        rhs, _, _ = BI._rhs_volume_targets_hybrid(
            sources, charges, targets, normals, eps_src, 1e-9, kmax_factor * base_kmax, is_near,
        )
        rel = norm(rhs .- rhs_ref) / max(norm(rhs_ref), eps(Float64))
        push!(rel_errors, rel)
    end

    @test minimum(rel_errors[2:end]) < rel_errors[1]
    @test minimum(rel_errors) <= 0.12
    @test rel_errors[end] <= 0.12
end
