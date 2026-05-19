using BoundaryIntegral
import BoundaryIntegral as BI
using LinearAlgebra, Krylov
using Random
using SpecialFunctions: erf
using Test

@testset "dielectric_box3d" begin
    eps_box = 4.0
    interface = BI.single_dielectric_box3d(1.2, 0.8, 0.6, 4, 0.2, eps_box, 1.0, Float64; alpha = sqrt(2))

    lhs = BI.lhs_dielectric_box3d(interface)
    lhs_fmm3d = BI.lhs_dielectric_box3d_fmm3d(interface, 1e-12)
    rhs = BI.rhs_dielectric_box3d(interface, BI.PointSource((0.1, 0.1, 0.1), 1.0), eps_box)
    ws = BI.all_weights(interface)

    x = BI.solve_lu(lhs, rhs)
    @test norm(lhs * x - rhs) < 1e-10

    x_trial = randn(BI.num_points(interface))
    @test norm(lhs * x_trial - lhs_fmm3d * x_trial) < 1e-8

    x_gmres = BI.solve_gmres(lhs_fmm3d, rhs, 1e-12, 1e-12)
    @test norm(lhs_fmm3d * x_gmres - rhs) < 1e-10

    total_flux = dot(ws, x)
    @test isapprox(total_flux + 1.0 / eps_box, 1.0, atol = 1e-1)

    total_flux_gmres = dot(ws, x_gmres)
    @test isapprox(total_flux_gmres + 1.0 / eps_box, 1.0, atol = 1e-1)
end

@testset "laplace3d_pottrg_near near/far" begin
    eps_box = 4.0
    interface = BI.single_dielectric_box3d(1.0, 1.0, 1.0, 2, 0.2, eps_box, 1.0, Float64; alpha = sqrt(2))
    sol = randn(BI.num_points(interface))

    # far away target
    target = (100.0, 100.0, 100.0)
    pot_direct = (BI.laplace3d_pottrg(interface, reshape(collect(target), 3, 1)) * sol)[1]
    pot_hc_1 = BI.laplace3d_pottrg_near(interface, target, sol, 1e-8; range_factor = 0.0)
    pot_hc_2 = BI.laplace3d_pottrg_near(interface, target, sol, 1e-8; range_factor = Inf)
    @test isapprox(pot_hc_1, pot_direct, rtol = 1e-8)
    @test isapprox(pot_hc_2, pot_direct, rtol = 1e-8)

    # near target
    target = (0.6, 0.1, 0.2)
    pot_direct = (BI.laplace3d_pottrg(interface, reshape(collect(target), 3, 1)) * sol)[1]
    pot_hc_1 = BI.laplace3d_pottrg_near(interface, target, sol, 1e-6; range_factor = 5.0)
    pot_hc_2 = BI.laplace3d_pottrg_near(interface, target, sol, 1e-6; range_factor = Inf)
    # @test isapprox(pot_hc_1, pot_direct, rtol = 0.15)
    @test isapprox(pot_hc_1, pot_hc_2, atol = 1e-4)
end

@testset "dielectric_box3d corrected" begin
    eps_box = 4.0
    interface = BI.single_dielectric_box3d(3.0, 3.0, 1.0, 6, 0.2, eps_box, 1.0, Float64; alpha = sqrt(2))

    lhs_uncorrected = BI.lhs_dielectric_box3d_fmm3d(interface, 1e-6)
    lhs_corrected = BI.lhs_dielectric_box3d_fmm3d_corrected(interface, 1e-6, 1e-6, 12)
    rhs = BI.rhs_dielectric_box3d(interface, BI.PointSource((0.1, 0.1, 0.1), 1.0), eps_box)
    ws = BI.all_weights(interface)

    x_gmres_corrected = BI.solve_gmres(lhs_corrected, rhs, 1e-6, 1e-6)
    x_gmres_uncorrected = BI.solve_gmres(lhs_uncorrected, rhs, 1e-6, 1e-6)
    @test norm(lhs_corrected * x_gmres_corrected - rhs) < 1e-5
    @test norm(lhs_uncorrected * x_gmres_uncorrected - rhs) < 1e-5

    total_flux_corrected = dot(ws, x_gmres_corrected)
    total_flux_uncorrected = dot(ws, x_gmres_uncorrected)
    @test isapprox(total_flux_uncorrected + 1.0 / eps_box, 1.0, atol = 1e-1)
    @test isapprox(total_flux_corrected + 1.0 / eps_box, 1.0, atol = 1e-1)
end

@testset "dielectric_box3d rhs adaptive" begin
    eps_box = 4.0
    ps = BI.PointSource((0.1, 0.1, 0.1), 1.0)
    interface = BI.single_dielectric_box3d_rhs_adaptive(
        1.2,
        0.8,
        0.6,
        2,
        ps,
        eps_box,
        0.099,
        1e-8,
        eps_box,
        1.0,
        Float64;
        max_depth = 2,
    )
    lhs = BI.lhs_dielectric_box3d(interface)
    rhs = BI.rhs_dielectric_box3d(interface, ps, eps_box)
    x = BI.solve_gmres(lhs, rhs, 1e-6, 1e-6)
    @test norm(lhs * x - rhs) < 1e-5
    @test abs(dot(BI.all_weights(interface), x) +  1.0 / eps_box - 1.0) < 1e-2

    ps = BI.PointSource((1.0, 1.0, 1.0), 1.0)
    interface = BI.single_dielectric_box3d_rhs_adaptive(
        1.2,
        0.8,
        0.6,
        2,
        ps,
        1.0,
        0.099,
        1e-8,
        eps_box,
        1.0,
        Float64;
        max_depth = 2,
    )
    lhs = BI.lhs_dielectric_box3d(interface)
    rhs = BI.rhs_dielectric_box3d(interface, ps, 1.0)
    x = BI.solve_gmres(lhs, rhs, 1e-6, 1e-6)
    @test norm(lhs * x - rhs) < 1e-5
    @test abs(dot(BI.all_weights(interface), x)) < 1e-2
end

@testset "dielectric_box3d rhs volume source" begin
    eps_box = 4.0
    eps_src = 2.0
    interface = BI.single_dielectric_box3d(1.0, 1.0, 1.0, 2, 0.2, eps_box, 1.0, Float64; alpha = sqrt(2))

    xs = [2.0, 2.5]
    ys = [2.0, 2.5]
    zs = [2.0, 2.5]
    weights = fill(1.0, 2, 2, 2)
    density = fill(1.0, 2, 2, 2)
    vs = BI.VolumeSource{Float64, 3}((xs, ys, zs), weights, density)

    rhs_direct = BI.rhs_dielectric_box3d(interface, vs, eps_src)
    rhs_manual = zeros(Float64, BI.num_points(interface))
    for (i, point) in enumerate(BI.eachpoint(interface))
        acc = 0.0
        for ix in eachindex(xs), iy in eachindex(ys), iz in eachindex(zs)
            pos = (xs[ix], ys[iy], zs[iz])
            acc += weights[ix, iy, iz] * density[ix, iy, iz] *
                BI.laplace3d_grad(pos, point.panel_point.point, point.panel_point.normal)
        end
        rhs_manual[i] = -acc / eps_src
    end
    @test norm(rhs_direct - rhs_manual) <= 1e-12

    rhs_fmm = BI.rhs_dielectric_box3d_fmm3d(interface, vs, eps_src, 1e-8)
    @test norm(rhs_fmm - rhs_direct) / norm(rhs_direct) < 1e-6
end

@testset "dielectric_box3d rhs volume source hybrid near/far" begin
    eps_box = 4.0
    eps_src = 1.0
    interface = BI.single_dielectric_box3d(1.0, 1.0, 1.0, 4, 0.2, eps_box, 1.0, Float64; alpha = sqrt(2))
    vs = BI.GaussianVolumeSource((0.0, 0.0, 0.6), 0.1, 40, 1e-8)

    n_points = BI.num_points(interface)
    targets = Matrix{Float64}(undef, 3, n_points)
    for (i, point) in enumerate(BI.eachpoint(interface))
        targets[:, i] .= point.panel_point.point
    end
    h = BI._estimate_source_spacing(vs)
    is_near = BI._classify_near_far_targets(targets, vs, h)
    @test any(is_near)
    @test !all(is_near)

    rhs_hybrid = BI.rhs_dielectric_box3d_hybrid(interface, vs, eps_src, 1e-8)

    # use the analytical results for Gaussian to validate the rhs
    rhs_exact = BI.Rhs_dielectric_box3d_gaussian(interface, (0.0, 0.0, 0.6), 0.1, 1.0)

    @test norm(rhs_hybrid - rhs_exact) / norm(rhs_exact) < 1e-8
end

@testset "screened_volume_source" begin
    eps_in = 4.0
    eps_out = 1.0
    interface = BI.single_dielectric_box3d(1.0, 1.0, 1.0, 2, 0.2, eps_in, eps_out, Float64; alpha = sqrt(2))

    # mode is required (no default)
    vs_small = BI.VolumeSource([(0.0, 0.0, 0.0), (0.8, 0.0, 0.0)], [0.5, 0.25], [2.0, -3.0])
    @test_throws MethodError BI.screened_volume_source(interface, vs_small)
    @test_throws MethodError BI.screened_volume_source(1.0, 1.0, 1.0, vs_small, eps_in, eps_out)

    # bandwidth validation
    @test_throws ArgumentError BI.SoftMixPermittivity(0.0)
    @test_throws ArgumentError BI.SoftMixInversePermittivity(-0.1)

    # sharp screening
    rho0 = 2.0
    points = [
        (0.0, 0.0, 0.0),   # interior
        (0.5, 0.0, 0.0),   # face
        (0.5, 0.5, 0.0),   # edge
        (0.5, 0.5, 0.5),   # corner
        (1.5, 0.0, 0.0),   # exterior
    ]
    vs = BI.VolumeSource(points, ones(length(points)), fill(rho0, length(points)))

    sharp = BI.screened_volume_source(interface, vs, BI.SharpScreening())
    sharp_box = BI.screened_volume_source(1.0, 1.0, 1.0, vs, eps_in, eps_out, BI.SharpScreening())
    @test sharp.density ≈ sharp_box.density
    @test sharp.density ≈ [rho0/eps_in, rho0/eps_in, rho0/eps_in, rho0/eps_in, rho0/eps_out]

    # soft mix — reference formulas
    bandwidth = 0.05
    min_corner = (-0.5, -0.5, -0.5)
    max_corner = (0.5, 0.5, 0.5)
    H(t) = (1 + erf(t / bandwidth)) / 2
    screen(p) =
        H(p[1] - min_corner[1]) * H(max_corner[1] - p[1]) *
        H(p[2] - min_corner[2]) * H(max_corner[2] - p[2]) *
        H(p[3] - min_corner[3]) * H(max_corner[3] - p[3])

    # SoftMixPermittivity
    soft_eps = BI.screened_volume_source(interface, vs, BI.SoftMixPermittivity(bandwidth))
    soft_eps_box = BI.screened_volume_source(1.0, 1.0, 1.0, vs, eps_in, eps_out, BI.SoftMixPermittivity(bandwidth))
    @test soft_eps.density ≈ soft_eps_box.density
    ref_eps = [rho0 / (eps_in * screen(p) + eps_out * (1 - screen(p))) for p in points]
    @test soft_eps.density ≈ ref_eps atol = 1e-12
    @test soft_eps.density[1] ≈ rho0 / eps_in atol = 1e-12
    @test soft_eps.density[end] ≈ rho0 / eps_out atol = 1e-12

    # SoftMixInversePermittivity
    soft_inv = BI.screened_volume_source(interface, vs, BI.SoftMixInversePermittivity(bandwidth))
    soft_inv_box = BI.screened_volume_source(1.0, 1.0, 1.0, vs, eps_in, eps_out, BI.SoftMixInversePermittivity(bandwidth))
    @test soft_inv.density ≈ soft_inv_box.density
    ref_inv = [rho0 * (screen(p) / eps_in + (1 - screen(p)) / eps_out) for p in points]
    @test soft_inv.density ≈ ref_inv atol = 1e-12
    @test soft_inv.density[1] ≈ rho0 / eps_in atol = 1e-12
    @test soft_inv.density[end] ≈ rho0 / eps_out atol = 1e-12

    # RHS convenience overloads use sharp screening internally
    vs_sharp = BI.screened_volume_source(interface, vs_small, BI.SharpScreening())
    @test BI.rhs_dielectric_box3d(interface, vs_small) ≈ BI.rhs_dielectric_box3d(interface, vs_sharp, 1.0)
end

@testset "dielectric_box3d corrected edges flag" begin
    eps_box = 4.0
    interface = BI.single_dielectric_box3d(3.0, 3.0, 1.0, 6, 0.2, eps_box, 1.0, Float64; alpha = sqrt(2))
    rhs = BI.rhs_dielectric_box3d(interface, BI.PointSource((0.1, 0.1, 0.1), 1.0), eps_box)
    ws = BI.all_weights(interface)

    lhs_off = BI.lhs_dielectric_box3d_fmm3d_corrected(interface, 1e-6, 1e-6, 12;
                                                     correct_edges = false)
    lhs_on  = BI.lhs_dielectric_box3d_fmm3d_corrected(interface, 1e-6, 1e-6, 12;
                                                     correct_edges = true,
                                                     adaptive_atol = 1e-6)

    x_off = BI.solve_gmres(lhs_off, rhs, 1e-6, 1e-6)
    x_on  = BI.solve_gmres(lhs_on,  rhs, 1e-6, 1e-6)

    flux_off = dot(ws, x_off)
    flux_on  = dot(ws, x_on)

    err_off = abs(flux_off + 1.0 / eps_box - 1.0)
    err_on  = abs(flux_on  + 1.0 / eps_box - 1.0)
    @info "dielectric box flux error: off=$(err_off) on=$(err_on)"

    # Both should converge GMRES.
    @test norm(lhs_off * x_off - rhs) < 1e-5
    @test norm(lhs_on  * x_on  - rhs) < 1e-5
    # Both modes should produce physically reasonable flux values (loose check).
    @test err_off < 1e-1
    @test err_on  < 1e-1
end

@testset "dielectric_box3d volume backend wiring" begin
    root = pkgdir(BoundaryIntegral)
    entrypoint = read(joinpath(root, "src", "BoundaryIntegral.jl"), String)
    shape_dir = joinpath(root, "src", "shape")
    shape_sources = join([read(joinpath(shape_dir, f), String)
                          for f in readdir(shape_dir) if endswith(f, ".jl")], "\n")

    @test occursin("using TKM3D", entrypoint)
    @test !occursin("using FBCPoisson", entrypoint)
    @test occursin("ltkm3dc", shape_sources)
    @test !occursin("lfbc3d", shape_sources)
end

@testset "dielectric_box3d correct_edges improves edge-near pointwise potential" begin
    eps_box = 4.0
    Lx, Ly, Lz = 3.0, 3.0, 1.0
    source = BI.PointSource((0.1, 0.1, 0.1), 1.0)
    near_corner = (1.40, 1.40, 0.40)   # close to the +x/+y/+z corner of the box

    # Direct Coulomb potential of the point source (free space inside the box).
    function source_pot(target, src::BI.PointSource)
        r = sqrt((target[1]-src.point[1])^2 + (target[2]-src.point[2])^2 + (target[3]-src.point[3])^2)
        return src.charge / (4π * r)
    end

    function pot_at(interface, σ, target)
        induced = BI.laplace3d_pottrg_near(interface, target, σ, 1e-8; range_factor = 5.0)
        return source_pot(target, source) / eps_box + induced
    end

    function solve_at(l_ec, correct_edges_flag)
        interface = BI.single_dielectric_box3d(Lx, Ly, Lz, 6, l_ec, eps_box, 1.0, Float64; alpha = sqrt(2))
        lhs = BI.lhs_dielectric_box3d_fmm3d_corrected(interface, 1e-6, 1e-6, 12;
                                                     correct_edges = correct_edges_flag,
                                                     adaptive_atol = 1e-8)
        rhs = BI.rhs_dielectric_box3d(interface, source, eps_box)
        σ = BI.solve_gmres(lhs, rhs, 1e-7, 1e-7)
        return pot_at(interface, σ, near_corner)
    end

    # Reference at fine resolution with correction enabled.
    pot_ref = solve_at(0.06, true)

    # Coarse-mesh comparison.
    pot_off = solve_at(0.10, false)
    pot_on  = solve_at(0.10, true)

    err_off = abs(pot_off - pot_ref)
    err_on  = abs(pot_on  - pot_ref)
    @info "near-corner pointwise error" err_off err_on

    # The corrected version must be meaningfully better near the corner.
    # Empirical baseline at l_ec=0.10: err_off ≈ 1.3e-4, err_on ≈ 3.7e-5 (3.5× better).
    # We require at least 2× improvement to allow some headroom.
    @test err_on < err_off / 2
    @test err_on < 1e-4     # absolute bound
end
