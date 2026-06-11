# Reference O(N^2) sums in the free-space 1/(4π r) convention.
function _ref_potential(sources, charges, targets)
    n = size(targets, 2)
    out = zeros(n)
    for i in 1:n, j in eachindex(charges)
        dx = targets[1, i] - sources[1, j]
        dy = targets[2, i] - sources[2, j]
        dz = targets[3, i] - sources[3, j]
        out[i] += charges[j] / sqrt(dx^2 + dy^2 + dz^2)
    end
    return out ./ (4π)
end

function _ref_gradient(sources, charges, targets)
    n = size(targets, 2)
    out = zeros(3, n)
    for i in 1:n, j in eachindex(charges)
        dx = targets[1, i] - sources[1, j]
        dy = targets[2, i] - sources[2, j]
        dz = targets[3, i] - sources[3, j]
        r2 = dx^2 + dy^2 + dz^2
        s = charges[j] / (r2 * sqrt(r2))
        out[1, i] -= s * dx; out[2, i] -= s * dy; out[3, i] -= s * dz
    end
    return out ./ (4π)
end

# Analytic free-space potential and gradient for a unit Gaussian N(0, σ²) charge distribution.
# Uses the exact result: φ(r) = erf(r / (√2 σ)) / (4π r).
function _gaussian_analytic_potential(target, sigma)
    r = sqrt(target[1]^2 + target[2]^2 + target[3]^2)
    r < eps() && return sqrt(2 / π) / (4π * sigma)      # L'Hopital limit: erf(x)/x → 2/√π as x→0
    return BoundaryIntegral.SpecialFunctions.erf(r / (sqrt(2) * sigma)) / (4π * r)
end

function _gaussian_analytic_gradient(target, sigma)
    r = sqrt(target[1]^2 + target[2]^2 + target[3]^2)
    r < eps() && return zeros(3)
    a = sqrt(2) * sigma
    dfdr = (2 / sqrt(π) / a * exp(-(r / a)^2) * r - BoundaryIntegral.SpecialFunctions.erf(r / a)) / r^2
    return (dfdr / (4π * r)) .* [target[1], target[2], target[3]]
end

@testset "PrecomputedVolumeField potential/gradient" begin
    gsrc = BoundaryIntegral.GaussianVolumeSource((0.0, 0.0, 0.0), 0.3, 12, 1e-6)
    field = PrecomputedVolumeField(gsrc; tol = 1e-6)
    src, q = BoundaryIntegral._volume_source_fmm_sources(gsrc)

    # mixed batch: 4 targets inside the field box, 4 well outside it
    targets = [0.05  0.0   0.2  -0.15  3.0  0.0  -3.0  4.0;
               0.0   0.1  -0.1   0.05  0.0  3.5   2.0  4.0;
               0.0   0.02  0.1  -0.1   0.0  1.0  -2.0  4.0]
    @test count(i -> BoundaryIntegral.in_field_box(field, targets, i), 1:8) == 4

    pot = volume_field_potential(field, targets)
    grad = volume_field_gradient(field, targets)

    # in-box: spectral path — compare against the analytic Gaussian integral.
    # The discrete O(N²) sum is NOT used here because the 12³ quadrature grid has
    # ~5% near-field discretisation error; the spectral TKM sum correctly recovers
    # the continuous Gaussian potential to ~1e-4 (NUFFT tol + kmax truncation).
    sigma = 0.3
    for i in 1:4
        tanal = targets[:, i]
        panal = _gaussian_analytic_potential(tanal, sigma)
        ganal = _gaussian_analytic_gradient(tanal, sigma)
        # Spectral method accuracy: NUFFT tol=1e-6 + kmax Nyquist truncation.
        # Potential: ~1e-4 relative error; gradient: ~1e-2 (kmax harder to resolve).
        @test isapprox(pot[i], panal; rtol = 5e-4)
        @test isapprox(grad[:, i], ganal; rtol = 1e-2)
    end

    # out-of-box: FMM at f.tol = 1e-6 vs exact point sum
    pref = _ref_potential(src, q, targets)
    gref = _ref_gradient(src, q, targets)
    for i in 5:8
        @test isapprox(pot[i], pref[i]; rtol = 1e-5)
        @test isapprox(grad[:, i], gref[:, i]; rtol = 1e-5)
    end

    # r→0 branch of the analytic helper: check continuity instead of testing the implementation against itself
    @test isapprox(_gaussian_analytic_potential([0.0, 0.0, 0.0], 0.3), _gaussian_analytic_potential([1e-8, 0.0, 0.0], 0.3); rtol = 1e-8)

    # gradient-only / potential-only construction guards
    fg = PrecomputedVolumeField(gsrc; tol = 1e-6, compute_pot = false)
    @test_throws ArgumentError volume_field_potential(fg, targets)
    fp = PrecomputedVolumeField(gsrc; tol = 1e-6, compute_grad = false)
    @test_throws ArgumentError volume_field_gradient(fp, targets)
    # partial fields still evaluate their available quantity correctly
    # two independent FMM calls; thread-order nondeterminism
    @test isapprox(volume_field_gradient(fg, targets), grad; rtol = 1e-8)
    @test isapprox(volume_field_potential(fp, targets), pot; rtol = 1e-8)
end

@testset "_rhs_volume_targets_field vs hybrid" begin
    gsrc = BoundaryIntegral.GaussianVolumeSource((0.0, 0.0, 0.0), 0.3, 12, 1e-6)
    field = PrecomputedVolumeField(gsrc; tol = 1e-6, compute_pot = false)
    src, q = BoundaryIntegral._volume_source_fmm_sources(gsrc)
    h = BoundaryIntegral._estimate_source_spacing(gsrc)
    kmax = BoundaryIntegral._estimate_tkm3dc_kmax(h)

    targets = [0.05  0.2  -0.15  3.0  -3.0  4.0;
               0.0  -0.1   0.05  0.0   2.0  4.0;
               0.0   0.1  -0.1   0.0  -2.0  4.0]
    normals = [1.0  0.0  0.0  0.0  1.0  0.0;
               0.0  1.0  0.0  0.0  0.0  1.0;
               0.0  0.0  1.0  1.0  0.0  0.0]

    rhs_field = BoundaryIntegral._rhs_volume_targets_field(field, targets, normals, 1.0)

    is_near = BoundaryIntegral._classify_near_far_targets(targets, gsrc, h)
    rhs_hyb, _, _ = BoundaryIntegral._rhs_volume_targets_hybrid(
        src, q, targets, normals, 1.0, 1e-6, kmax, is_near)

    scale = maximum(abs, rhs_hyb)
    # Both paths approximate the CONTINUOUS field independently (hybrid: ltkm3dc
    # direct TKM; field: precomputed spectral type-2), each with kmax-truncation
    # error ~1e-4 on this coarse 12^3 grid, so their mutual deviation is bounded
    # by the sum of the two errors. 5e-4 * scale is well inside the production
    # RHS budget (rhs_atol = 1e-3); the production-scale validation measured
    # 1.4e-5 relative deviation (exp65 bench_meshgen, kmax = 40.2).
    for i in 1:6
        @test abs(rhs_field[i] - rhs_hyb[i]) <= 5e-4 * scale
    end
end

@testset "rhs_dielectric_box3d_field vs hybrid" begin
    gsrc = BoundaryIntegral.GaussianVolumeSource((0.0, 0.0, 0.0), 0.3, 12, 1e-6)
    field = PrecomputedVolumeField(gsrc; tol = 1e-6, compute_pot = false)
    iface = BoundaryIntegral.single_dielectric_box3d_rhs_adaptive(
        4.0, 4.0, 1.0, 4, gsrc, 1.0, 0.26, 1e-3, 3.0, 1.0, Float64; max_depth = 8)

    rhs_f = rhs_dielectric_box3d_field(iface, field, 1.0)
    rhs_h = BoundaryIntegral.rhs_dielectric_box3d_hybrid(iface, gsrc, 1.0, 1e-6)

    @test length(rhs_f) == BoundaryIntegral.num_points(iface)
    @test maximum(abs, rhs_f .- rhs_h) <= 5e-4 * maximum(abs, rhs_h)
end

@testset "cache_fft mode matches standard spectral path" begin
    gsrc = BoundaryIntegral.GaussianVolumeSource((0.0, 0.0, 0.0), 0.3, 12, 1e-6)
    f0 = PrecomputedVolumeField(gsrc; tol = 1e-6)
    f1 = PrecomputedVolumeField(gsrc; tol = 1e-6, cache_fft = true)

    # cached mode drops the coefficient arrays and stores fine grids instead
    @test f1.coeff === nothing && f1.grad_coeff === nothing
    @test f1.pot_grid !== nothing && f1.grad_grid !== nothing
    @test size(f1.pot_grid) == f1.nfdim
    @test size(f1.grad_grid) == (f1.nfdim..., 3)

    # mixed batch: 200 quasi-random in-box targets (interp-only vs standard
    # type-2) + 4 out-of-box targets (FMM in both modes — identical path)
    nin = 200
    targets = Matrix{Float64}(undef, 3, nin + 4)
    alphas = (sqrt(2) - 1, sqrt(3) - 1, sqrt(5) - 2)   # deterministic low-discrepancy fill
    for d in 1:3
        targets[d, 1:nin] .= f0.lo[d] .+ mod.(alphas[d] .* (1:nin), 1.0) .* (f0.hi[d] - f0.lo[d])
    end
    targets[:, nin + 1:end] .= [3.0  0.0  -3.0  4.0;
                                0.0  3.5   2.0  4.0;
                                0.0  1.0  -2.0  4.0]
    @test count(i -> BoundaryIntegral.in_field_box(f0, targets, i), 1:(nin + 4)) == nin

    pot0 = volume_field_potential(f0, targets)
    pot1 = volume_field_potential(f1, targets)
    grad0 = volume_field_gradient(f0, targets)
    grad1 = volume_field_gradient(f1, targets)

    # elementwise agreement relative to the field scale (stricter than vector-norm isapprox)
    @test maximum(abs, pot1 .- pot0) <= 1e-6 * maximum(abs, pot0)
    @test maximum(abs, grad1 .- grad0) <= 1e-6 * maximum(abs, grad0)

    # partial-construction guards behave the same in cached mode
    fg = PrecomputedVolumeField(gsrc; tol = 1e-6, compute_pot = false, cache_fft = true)
    @test fg.pot_grid === nothing && fg.grad_grid !== nothing
    @test_throws ArgumentError volume_field_potential(fg, targets)
    @test maximum(abs, volume_field_gradient(fg, targets) .- grad0) <= 1e-6 * maximum(abs, grad0)
    fp = PrecomputedVolumeField(gsrc; tol = 1e-6, compute_grad = false, cache_fft = true)
    @test fp.pot_grid !== nothing && fp.grad_grid === nothing
    @test_throws ArgumentError volume_field_gradient(fp, targets)
    @test maximum(abs, volume_field_potential(fp, targets) .- pot0) <= 1e-6 * maximum(abs, pot0)
end

@testset "adaptive builder: field overload reproduces VolumeSource path" begin
    gsrc = BoundaryIntegral.GaussianVolumeSource((0.0, 0.0, 0.0), 0.3, 12, 1e-6)
    field = PrecomputedVolumeField(gsrc; tol = 1e-4, compute_pot = false)

    iface_vs = BoundaryIntegral.single_dielectric_box3d_rhs_adaptive(
        4.0, 4.0, 1.0, 4, gsrc, 1.0, 0.26, 1e-3, 3.0, 1.0, Float64; max_depth = 8)
    iface_f = BoundaryIntegral.single_dielectric_box3d_rhs_adaptive(
        4.0, 4.0, 1.0, 4, field, 1.0, 0.26, 1e-3, 3.0, 1.0, Float64; max_depth = 8)

    @test length(iface_f.panels) == length(iface_vs.panels)
    @test BoundaryIntegral.num_points(iface_f) == BoundaryIntegral.num_points(iface_vs)
    # identical refinement path => identical panel geometry
    corner_key(iface) = sort([(p.corners[1]..., p.corners[3]...) for p in iface.panels])
    @test corner_key(iface_f) == corner_key(iface_vs)
end
