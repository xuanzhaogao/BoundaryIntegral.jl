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

    # out-of-box: direct sum, identical arithmetic to the reference
    pref = _ref_potential(src, q, targets)
    gref = _ref_gradient(src, q, targets)
    for i in 5:8
        @test isapprox(pot[i], pref[i]; rtol = 1e-12)
        @test isapprox(grad[:, i], gref[:, i]; rtol = 1e-12)
    end

    # r→0 branch of the analytic helper: check continuity instead of testing the implementation against itself
    @test isapprox(_gaussian_analytic_potential([0.0, 0.0, 0.0], 0.3), _gaussian_analytic_potential([1e-8, 0.0, 0.0], 0.3); rtol = 1e-8)

    # gradient-only / potential-only construction guards
    fg = PrecomputedVolumeField(gsrc; tol = 1e-6, compute_pot = false)
    @test_throws ArgumentError volume_field_potential(fg, targets)
    fp = PrecomputedVolumeField(gsrc; tol = 1e-6, compute_grad = false)
    @test_throws ArgumentError volume_field_gradient(fp, targets)
    # partial fields still evaluate their available quantity correctly
    @test isapprox(volume_field_gradient(fg, targets), grad; rtol = 1e-12)
    @test isapprox(volume_field_potential(fp, targets), pot; rtol = 1e-12)
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
    for i in 1:6
        @test abs(rhs_field[i] - rhs_hyb[i]) <= 1e-4 * scale
    end
end
