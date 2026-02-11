using BoundaryIntegral
using Random
using Test

@testset "VolumeSource grid" begin
    Random.seed!(0)
    xs = [0.0, 1.0]
    ys = [-1.0, 2.0]
    zs = [0.5, 1.5]
    points = NTuple{3, Float64}[]
    for x in xs, y in ys, z in zs
        push!(points, (x, y, z))
    end
    weights = collect(1.0:length(points))
    density = collect(10.0 .+ (1:length(points)))
    order = randperm(length(points))
    src = VolumeSource(points[order], weights[order], density[order])

    @test src.axes[1] == xs
    @test src.axes[2] == ys
    @test src.axes[3] == zs
    @test size(src.weights) == (2, 2, 2)
    @test size(src.density) == (2, 2, 2)

    lookup_w = Dict(points[i] => weights[i] for i in 1:length(points))
    lookup_d = Dict(points[i] => density[i] for i in 1:length(points))
    @test src.weights[1, 1, 1] == lookup_w[(xs[1], ys[1], zs[1])]
    @test src.density[2, 2, 2] == lookup_d[(xs[2], ys[2], zs[2])]
end

@testset "GaussianVolumeSource grid" begin
    src = BoundaryIntegral.GaussianVolumeSource((0.0, 0.0, 0.0), 1.0, 3, 1e-2)
    @test size(src.density) == (3, 3, 3)
    @test size(src.weights) == (3, 3, 3)
    @test length(src.axes) == 3
    @test issorted(src.axes[1])
    @test issorted(src.axes[2])
    @test issorted(src.axes[3])
end

@testset "GaussianVolumeSource auto n" begin
    center = (0.0, 0.0, 0.0)
    σ = 1.0
    tol_lo = 1e-2
    tol_hi = 1e-6

    src_lo = BoundaryIntegral.GaussianVolumeSource(center, σ, tol_lo)
    src_hi = BoundaryIntegral.GaussianVolumeSource(center, σ, tol_hi)

    n_lo = size(src_lo.density, 1)
    n_hi = size(src_hi.density, 1)
    @test n_lo >= 1
    @test n_hi >= n_lo

    src_lo_manual = BoundaryIntegral.GaussianVolumeSource(center, σ, n_lo, tol_lo)
    @test size(src_lo_manual.density) == size(src_lo.density)
end

@testset "GaussianVolumeSource auto n interp error" begin
    center = (0.0, 0.0, 0.0)
    σ = 1.0
    tol = 1e-3
    src = BoundaryIntegral.GaussianVolumeSource(center, σ, tol)
    xs, ys, zs = src.axes
    two_sigma2 = 2 * σ * σ
    norm_factor = inv((sqrt(2 * pi) * σ)^3)
    n = length(xs)
    ns, ws = BoundaryIntegral.gausslegendre(n)
    support_r = sqrt(two_sigma2 * log(inv(tol)))
    λ = BoundaryIntegral.gl_barycentric_weights(ns, ws)
    gx = exp.(-((support_r .* ns) .^ 2) ./ two_sigma2)
    gy = gx
    gz = gx

    n_sample = 10
    us = collect(LinRange(-1.0, 1.0, n_sample))

    max_err = 0.0
    for i in eachindex(us), j in eachindex(us), k in eachindex(us)
        rx = BoundaryIntegral.barycentric_row(ns, λ, us[i])
        ry = BoundaryIntegral.barycentric_row(ns, λ, us[j])
        rz = BoundaryIntegral.barycentric_row(ns, λ, us[k])
        gx_i = sum(rx .* gx)
        gy_j = sum(ry .* gy)
        gz_k = sum(rz .* gz)
        approx = norm_factor * gx_i * gy_j * gz_k
        x = support_r * us[i]
        y = support_r * us[j]
        z = support_r * us[k]
        r2 = (x - center[1])^2 + (y - center[2])^2 + (z - center[3])^2
        exact = norm_factor * exp(-r2 / two_sigma2)
        max_err = max(max_err, abs(approx - exact))
    end

    @test max_err <= tol
end

@testset "VolumeSource resample uniform" begin
    xs = [0.0, 0.5, 2.0]
    ys = [-1.0, 0.0, 1.5]
    zs = [0.0, 0.2, 1.0]
    dens = Array{Float64, 3}(undef, 3, 3, 3)
    for i in 1:3, j in 1:3, k in 1:3
        dens[i, j, k] = xs[i] + 2 * ys[j] + 3 * zs[k]
    end

    @test_throws UndefVarError BoundaryIntegral._resample_volume_to_uniform((xs, ys, zs), dens)
end
