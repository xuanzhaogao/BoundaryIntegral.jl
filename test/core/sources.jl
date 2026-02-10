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

@testset "VolumeSource resample uniform" begin
    xs = [0.0, 0.5, 2.0]
    ys = [-1.0, 0.0, 1.5]
    zs = [0.0, 0.2, 1.0]
    dens = Array{Float64, 3}(undef, 3, 3, 3)
    for i in 1:3, j in 1:3, k in 1:3
        dens[i, j, k] = xs[i] + 2 * ys[j] + 3 * zs[k]
    end

    axes_u, dens_u = BoundaryIntegral._resample_volume_to_uniform((xs, ys, zs), dens)
    xsu, ysu, zsu = axes_u
    @test size(dens_u) == (3, 3, 3)
    @test issorted(xsu)
    @test issorted(ysu)
    @test issorted(zsu)

    for i in 1:3, j in 1:3, k in 1:3
        expected = xsu[i] + 2 * ysu[j] + 3 * zsu[k]
        @test isapprox(dens_u[i, j, k], expected; rtol = 1e-12, atol = 1e-12)
    end
end
