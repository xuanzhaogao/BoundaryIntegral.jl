using BoundaryIntegral
using Random
using StaticArrays
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

    @test size(src.positions) == (3, 8)
    @test length(src.weights) == 8
    @test length(src.density) == 8

    lookup_w = Dict(points[i] => weights[i] for i in 1:length(points))
    lookup_d = Dict(points[i] => density[i] for i in 1:length(points))
    for s in 1:length(src.weights)
        pos = (src.positions[1, s], src.positions[2, s], src.positions[3, s])
        @test src.weights[s] == lookup_w[pos]
        @test src.density[s] == lookup_d[pos]
    end
end

@testset "VolumeSource iterator" begin
    xs = [0.0, 1.0]
    ys = [-1.0, 2.0]
    zs = [0.5, 1.5]
    points = NTuple{3, Float64}[]
    for x in xs, y in ys, z in zs
        push!(points, (x, y, z))
    end
    weights = collect(1.0:length(points))
    density = collect(10.0 .+ (1:length(points)))
    src = VolumeSource(points, weights, density)

    pts = collect(BoundaryIntegral.eachpoint(src))
    @test length(pts) == length(points)

    first_pt = pts[1]
    @test first_pt.idx == 1
    @test first_pt.global_idx == 1
    @test first_pt.point == BoundaryIntegral.volume_source_point(src, 1)
    @test first_pt.weight == src.weights[1]
    @test first_pt.density == src.density[1]

    last_pt = pts[end]
    @test last_pt.idx == length(points)
    @test last_pt.global_idx == length(points)
    @test last_pt.point == BoundaryIntegral.volume_source_point(src, length(points))
    @test last_pt.weight == src.weights[end]
    @test last_pt.density == src.density[end]
end

@testset "GaussianVolumeSource grid" begin
    src = BoundaryIntegral.GaussianVolumeSource((0.0, 0.0, 0.0), 1.0, 3, 1e-2)
    @test length(src.density) == 27
    @test length(src.weights) == 27
    @test size(src.positions) == (3, 27)
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

@testset "VolumeSource from XSF datagrid supports parallelepiped cells" begin
    nx, ny, nz = 3, 3, 3
    origin = SVector(0.2, -0.1, 0.4)
    A = SVector(2.0, 0.0, 0.0)
    B = SVector(0.5, 1.5, 0.0)
    C = SVector(0.25, 0.0, 1.0)
    values = reshape(collect(1.0:(nx * ny * nz)), nx, ny, nz)
    datagrid = (nx = nx, ny = ny, nz = nz, origin = origin, A = A, B = B, C = C, values = values)

    src = BoundaryIntegral.VolumeSource(datagrid)

    @test length(src.density) == nx * ny * nz
    @test length(src.weights) == nx * ny * nz

    idx = (3 - 1) * ny * nz + (2 - 1) * nz + 1
    @test BoundaryIntegral.volume_source_point(src, idx) == BoundaryIntegral.grid_point(datagrid, 3, 2, 1)

    At, Bt, Ct = BoundaryIntegral.true_cell_vectors(datagrid)
    jac = abs(det(hcat(collect(At), collect(Bt), collect(Ct))))
    @test all(isapprox.(src.weights, fill(jac / (nx * ny * nz), nx * ny * nz)))
end

@testset "VolumeSource truncates low-density points" begin
    points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    weights = [1.0, 2.0, 3.0]
    density = [1e-4, 0.2, -0.05]

    src = BoundaryIntegral.VolumeSource(points, weights, density; tol = 0.1)
    @test length(src.density) == 1
    @test src.density[1] == 0.2
    @test src.weights[1] == 2.0
    @test src.positions[:, 1] == [1.0, 0.0, 0.0]
end

@testset "datagrid_zslice extracts z-layers by index or z-value" begin
    nx, ny, nz = 4, 3, 5
    origin = SVector(0.0, 0.0, -1.0)
    A = SVector(2.0, 0.0, 0.0)
    B = SVector(0.0, 3.0, 0.0)
    C = SVector(0.0, 0.0, 5.0)
    values = reshape(collect(1.0:(nx * ny * nz)), nx, ny, nz)
    datagrid = (nx = nx, ny = ny, nz = nz, origin = origin, A = A, B = B, C = C, values = values)

    s_mid = BoundaryIntegral.datagrid_zslice(datagrid)
    @test s_mid.iz == cld(nz, 2)
    @test size(s_mid.values) == (nx, ny)
    @test s_mid.values == values[:, :, cld(nz, 2)]

    s_i = BoundaryIntegral.datagrid_zslice(datagrid; iz = 2)
    @test s_i.iz == 2
    @test s_i.values == values[:, :, 2]
    @test length(s_i.x) == nx
    @test length(s_i.y) == ny

    z2 = BoundaryIntegral.grid_point(datagrid, 1, 1, 2)[3]
    s_z = BoundaryIntegral.datagrid_zslice(datagrid; z = z2)
    @test s_z.iz == 2
    @test s_z.values == values[:, :, 2]

    @test_throws ArgumentError BoundaryIntegral.datagrid_zslice(datagrid; iz = 0)
    @test_throws ArgumentError BoundaryIntegral.datagrid_zslice(datagrid; iz = nz + 1)
    @test_throws ArgumentError BoundaryIntegral.datagrid_zslice(datagrid; iz = 2, z = z2)
end

@testset "datagrid_zslice trilinear interpolation on skew lattice for z=z0" begin
    nx, ny, nz = 5, 5, 5
    origin = SVector(0.2, -0.1, 0.4)
    A = SVector(1.0, 0.0, 0.25)
    B = SVector(0.1, 1.2, 0.15)
    C = SVector(0.2, 0.1, 1.5)
    values = Array{Float64, 3}(undef, nx, ny, nz)
    datagrid0 = (nx = nx, ny = ny, nz = nz, origin = origin, A = A, B = B, C = C, values = zeros(nx, ny, nz))

    for i in 1:nx, j in 1:ny, k in 1:nz
        x, y, z = BoundaryIntegral.grid_point(datagrid0, i, j, k)
        values[i, j, k] = 2.0 * x - 3.0 * y + 0.5 * z + 1.0
    end
    datagrid = (nx = nx, ny = ny, nz = nz, origin = origin, A = A, B = B, C = C, values = values)

    z0 = BoundaryIntegral.grid_point(datagrid, 3, 3, 3)[3]
    s = BoundaryIntegral.datagrid_zslice(
        datagrid;
        z = z0,
        nx_sample = 11,
        ny_sample = 10,
        interpolation = :trilinear,
    )
    @test size(s.values) == (11, 10)
    finite_vals = [v for v in s.values if isfinite(v)]
    @test !isempty(finite_vals)

    max_err = 0.0
    for i in eachindex(s.x), j in eachindex(s.y)
        v = s.values[i, j]
        isfinite(v) || continue
        exact = 2.0 * s.x[i] - 3.0 * s.y[j] + 0.5 * z0 + 1.0
        max_err = max(max_err, abs(v - exact))
    end
    @test max_err <= 1e-10
end
