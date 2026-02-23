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
    @test first_pt.idx == (1, 1, 1)
    @test first_pt.global_idx == 1
    @test first_pt.point == BoundaryIntegral.volume_source_point(src, 1, 1, 1)
    @test first_pt.weight == src.weights[1, 1, 1]
    @test first_pt.density == src.density[1, 1, 1]

    last_pt = pts[end]
    @test last_pt.idx == (2, 2, 2)
    @test last_pt.global_idx == length(points)
    @test last_pt.point == BoundaryIntegral.volume_source_point(src, 2, 2, 2)
    @test last_pt.weight == src.weights[2, 2, 2]
    @test last_pt.density == src.density[2, 2, 2]

    # Iteration order should match ix outer, iy middle, iz inner.
    k = 0
    for ix in 1:length(xs), iy in 1:length(ys), iz in 1:length(zs)
        k += 1
        @test pts[k].idx == (ix, iy, iz)
        @test pts[k].global_idx == k
    end
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

@testset "VolumeSource from XSF datagrid supports parallelepiped cells" begin
    nx, ny, nz = 3, 3, 3
    origin = SVector(0.2, -0.1, 0.4)
    A = SVector(2.0, 0.0, 0.0)
    B = SVector(0.5, 1.5, 0.0)
    C = SVector(0.25, 0.0, 1.0)
    values = reshape(collect(1.0:(nx * ny * nz)), nx, ny, nz)
    datagrid = (nx = nx, ny = ny, nz = nz, origin = origin, A = A, B = B, C = C, values = values)

    src = BoundaryIntegral.VolumeSource(datagrid)

    @test size(src.density) == (nx, ny, nz)
    @test src.density == values
    @test BoundaryIntegral.volume_source_point(src, 3, 2, 1) ==
        BoundaryIntegral.grid_point(datagrid, 3, 2, 1)

    At, Bt, Ct = BoundaryIntegral.true_cell_vectors(datagrid)
    jac = abs(det(hcat(collect(At), collect(Bt), collect(Ct))))
    @test all(isapprox.(src.weights, fill(jac / (nx * ny * nz), nx, ny, nz)))
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
