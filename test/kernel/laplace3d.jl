using BoundaryIntegral
import BoundaryIntegral as BI
using LinearAlgebra
using Random
using Test

function box3d_interface()
    return BI.single_dielectric_box3d(1.2, 0.8, 0.6, 4, 0.2, 2.0, 1.0, Float64; alpha = sqrt(2))
end

@testset "laplace3d kernels on meshed box" begin
    interface = box3d_interface()

    n = BI.num_points(interface)
    charges = randn(n)
    tol = 1e-12

    S_direct = BI.laplace3d_S(interface)
    S_direct[diagind(S_direct)] .= 0.0
    targets = zeros(Float64, 3, n)
    for (i, point) in enumerate(BI.eachpoint(interface))
        targets[1, i] = point.panel_point.point[1]
        targets[2, i] = point.panel_point.point[2]
        targets[3, i] = point.panel_point.point[3]
    end
    S_fmm = BI.laplace3d_pottrg_fmm3d(interface, targets, tol) * charges
    @test norm(S_direct * charges - S_fmm) < 1e-8

    D_direct = BI.laplace3d_D(interface)
    DT_direct = BI.laplace3d_DT(interface)
    D_direct[diagind(D_direct)] .= 0.0
    DT_direct[diagind(DT_direct)] .= 0.0
    D_fmm = BI.laplace3d_D_fmm3d(interface, tol)
    DT_fmm = BI.laplace3d_DT_fmm3d(interface, tol)
    @test norm(D_direct * charges - D_fmm * charges) < 1e-8
    @test norm(DT_direct * charges - DT_fmm * charges) < 1e-8
end

function direct_D_trg(points, normals, weights, targets, charges)
    n = length(points)
    m = size(targets, 2)
    out = zeros(eltype(charges), m)
    for i in 1:m
        acc = zero(eltype(charges))
        trg = (targets[1, i], targets[2, i], targets[3, i])
        for j in 1:n
            acc += BI.laplace3d_grad(points[j], trg, normals[j]) * weights[j] * charges[j]
        end
        out[i] = acc
    end
    return out
end

@testset "laplace3d FMM3D comparisons" begin
    Random.seed!(0)
    interface = box3d_interface()

    points = NTuple{3, Float64}[]
    normals = NTuple{3, Float64}[]
    weights = Float64[]
    for point in BI.eachpoint(interface)
        push!(points, point.panel_point.point)
        push!(normals, point.panel_point.normal)
        push!(weights, point.panel_point.weight)
    end

    n = length(points)
    charges = randn(n)
    tol = 1e-12

    targets = zeros(Float64, 3, n)
    for i in 1:n
        targets[1, i] = points[i][1]
        targets[2, i] = points[i][2]
        targets[3, i] = points[i][3]
    end

    S_direct = BI.laplace3d_S(interface)
    S_direct[diagind(S_direct)] .= 0.0
    S_fmm = BI.laplace3d_pottrg_fmm3d(interface, targets, tol) * charges
    @test norm(S_direct * charges - S_fmm) < 5e-9

    D_direct = BI.laplace3d_D(interface)
    D_direct[diagind(D_direct)] .= 0.0
    DT_direct = BI.laplace3d_DT(interface)
    DT_direct[diagind(DT_direct)] .= 0.0

    D_fmm = BI.laplace3d_D_fmm3d(interface, tol)
    DT_fmm = BI.laplace3d_DT_fmm3d(interface, tol)
    @test norm(D_direct * charges - D_fmm * charges) < 5e-9
    @test norm(DT_direct * charges - DT_fmm * charges) < 5e-9

    m = n + 2
    targets_off = zeros(Float64, 3, m)
    for i in 1:m
        idx = ((i - 1) % n) + 1
        targets_off[1, i] = points[idx][1] + 0.2 * normals[idx][1]
        targets_off[2, i] = points[idx][2] + 0.2 * normals[idx][2]
        targets_off[3, i] = points[idx][3] + 0.2 * normals[idx][3]
    end

    pot_direct = BI.laplace3d_pottrg(interface, targets_off) * charges
    pot_fmm = BI.laplace3d_pottrg_fmm3d(interface, targets_off, tol) * charges
    @test norm(pot_direct - pot_fmm) < 5e-9

    D_trg_direct = direct_D_trg(points, normals, weights, targets_off, charges)
    D_trg_fmm = BI.laplace3d_D_trg_fmm3d(interface, targets_off, tol) * charges
    @test norm(D_trg_direct - D_trg_fmm) < 5e-9
end

@testset "laplace3d box flux convergence" begin
    src = (0.1, 0.1, 0.1)
    n_quads = (2, 4, 6)
    errs = Float64[]

    for n_quad in n_quads
        interface = BI.single_dielectric_box3d(1.0, 1.0, 1.0, n_quad, 0.3, 2.0, 1.0, Float64; alpha = sqrt(2))
        flux = 0.0
        for p in BI.eachpoint(interface)
            flux += BI.laplace3d_grad(src, p.panel_point.point, p.panel_point.normal) * p.panel_point.weight
        end
        push!(errs, abs(flux - 1.0))
    end

    @test errs[2] <= errs[1]
    @test errs[3] <= errs[2]
    @test errs[3] < 1e-8
end
