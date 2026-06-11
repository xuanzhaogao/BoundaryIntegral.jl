using BoundaryIntegral
using Krylov
using Test

# Fast, self-contained tests of the Vector{VolumeSource} package API:
#   - multi_dielectric_box3d_rhs_adaptive(..., vss::Vector, ...)  (per-source union refine)
#   - rhs_dielectric_box3d_fmm3d(interface, vss::Vector, thresh)  (N×K, nd-batched)
#   - solve_dielectric_box3d_block(interface, vss; ...)           (block GMRES)
# Uses a coarse interface + loose tolerances on purpose — these check internal AGREEMENT
# (vector == looped single-source), which is resolution-independent, so they stay cheap.

@testset "Vector{VolumeSource} multi-RHS API" begin
    boxes = [(center = (0.0, 0.0, 0.0), Lx = 4.0, Ly = 4.0, Lz = 4.0)]
    epses = [11.7]
    eps_out = 1.0

    # two sources on the SAME small grid (exercises the nd=K batched path)
    xs = -0.3:0.3:0.3
    P = [(x, y, z) for x in xs for y in xs for z in xs]
    pos = reduce(hcat, collect.(P))
    w = fill(0.027, length(P))
    d1 = [exp(-((p[1])^2 + p[2]^2 + p[3]^2) / 0.1) for p in P]
    d2 = [exp(-((p[1] - 0.2)^2 + p[2]^2 + p[3]^2) / 0.1) for p in P]
    vs1 = VolumeSource(pos, copy(w), d1)
    vs2 = VolumeSource(pos, copy(w), d2)

    interface = multi_dielectric_box3d_rhs_adaptive(
        4, 2.0, boxes, epses, [vs1, vs2], 1e-2; eps_out = eps_out)
    Np = BoundaryIntegral.num_points(interface)

    @testset "vector RHS == per-column single-source" begin
        F = rhs_dielectric_box3d_fmm3d(interface, [vs1, vs2], 1e-6)
        @test size(F) == (Np, 2)
        @test maximum(abs.(F[:, 1] .- rhs_dielectric_box3d_fmm3d(interface, vs1, 1e-6))) < 1e-10
        @test maximum(abs.(F[:, 2] .- rhs_dielectric_box3d_fmm3d(interface, vs2, 1e-6))) < 1e-10
    end

    @testset "block solve == per-column gmres" begin
        op = batched_lhs_dielectric_box3d_fmm3d_corrected(interface, 1e-6, 1e-6, 8)
        F = rhs_dielectric_box3d_fmm3d(interface, [vs1, vs2], 1e-6)
        Σ, _ = solve_dielectric_box3d_block(interface, [vs1, vs2];
            fmm_tol = 1e-6, up_tol = 1e-6, max_order = 8, rtol = 1e-8, itmax = 300)
        @test size(Σ) == (Np, 2)
        for c in 1:2
            xc, _ = Krylov.gmres(op, F[:, c]; rtol = 1e-8, itmax = 300)
            @test maximum(abs.(Σ[:, c] .- xc)) < 1e-6
        end
    end
end
