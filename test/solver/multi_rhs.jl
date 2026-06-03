using BoundaryIntegral
using Test

@testset "multi_rhs" begin
    fixdir = joinpath(@__DIR__, "..", "fixtures")

    @testset "pair_density_source shared grid" begin
        # orb_a has all-ones density on a 2x2x2 grid. rho = phi_a * phi_a = 1 everywhere.
        _, dg = BoundaryIntegral.read_xsf(joinpath(fixdir, "orb_a.xsf"))
        vs = BoundaryIntegral.pair_density_source(
            joinpath(fixdir, "orb_a.xsf"), joinpath(fixdir, "orb_a.xsf"))
        @test length(vs.density) == 8          # 2*2*2 grid points
        @test all(vs.density .≈ 1.0)            # 1 * 1
        # weights must match VolumeSource(datagrid) exactly so a pair density
        # integrates identically to a single-orbital density
        @test sum(vs.weights) ≈ sum(BoundaryIntegral.VolumeSource(dg).weights)
    end
end
