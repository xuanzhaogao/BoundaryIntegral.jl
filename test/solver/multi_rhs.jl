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

    @testset "assemble_rhs_group" begin
        fixdir = joinpath(@__DIR__, "..", "fixtures")
        si = read_system_input(joinpath(fixdir, "system_small.bie"))
        # center 1 groups only with itself under cutoff 3.0
        g = assemble_rhs_group(si, 1)
        @test g.center_id == 1
        @test g.neighbor_ids == [1]
        @test size(g.densities, 2) == 1           # K = 1
        @test size(g.positions, 2) == size(g.densities, 1)
        @test all(g.densities[:, 1] .≈ 1.0)        # rho_11 = 1
    end

    @testset "envelope + group interface" begin
        fixdir = joinpath(@__DIR__, "..", "fixtures")
        si = read_system_input(joinpath(fixdir, "system_small.bie"))
        g = assemble_rhs_group(si, 1)
        env = BoundaryIntegral.envelope_volume_source(g)
        @test length(env.density) == size(g.densities, 1)
        @test all(env.density .≈ 1.0)              # rss of a single all-ones column

        interface = build_group_interface(si, g; n_quad = 6, rhs_atol = 1e-3, l_ec = 0.25)
        @test interface isa DielectricInterface
        @test length(interface.panels) >= 6        # at least the 6 box faces
    end
end
