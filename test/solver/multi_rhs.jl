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

    @testset "batched RHS == single-RHS (K=1 regression)" begin
        fixdir = joinpath(@__DIR__, "..", "fixtures")
        si = read_system_input(joinpath(fixdir, "system_small.bie"))
        g = assemble_rhs_group(si, 1)
        interface = build_group_interface(si, g; n_quad = 6, rhs_atol = 1e-3, l_ec = 0.25)

        F = rhs_dielectric_box3d_fmm3d_batched(interface, si, g, 1e-9)
        @test size(F) == (BoundaryIntegral.num_points(interface), 1)

        # reference: the existing per-source screened FMM RHS for rho_11
        vs = VolumeSource(copy(g.positions), copy(g.weights), g.densities[:, 1])
        f_ref = rhs_dielectric_box3d_fmm3d(interface, vs, 1e-9)   # screened convenience method
        @test maximum(abs.(F[:, 1] .- f_ref)) < 1e-6
    end
end
