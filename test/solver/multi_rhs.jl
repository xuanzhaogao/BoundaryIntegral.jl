using BoundaryIntegral
using Test
using Krylov

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

    @testset "batched matvec == columnwise (Unit 5)" begin
        fixdir = joinpath(@__DIR__, "..", "fixtures")
        si = read_system_input(joinpath(fixdir, "system_small.bie"))
        g = assemble_rhs_group(si, 1)
        interface = build_group_interface(si, g; n_quad = 6, rhs_atol = 1e-3, l_ec = 0.25)

        Np = BoundaryIntegral.num_points(interface)
        op  = batched_lhs_dielectric_box3d_fmm3d_corrected(interface, 1e-9, 1e-9, 8)
        ref = lhs_dielectric_box3d_fmm3d_corrected(interface, 1e-9, 1e-9, 8)

        # vector path matches the reference LinearMap
        x = collect(range(0.1, 1.0; length = Np))
        @test maximum(abs.(op * x .- ref * x)) < 1e-9

        # matrix path equals applying the reference to each column
        X = hcat(x, reverse(x), fill(0.5, Np))
        Y = op * X
        @test size(Y) == size(X)
        for c in 1:size(X, 2)
            @test maximum(abs.(Y[:, c] .- ref * X[:, c])) < 1e-8
        end
    end

    @testset "block solve == looped single solves (Unit 6)" begin
        fixdir = joinpath(@__DIR__, "..", "fixtures")
        si = read_system_input(joinpath(fixdir, "system_small.bie"))
        g = assemble_rhs_group(si, 1)
        interface = build_group_interface(si, g; n_quad = 6, rhs_atol = 1e-3, l_ec = 0.25)
        op = batched_lhs_dielectric_box3d_fmm3d_corrected(interface, 1e-9, 1e-9, 8)
        F = rhs_dielectric_box3d_fmm3d_batched(interface, si, g, 1e-9)

        Σ, stats = BoundaryIntegral._block_gmres_solve(op, F; rtol = 1e-10, itmax = 200)
        @test size(Σ) == size(F)

        for c in 1:size(F, 2)
            xc, _ = Krylov.gmres(op, F[:, c]; rtol = 1e-10, itmax = 200)
            @test maximum(abs.(Σ[:, c] .- xc)) < 1e-6
        end
    end

    @testset "end-to-end .bie -> Sigma" begin
        fixdir = joinpath(@__DIR__, "..", "fixtures")
        out = solve_dielectric_box3d_group(
            joinpath(fixdir, "system_small.bie"), 1;
            n_quad = 6, rhs_atol = 1e-3, l_ec = 0.25,
            fmm_tol = 1e-9, up_tol = 1e-9, max_order = 8, rtol = 1e-10)
        @test size(out.sigma, 2) == 1
        @test size(out.sigma, 1) == BoundaryIntegral.num_points(out.interface)
        @test all(isfinite, out.sigma)
    end

    @testset "K=2 batched solve matches per-column" begin
        fixdir = joinpath(@__DIR__, "..", "fixtures")
        si = read_system_input(joinpath(fixdir, "system_pair.bie"))
        g = assemble_rhs_group(si, 1)
        @test num_pairs(g) == 2

        interface = build_group_interface(si, g; n_quad = 6, rhs_atol = 1e-3, l_ec = 0.25)
        op = batched_lhs_dielectric_box3d_fmm3d_corrected(interface, 1e-9, 1e-9, 8)
        F = rhs_dielectric_box3d_fmm3d_batched(interface, si, g, 1e-9)
        @test size(F, 2) == 2

        Σ, _ = BoundaryIntegral._block_gmres_solve(op, F; rtol = 1e-10, itmax = 300)
        for c in 1:2
            xc, _ = Krylov.gmres(op, F[:, c]; rtol = 1e-10, itmax = 300)
            @test maximum(abs.(Σ[:, c] .- xc)) < 1e-6
        end
    end

    @testset "union-support truncation + grid cache" begin
        fixdir = joinpath(@__DIR__, "..", "fixtures")
        si = read_system_input(joinpath(fixdir, "system_spike.bie"))
        # orb_spike is a 3x3x3 grid with a single nonzero point -> rho_11 spikes at one node
        g_full  = assemble_rhs_group(si, 1; support_rtol = 0.0)
        g_trunc = assemble_rhs_group(si, 1; support_rtol = 1e-3)
        @test size(g_full.positions, 2) == 27          # full grid kept
        @test size(g_trunc.positions, 2) == 1          # only the spike survives
        @test size(g_trunc.positions, 2) == size(g_trunc.densities, 1)  # mask shared across cols
        @test g_trunc.densities[1, 1] ≈ 1.0

        # grid cache is populated and reused (orbital read once)
        cache = Dict{String, Any}()
        assemble_rhs_group(si, 1; grid_cache = cache)
        @test haskey(cache, si.orbitals[1].xsf_path)
    end
end
