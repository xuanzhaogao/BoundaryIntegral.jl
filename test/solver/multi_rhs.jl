using BoundaryIntegral
using Test
using Krylov

# These tests check INTERNAL AGREEMENT (batched == single-source, block == looped,
# matvec == columnwise) which is resolution-independent — so they use a COARSE shared
# interface and loose tolerances, built ONCE and reused, to stay fast.

@testset "multi_rhs" begin
    fixdir = joinpath(@__DIR__, "..", "fixtures")

    # coarse agreement-test parameters
    NQ, RTOL_REFINE, LEC = 4, 1e-2, 2.0      # n_quad, rhs_atol, l_ec
    FT, UT, MO = 1e-6, 1e-6, 8               # fmm_tol, up_tol, max_order
    GTOL = 1e-8                              # gmres rtol

    # ---- fast tests that need no interface ----
    @testset "pair_density_source shared grid" begin
        _, dg = BoundaryIntegral.read_xsf(joinpath(fixdir, "orb_a.xsf"))
        vs = BoundaryIntegral.pair_density_source(
            joinpath(fixdir, "orb_a.xsf"), joinpath(fixdir, "orb_a.xsf"))
        @test length(vs.density) == 8
        @test all(vs.density .≈ 1.0)
        @test sum(vs.weights) ≈ sum(BoundaryIntegral.VolumeSource(dg).weights)
    end

    @testset "assemble_rhs_group" begin
        si = read_system_input(joinpath(fixdir, "system_small.bie"))
        g = assemble_rhs_group(si, 1)
        @test g.center_id == 1
        @test g.neighbor_ids == [1]
        @test size(g.densities, 2) == 1
        @test size(g.positions, 2) == size(g.densities, 1)
        @test all(g.densities[:, 1] .≈ 1.0)
    end

    @testset "union-support truncation + grid cache" begin
        si = read_system_input(joinpath(fixdir, "system_spike.bie"))
        g_full  = assemble_rhs_group(si, 1; support_rtol = 0.0)
        g_trunc = assemble_rhs_group(si, 1; support_rtol = 1e-3)
        @test size(g_full.positions, 2) == 27
        @test size(g_trunc.positions, 2) == 1
        @test size(g_trunc.positions, 2) == size(g_trunc.densities, 1)
        @test g_trunc.densities[1, 1] ≈ 1.0
        cache = Dict{String, Any}()
        assemble_rhs_group(si, 1; grid_cache = cache)
        @test haskey(cache, si.orbitals[1].xsf_path)
    end

    # ---- K=1 small system: build the interface / operator / RHS ONCE ----
    si = read_system_input(joinpath(fixdir, "system_small.bie"))
    g  = assemble_rhs_group(si, 1)
    interface = build_group_interface(si, g; n_quad = NQ, rhs_atol = RTOL_REFINE, l_ec = LEC)
    Np = BoundaryIntegral.num_points(interface)
    op = batched_lhs_dielectric_box3d_fmm3d_corrected(interface, FT, UT, MO)
    F  = rhs_dielectric_box3d_fmm3d_batched(interface, si, g, FT)

    @testset "group interface builds" begin
        @test interface isa DielectricInterface
        @test length(interface.panels) >= 6
        env = BoundaryIntegral.envelope_volume_source(g)
        @test length(env.density) == size(g.densities, 1)
        @test all(env.density .≈ 1.0)
    end

    @testset "batched RHS == single-RHS (K=1 regression)" begin
        @test size(F) == (Np, 1)
        vs = VolumeSource(copy(g.positions), copy(g.weights), g.densities[:, 1])
        f_ref = rhs_dielectric_box3d_fmm3d(interface, vs, FT)
        @test maximum(abs.(F[:, 1] .- f_ref)) < 1e-6
    end

    @testset "batched matvec == columnwise (Unit 5)" begin
        ref = lhs_dielectric_box3d_fmm3d_corrected(interface, FT, UT, MO)
        x = collect(range(0.1, 1.0; length = Np))
        @test maximum(abs.(op * x .- ref * x)) < 1e-9
        X = hcat(x, reverse(x), fill(0.5, Np))
        Y = op * X
        @test size(Y) == size(X)
        for c in 1:size(X, 2)
            @test maximum(abs.(Y[:, c] .- ref * X[:, c])) < 1e-8
        end
    end

    @testset "block solve == looped single solves (Unit 6)" begin
        Σ, _ = BoundaryIntegral._block_gmres_solve(op, F; rtol = GTOL, itmax = 200)
        @test size(Σ) == size(F)
        for c in 1:size(F, 2)
            xc, _ = Krylov.gmres(op, F[:, c]; rtol = GTOL, itmax = 200)
            @test maximum(abs.(Σ[:, c] .- xc)) < 1e-6
        end
    end

    @testset "end-to-end .bie -> Sigma" begin
        out = solve_dielectric_box3d_group(
            joinpath(fixdir, "system_small.bie"), 1;
            n_quad = NQ, rhs_atol = RTOL_REFINE, l_ec = LEC,
            fmm_tol = FT, up_tol = UT, max_order = MO, rtol = GTOL)
        @test size(out.sigma, 2) == 1
        @test size(out.sigma, 1) == BoundaryIntegral.num_points(out.interface)
        @test all(isfinite, out.sigma)
    end

    # ---- K=2 system: build ONCE ----
    @testset "K=2 batched solve matches per-column" begin
        si2 = read_system_input(joinpath(fixdir, "system_pair.bie"))
        g2 = assemble_rhs_group(si2, 1)
        @test num_pairs(g2) == 2
        interface2 = build_group_interface(si2, g2; n_quad = NQ, rhs_atol = RTOL_REFINE, l_ec = LEC)
        op2 = batched_lhs_dielectric_box3d_fmm3d_corrected(interface2, FT, UT, MO)
        F2 = rhs_dielectric_box3d_fmm3d_batched(interface2, si2, g2, FT)
        @test size(F2, 2) == 2
        Σ2, _ = BoundaryIntegral._block_gmres_solve(op2, F2; rtol = GTOL, itmax = 300)
        for c in 1:2
            xc, _ = Krylov.gmres(op2, F2[:, c]; rtol = GTOL, itmax = 300)
            @test maximum(abs.(Σ2[:, c] .- xc)) < 1e-6
        end
    end
end
