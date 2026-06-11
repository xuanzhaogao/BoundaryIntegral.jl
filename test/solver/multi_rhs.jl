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

    # ---- K=1 system: build the interface / operator / RHS ONCE via .toml ----
    # system_small.toml: orb_a.xsf at (0,0,0), single on-site pair (1,1)
    c = load_campaign(joinpath(fixdir, "system_small.toml"))
    st1, dg1 = BoundaryIntegral.read_xsf(c.templates[1])
    insts = Dict(1 => OrbitalInstance(1, 1, (0, 0, 0)))
    b = assemble_lattice_batch([dg1], insts, [(1, 1)]; support_rtol = 1e-6)
    res = solve_dielectric_lattice_batch(c.boxes, c.epses, c.eps_out, b;
        n_quad = NQ, rhs_atol = RTOL_REFINE, l_ec = LEC, fmm_tol = FT, gmres_rtol = GTOL)
    interface = res.interface
    vss = BoundaryIntegral.batch_volume_sources(b)
    Np = BoundaryIntegral.num_points(interface)
    op = batched_lhs_dielectric_box3d_fmm3d_corrected(interface, FT, UT, MO)
    F  = rhs_dielectric_box3d_fmm3d(interface, vss, FT)

    @testset "batched RHS == single-RHS (K=1 regression)" begin
        @test size(F) == (Np, 1)
        f_ref = rhs_dielectric_box3d_fmm3d(interface, vss[1], FT)
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
end
