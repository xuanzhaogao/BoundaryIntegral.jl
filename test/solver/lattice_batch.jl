# test/solver/lattice_batch.jl
using BoundaryIntegral
using Test
using LinearAlgebra

@testset "lattice_batch" begin
    fixdir = joinpath(@__DIR__, "..", "fixtures")

    @testset "lattice_grid_steps" begin
        st, dg = BoundaryIntegral.read_xsf(joinpath(fixdir, "orb_lat.xsf"))
        # orb_lat: 4x4x4 grid, span 1.5 (true cell 2.0, step 0.5), primvec = I => 2 steps/cell
        @test lattice_grid_steps(dg, st.primvec, (1, 0, 0)) == (2, 0, 0)
        @test lattice_grid_steps(dg, st.primvec, (0, -2, 1)) == (0, -4, 2)
        @test lattice_grid_steps(dg, st.primvec, (0, 0, 0)) == (0, 0, 0)
    end

    @testset "OrbitalInstance + frame overlap" begin
        inst = OrbitalInstance(7, 1, (2, 0, 0))
        @test inst.id == 7 && inst.template_id == 1 && inst.steps == (2, 0, 0)
        # frames of length 4 offset by 0 and 2 overlap on global indices 3:4
        @test BoundaryIntegral._frame_overlap(4, 0, 2) == 3:4
        @test BoundaryIntegral._frame_overlap(4, 2, 0) == 3:4
        @test BoundaryIntegral._frame_overlap(4, 0, 0) == 1:4
        @test BoundaryIntegral._frame_overlap(4, 0, 4) === nothing   # disjoint
        @test BoundaryIntegral._frame_overlap(4, 0, 9) === nothing
        @test BoundaryIntegral._frame_overlap(4, 0, -2) == 1:2   # negative offset
        @test BoundaryIntegral._frame_overlap(4, -2, 0) == 1:2   # symmetric
        @test BoundaryIntegral._frame_overlap(4, 0, 3) == 4:4    # single-point overlap
    end

    @testset "assemble_lattice_batch multi-anchor + disjoint pair" begin
        st, dg = BoundaryIntegral.read_xsf(joinpath(fixdir, "orb_smooth.xsf"))
        s1 = lattice_grid_steps(dg, st.primvec, (1, 0, 0))
        insts = Dict(1 => OrbitalInstance(1, 1, (0, 0, 0)),
                     2 => OrbitalInstance(2, 1, s1),
                     3 => OrbitalInstance(3, 1, (40, 0, 0)))   # far away: zero overlap with 1
        b = assemble_lattice_batch([dg], insts, [(1, 1), (2, 2), (1, 3)]; support_rtol = 1e-6)
        @test size(b.densities, 2) == 3
        @test all(b.densities[:, 3] .== 0.0)                    # disjoint frames -> zero column
        # column 2 is column 1 translated by s1: same multiset of values
        @test sort(b.densities[findall(!iszero, b.densities[:, 1]), 1]) ≈
              sort(b.densities[findall(!iszero, b.densities[:, 2]), 2])
        # gidx sorted & unique
        @test issorted(b.gidx) && allunique(b.gidx)
        # support_rtol = 0: shifted pair occupies SHIFTED global indices, union grows
        b0 = assemble_lattice_batch([dg], insts, [(1, 1), (2, 2)]; support_rtol = 0.0)
        gx1 = sort(unique(g[1] for (s, g) in enumerate(b0.gidx) if b0.densities[s, 1] != 0))
        gx2 = sort(unique(g[1] for (s, g) in enumerate(b0.gidx) if b0.densities[s, 2] != 0))
        @test gx2 == gx1 .+ 2          # pair (2,2) is pair (1,1) translated by s1=(2,0,0)
        @test length(b0.gidx) > count(!iszero, b0.densities[:, 1])   # union grew
    end

    # shared solved batch for evaluation tests (coarse)
    # Uses the .toml-based setup (system_smooth_lat.toml replaces system_smooth_lat.bie)
    c_e = load_campaign(joinpath(fixdir, "system_smooth_lat.toml"))
    st_e, dg_e = BoundaryIntegral.read_xsf(c_e.templates[1])
    insts_e = Dict(1 => OrbitalInstance(1, 1, (0, 0, 0)),
                   2 => OrbitalInstance(2, 1, lattice_grid_steps(dg_e, st_e.primvec, (1, 0, 0))))
    b_e = assemble_lattice_batch([dg_e], insts_e, [(1, 1), (1, 2)]; support_rtol = 1e-6)
    res_e = solve_dielectric_lattice_batch(c_e.boxes, c_e.epses, c_e.eps_out, b_e;
        n_quad = 4, rhs_atol = 1e-2, l_ec = 2.0, fmm_tol = 1e-6, gmres_rtol = 1e-8)

    @testset "evaluate_batch_potential vs TKM (near targets)" begin
        # targets: the batch's own support points only (all near — coarse fixture is under-resolved,
        # so only a near-only comparison gives an apples-to-apples reference with TKM)
        targets = b_e.positions
        far_pad = 2.0 * maximum(norm.(BoundaryIntegral.true_cell_vectors(dg_e))) / dg_e.nx

        Φ = evaluate_batch_potential(res_e.interface, res_e.sigma, res_e.sources, targets;
            lhs_tol = 1e-6, volume_tol = 1e-8, far_pad = far_pad)
        @test size(Φ) == (size(targets, 2), 2)

        # reference: TKM at near targets + the same layer map
        pottrg = laplace3d_pottrg_fmm3d_corrected_hcubature(res_e.interface, targets, 1e-6, 1e-6, 5.0)
        for a in 1:2
            sa = BoundaryIntegral.screened_volume_source(res_e.interface, res_e.sources[a],
                BoundaryIntegral.SharpScreening())
            vals = BoundaryIntegral.TKM3D.ltkm3dc(1e-8, sa.positions;
                charges = sa.weights .* sa.density, targets = targets, pgt = 1,
                kmax = BoundaryIntegral._estimate_tkm3dc_kmax(sa))
            @test vals.ier == 0
            Φ_ref = real.(vals.pottarg) .+ (pottrg * res_e.sigma[:, a])
            scale = maximum(abs.(Φ_ref))
            @test maximum(abs.(Φ[:, a] .- Φ_ref)) < 1e-5 * scale
        end
    end

    @testset "near/far split consistency (well-resolved source)" begin
        # 24^3 grid over the same 3.0 cell, sigma = 0.4 blob: ~3.2 points per sigma.
        nres = 24
        h = 3.0 / nres
        pts = Matrix{Float64}(undef, 3, nres^3)
        den = Vector{Float64}(undef, nres^3)
        m = 0
        for k in 1:nres, j in 1:nres, i in 1:nres
            x = (i - 0.5) * h; y = (j - 0.5) * h; z = (k - 0.5) * h
            m += 1
            pts[1, m] = x; pts[2, m] = y; pts[3, m] = z
            den[m] = exp(-((x - 1.5)^2 + (y - 1.5)^2 + (z - 1.5)^2) / (2 * 0.4^2))
        end
        vs_res = VolumeSource(pts, fill(h^3, nres^3), den)

        far = hcat(([8.0 * cos(t) + 1.5, 8.0 * sin(t) + 1.5, 1.5] for t in range(0, 2π; length = 17)[1:16])...)
        targets = hcat(pts[:, 1:97:end], far)          # subsample of near points + far ring
        far_pad = 2.0 * h

        Σ0 = zeros(BoundaryIntegral.num_points(res_e.interface), 1)   # layer part off: pure u_inc test
        Φ = evaluate_batch_potential(res_e.interface, Σ0, [vs_res], targets;
            lhs_tol = 1e-6, volume_tol = 1e-8, far_pad = far_pad)

        sa = BoundaryIntegral.screened_volume_source(res_e.interface, vs_res,
            BoundaryIntegral.SharpScreening())
        vals = BoundaryIntegral.TKM3D.ltkm3dc(1e-8, sa.positions;
            charges = sa.weights .* sa.density, targets = targets, pgt = 1,
            kmax = BoundaryIntegral._estimate_tkm3dc_kmax(sa))
        @test vals.ier == 0
        Φ_ref = real.(vals.pottarg)
        scale = maximum(abs.(Φ_ref))
        max_rel_diff = maximum(abs.(Φ[:, 1] .- Φ_ref)) / scale
        @info "near/far split consistency: max_rel_diff = $max_rel_diff"
        @test max_rel_diff < 1e-5

        # K = 2 with a genuine near/far split (nd = 2 far reshape path)
        den2 = den .* (pts[1, :] .- 1.5).^2                    # second, distinct density
        vs_res2 = VolumeSource(copy(pts), fill(h^3, nres^3), den2)
        Σ0_2 = zeros(BoundaryIntegral.num_points(res_e.interface), 2)
        Φ2 = evaluate_batch_potential(res_e.interface, Σ0_2, [vs_res, vs_res2], targets;
            lhs_tol = 1e-6, volume_tol = 1e-8, far_pad = far_pad)
        for (a, v) in enumerate((vs_res, vs_res2))
            sa2 = BoundaryIntegral.screened_volume_source(res_e.interface, v,
                BoundaryIntegral.SharpScreening())
            vals2 = BoundaryIntegral.TKM3D.ltkm3dc(1e-8, sa2.positions;
                charges = sa2.weights .* sa2.density, targets = targets, pgt = 1,
                kmax = BoundaryIntegral._estimate_tkm3dc_kmax(sa2))
            @test vals2.ier == 0
            ref = real.(vals2.pottarg)
            max_rel_diff2 = maximum(abs.(Φ2[:, a] .- ref)) / maximum(abs.(ref))
            @info "K=2 near/far split, column $a: max_rel_diff = $max_rel_diff2"
            @test maximum(abs.(Φ2[:, a] .- ref)) < 1e-5 * maximum(abs.(ref))
        end

        # all-far branch: only the distant ring as targets
        Φf = evaluate_batch_potential(res_e.interface, Σ0_2, [vs_res, vs_res2], far;
            lhs_tol = 1e-6, volume_tol = 1e-8, far_pad = far_pad)
        @test size(Φf) == (size(far, 2), 2)
        @test all(isfinite, Φf)

        # mismatched source positions are rejected
        vs_bad = VolumeSource(pts .+ 0.1, fill(h^3, nres^3), den2)
        @test_throws ArgumentError evaluate_batch_potential(res_e.interface, Σ0_2,
            [vs_res, vs_bad], far; lhs_tol = 1e-6, volume_tol = 1e-8, far_pad = far_pad)
    end

    @testset "V via evaluate_batch_potential == four_index_matrix" begin
        V_ref = four_index_matrix(res_e.interface, res_e.sources, res_e.sigma;
                                  lhs_tol = 1e-6, volume_tol = 1e-8)
        targets = b_e.positions                        # the group grid = what four_index uses
        far_pad = 2.0 * maximum(norm.(BoundaryIntegral.true_cell_vectors(dg_e))) / dg_e.nx
        Φ = evaluate_batch_potential(res_e.interface, res_e.sigma, res_e.sources, targets;
            lhs_tol = 1e-6, volume_tol = 1e-8, far_pad = far_pad)
        K = 2
        V = [LinearAlgebra.dot(b_e.weights .* b_e.densities[:, a], Φ[:, bb]) for a in 1:K, bb in 1:K]
        @test maximum(abs.(V .- V_ref)) < 1e-6 * maximum(abs.(V_ref))
    end
end
