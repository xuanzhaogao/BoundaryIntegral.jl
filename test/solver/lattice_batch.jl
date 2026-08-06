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

    @testset "evaluate_batch_potential near-branch wiring" begin
        # NOT a TKM/physics-agreement check (see the name change from the original
        # "... vs TKM (near targets)"). Since GATE 2a this testset's reference calls
        # PrecomputedVolumeField on the same screened source at the same c_pad as
        # the code under test -- i.e. it is the same discretization, not an
        # independent one. What it DOES still check: that evaluate_batch_potential's
        # per-column loop selects the right source for each column, applies the
        # right screening, and adds the layer potential to the right target
        # indices/column -- a swapped source, a missed screening step, or a
        # column/index mixup would still be caught here even though a genuine
        # TKM-vs-spectral agreement check would not be. GATE 2's actual TKM
        # agreement numbers (measured, not asserted here) are in the c_pad-bound
        # testset below.
        targets = b_e.positions

        Φ = evaluate_batch_potential(res_e.interface, res_e.sigma, res_e.sources, targets;
            lhs_tol = 1e-6, volume_tol = 1e-8, c_pad = 5.0)
        @test size(Φ) == (size(targets, 2), 2)

        # reference: PrecomputedVolumeField at near targets + the same layer map, NOT
        # TKM3D.ltkm3dc. ltkm3dc derives its Fourier box from the combined
        # source+target bounding box, which is deliberately a *different* (also
        # valid) discretization from Eq. (3.11)/(3.16). On this fixture's 2-3-cell
        # source box the two discretizations disagree by ~8% (see GATE 2 note in
        # the c_pad-bound testset below) — not because either is wrong, but because
        # comparing them here would be checking two discretizations against each
        # other rather than checking the code under test. Reference against the
        # same near_field_geometry the code under test uses instead.
        pottrg = laplace3d_pottrg_fmm3d_corrected_hcubature(res_e.interface, targets, 1e-6, 1e-6, 5.0)
        for a in 1:2
            sa = BoundaryIntegral.screened_volume_source(res_e.interface, res_e.sources[a],
                BoundaryIntegral.SharpScreening())
            fld = BoundaryIntegral.PrecomputedVolumeField(sa;
                tol = 1e-8, c_pad = 5.0, compute_grad = false)
            Φ_inc = BoundaryIntegral.volume_field_potential(fld, targets)
            Φ_ref = Φ_inc .+ (pottrg * res_e.sigma[:, a])
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

        Σ0 = zeros(BoundaryIntegral.num_points(res_e.interface), 1)   # layer part off: pure u_inc test
        Φ = evaluate_batch_potential(res_e.interface, Σ0, [vs_res], targets;
            lhs_tol = 1e-6, volume_tol = 1e-8, c_pad = 5.0)

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
            lhs_tol = 1e-6, volume_tol = 1e-8, c_pad = 5.0)
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
            lhs_tol = 1e-6, volume_tol = 1e-8, c_pad = 5.0)
        @test size(Φf) == (size(far, 2), 2)
        @test all(isfinite, Φf)

        # mismatched source positions are rejected
        vs_bad = VolumeSource(pts .+ 0.1, fill(h^3, nres^3), den2)
        @test_throws ArgumentError evaluate_batch_potential(res_e.interface, Σ0_2,
            [vs_res, vs_bad], far; lhs_tol = 1e-6, volume_tol = 1e-8, c_pad = 5.0)
    end

    @testset "V via evaluate_batch_potential == four_index_matrix" begin
        # NOT an independent cross-check (see src/solver/multi_rhs.jl's
        # `four_index_matrix` docstring, which points here). Since Task 5's
        # migration, both sides build the incident term the same way --
        # `PrecomputedVolumeField` on the same screened source at the same
        # `c_pad` -- and both call `laplace3d_pottrg_fmm3d_corrected_hcubature`
        # with identical arguments for the layer term. The incident and layer
        # code paths are therefore identical, and the measured difference
        # between `V` and `V_ref` below is exactly 0.0 by construction, not an
        # agreement demonstrated by this test. What this test still genuinely
        # verifies is contraction index ordering and column mapping: `V` is
        # hand-rolled here as `dot(weights .* densities[:, a], Φ[:, bb])`
        # while `four_index_matrix` computes its own `V[a, b]` internally, so
        # a transposed or mis-mapped `V[a, b]` would still be caught here --
        # `V` is not symmetric, so such a bug would not cancel out. The 1e-6
        # tolerance is kept as a harmless upper bound on the (exactly zero)
        # difference.
        V_ref = four_index_matrix(res_e.interface, res_e.sources, res_e.sigma;
                                  lhs_tol = 1e-6, volume_tol = 1e-8)
        targets = b_e.positions                        # the group grid = what four_index uses
        Φ = evaluate_batch_potential(res_e.interface, res_e.sigma, res_e.sources, targets;
            lhs_tol = 1e-6, volume_tol = 1e-8, c_pad = 5.0)
        K = 2
        V = [LinearAlgebra.dot(b_e.weights .* b_e.densities[:, a], Φ[:, bb]) for a in 1:K, bb in 1:K]
        @test maximum(abs.(V .- V_ref)) < 1e-6 * maximum(abs.(V_ref))
    end

    @testset "evaluate_batch_potential geometry and the c_pad 2-vs-5 bound" begin
        # LatticeBatch propagates the lattice basis, so h = ||A_rho||_2 is available.
        @test b_e.lattice_basis !== nothing
        for v in batch_volume_sources(b_e)
            @test v.lattice_basis == b_e.lattice_basis
        end
        @test BoundaryIntegral.envelope_volume_source(b_e).lattice_basis == b_e.lattice_basis

        # The batch split agrees with the shared helper on the screened source.
        targets = b_e.positions
        sa = BoundaryIntegral.screened_volume_source(res_e.interface, res_e.sources[1],
            BoundaryIntegral.SharpScreening())
        g = BoundaryIntegral.near_field_geometry(sa; c_pad = 5.0)
        @test all(BoundaryIntegral.in_near_region(g, targets, i) for i in 1:size(targets, 2))

        # Bound the un-rerun Section 6.4 change: how much does c_pad 2 -> 5 move Phi?
        # This is measured on `b_e`, the coarse fixture whose own comment (line
        # 64-65 above) says it is under-resolved: its source box is only 2-3 grid
        # cells per axis (h = 0.5, source box l = (1.5, 1.0, 1.0)). On a source that
        # coarse, changing c_pad changes h_n = c_pad*h and therefore the Fourier box
        # (L, dk of Eqs. 3.11/3.16) used by PrecomputedVolumeField, even for targets
        # that never change near/far classification -- decomposing the diff shows
        # the always-near batch-support points alone account for the full bound
        # (only 1 of 20 mixed targets actually flips classification between c_pad=2
        # and c_pad=5, and that flipped target contributes only ~1.7e-4 of the total;
        # the rest comes from the Fourier-box change on the 19 never-reclassified
        # points). This is a real, measured effect of a 2-3-cell-wide source, not a
        # bug -- kept below as a regression sentinel, NOT as the production gate
        # (see the well-resolved measurement immediately after, which is the gate).
        far = hcat(([6.0 * cos(t), 6.0 * sin(t), 0.0] for t in range(0, 2π; length = 9)[1:8])...)
        mixed = hcat(targets, far)
        Φ2 = evaluate_batch_potential(res_e.interface, res_e.sigma, res_e.sources, mixed;
            lhs_tol = 1e-6, volume_tol = 1e-8, c_pad = 2.0)
        Φ5 = evaluate_batch_potential(res_e.interface, res_e.sigma, res_e.sources, mixed;
            lhs_tol = 1e-6, volume_tol = 1e-8, c_pad = 5.0)
        scale = maximum(abs.(Φ5))
        bound = maximum(abs.(Φ2 .- Φ5)) / scale
        @info "c_pad 2 vs 5 (COARSE fixture, regression sentinel only): max relative difference in Phi = $bound"
        # Not the production gate (see below) -- a documented sentinel so a jump to
        # e.g. 0.1 on this pathologically coarse fixture would still get noticed.
        # Measured 0.0038; 1e-2 leaves headroom without being vacuous.
        @test bound < 1e-2

        # THE GATE: the same measurement on a well-resolved source (24^3 grid over
        # a 3.0 cell, sigma = 0.4 blob, ~3.2 points per sigma -- same source as the
        # "near/far split consistency" testset above), pure u_inc (Sigma = 0 so the
        # layer potential contributes nothing), on a target set built to genuinely
        # straddle the near/far boundary at BOTH c_pad settings: a deep-near subset
        # (always near), 6 face-centered points offset from the source box by the
        # midpoint of h_n(c_pad=2) and h_n(c_pad=5) (guaranteed to flip), and a
        # deep-far ring (always far). Section 6.4's production data (150x150x192
        # grid over a 12.24 A cell, graphene Wannier orbitals ~1-2 A wide) is
        # better resolved than this (~9-12 points per orbital width vs ~3.2 points
        # per sigma here), so this is a conservative upper bound on the real effect.
        nres_g = 24
        hg = 3.0 / nres_g
        pts_g = Matrix{Float64}(undef, 3, nres_g^3)
        den_g = Vector{Float64}(undef, nres_g^3)
        mg = 0
        for k in 1:nres_g, j in 1:nres_g, i in 1:nres_g
            x = (i - 0.5) * hg; y = (j - 0.5) * hg; z = (k - 0.5) * hg
            mg += 1
            pts_g[1, mg] = x; pts_g[2, mg] = y; pts_g[3, mg] = z
            den_g[mg] = exp(-((x - 1.5)^2 + (y - 1.5)^2 + (z - 1.5)^2) / (2 * 0.4^2))
        end
        vs_g = VolumeSource(pts_g, fill(hg^3, nres_g^3), den_g)
        sa_g = BoundaryIntegral.screened_volume_source(res_e.interface, vs_g,
            BoundaryIntegral.SharpScreening())
        g2_g = BoundaryIntegral.near_field_geometry(sa_g; c_pad = 2.0)
        g5_g = BoundaryIntegral.near_field_geometry(sa_g; c_pad = 5.0)
        loB_g, hiB_g, _ = BoundaryIntegral.source_box(sa_g)
        hn_mid_g = (g2_g.hn + g5_g.hn) / 2
        c_g = g2_g.center
        straddle_g = Matrix{Float64}(undef, 3, 6)
        straddle_g[:, 1] = [loB_g[1] - hn_mid_g, c_g[2], c_g[3]]
        straddle_g[:, 2] = [hiB_g[1] + hn_mid_g, c_g[2], c_g[3]]
        straddle_g[:, 3] = [c_g[1], loB_g[2] - hn_mid_g, c_g[3]]
        straddle_g[:, 4] = [c_g[1], hiB_g[2] + hn_mid_g, c_g[3]]
        straddle_g[:, 5] = [c_g[1], c_g[2], loB_g[3] - hn_mid_g]
        straddle_g[:, 6] = [c_g[1], c_g[2], hiB_g[3] + hn_mid_g]
        near_sub_g = pts_g[:, 1:97:end]
        far_ring_g = hcat(([8.0 * cos(t) + 1.5, 8.0 * sin(t) + 1.5, 1.5]
            for t in range(0, 2π; length = 17)[1:16])...)
        targets_g = hcat(near_sub_g, straddle_g, far_ring_g)
        nflip_g = count(i -> BoundaryIntegral.in_near_region(g2_g, targets_g, i) !=
                             BoundaryIntegral.in_near_region(g5_g, targets_g, i), 1:size(targets_g, 2))
        @test nflip_g > 0   # the straddle points must actually flip classification

        Σ0_g = zeros(BoundaryIntegral.num_points(res_e.interface), 1)
        Φ2_g = evaluate_batch_potential(res_e.interface, Σ0_g, [vs_g], targets_g;
            lhs_tol = 1e-6, volume_tol = 1e-8, c_pad = 2.0)
        Φ5_g = evaluate_batch_potential(res_e.interface, Σ0_g, [vs_g], targets_g;
            lhs_tol = 1e-6, volume_tol = 1e-8, c_pad = 5.0)
        scale_g = maximum(abs.(Φ5_g))
        bound_g = maximum(abs.(Φ2_g .- Φ5_g)) / scale_g
        @info "c_pad 2 vs 5 (WELL-RESOLVED source, THE GATE): max relative difference in Phi = $bound_g"
        # Measured 1.7e-7; 1e-5 leaves ample headroom without being vacuous.
        # This bounds only the c_pad component of the un-rerun Section 6.4
        # change (2 -> 5, on a well-resolved source). It is NOT the whole
        # story: the near branch also swapped evaluators, from a direct
        # TKM3D.ltkm3dc call (Fourier box from the combined source+target
        # bounding box) to PrecomputedVolumeField (the paper's Eq. 3.11/3.16
        # box). That second, separate component is bounded at <1e-5 by the
        # "near/far split consistency (well-resolved source)" testset above.
        # The conclusion (Section 6.4 need not be rerun) is unchanged either
        # way: both components are far below Section 6.4's quoted precision
        # (its symmetry residual is 4.3e-3), giving ~400x margin on the
        # looser of the two (1e-5 vs 4.3e-3).
        @test bound_g < 1e-5

        # far_pad is gone
        @test_throws MethodError evaluate_batch_potential(res_e.interface, res_e.sigma,
            res_e.sources, targets; lhs_tol = 1e-6, volume_tol = 1e-8, far_pad = 0.1)
    end
end
