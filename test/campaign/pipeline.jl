using BoundaryIntegral, Serialization, LinearAlgebra, Test
include("fixture_campaign.jl")

@testset "prepare + serial phases + in-memory" begin
    mktempdir() do dir
        c = load_campaign(write_fixture_campaign(dir))
        prepare(c)
        @test isfile(manifest_path(c)) && isfile(centers_path(c))
        @test sort(pending_batches(c, :solve)) == [1, 2]
        for id in [1, 2]; solve_batch(c, id); end
        @test isempty(pending_batches(c, :solve))
        consolidate(c)
        for id in [1, 2]; eval_batch(c, id); end
        rep = assemble_v(c)
        @test rep.max_rel_asym < 1e-2
        @test isfile(joinpath(c.root, "V_full.tsv"))

        # in-memory path: same campaign (new dir so no files), compare to batched V files
        res = four_index_integrals(c)
        # rebuild dense V from the per-batch V files
        store = open(deserialize, rho_store_path(c))
        pid = store.pair_ids
        col = Dict(p => i for (i, p) in enumerate(pid))
        Vb = fill(NaN, length(pid), length(pid))
        for b in read_manifest(manifest_path(c))
            vr = BoundaryIntegral.load_v_rows(v_path(c, b.batch_id))
            for (k, sp) in enumerate(vr.source_pairs)
                Vb[:, col[sp]] = vr.V[:, k]
            end
        end
        @test res.pair_ids == pid
        @test maximum(abs.(res.V .- Vb)) < 1e-8 * max(maximum(abs.(Vb)), eps())
    end
end

@testset "eval_batch_core propagates lattice_basis on a skewed grid" begin
    # Regression guard for tasks.jl's `lb = ((At[1]/nx, ...), ...)` /
    # `lattice_basis = lb` construction (currently around lines 309-314): every
    # OTHER campaign fixture in this test suite (write_fixture_campaign,
    # system_smooth_lat.toml, etc.) is near-cubic, where ||A_rho||_2 and the
    # min-nearest-neighbour spacing nearly coincide -- so a silent regression to
    # `lattice_basis = nothing` there would pass every existing test. This uses a
    # hexagonal in-plane basis (two primitive vectors at 120 deg, equal length,
    # like the production graphene lattice) instead, where the two `h` estimates
    # genuinely differ (checked below), so a dropped basis is actually detectable.
    nx = ny = nz = 4
    a1 = (1.0, 0.0, 0.0)
    a2 = (-0.5, sqrt(3) / 2, 0.0)          # 120 deg from a1, |a2| == |a1|
    a3 = (0.0, 0.0, 1.0)
    # read_xsf's DATAGRID header vectors span (n-1) steps (closed-endpoint XCrySDen
    # convention); true_cell_vectors rescales by n/(n-1) back up to n .* a_i. This
    # hand-builds the same plain NamedTuple read_xsf returns (nx/ny/nz/origin/A/B/C/
    # values -- see src/utils/xsf_reader.jl) with header vectors (n-1).*a_i, so the
    # recovered true cell vectors come out to exactly n .* a_i and the primitive
    # step vectors are exactly a1, a2, a3 -- no XSF file needed.
    dg_hex = (nx = nx, ny = ny, nz = nz, origin = (0.0, 0.0, 0.0),
              A = (nx - 1) .* a1, B = (ny - 1) .* a2, C = (nz - 1) .* a3,
              values = [exp(-((i - 2.5)^2 + (j - 2.5)^2 + (k - 2.5)^2) / 2.0)
                        for i in 1:nx, j in 1:ny, k in 1:nz])

    At, Bt, Ct = BoundaryIntegral.true_cell_vectors(dg_hex)
    @test all(isapprox.(At, nx .* a1)) && all(isapprox.(Bt, ny .* a2)) && all(isapprox.(Ct, nz .* a3))
    lb_correct = (Tuple(At ./ nx), Tuple(Bt ./ ny), Tuple(Ct ./ nz))   # what eval_batch_core should compute

    A_mat = hcat(collect(lb_correct[1]), collect(lb_correct[2]), collect(lb_correct[3]))
    h_aniso = LinearAlgebra.opnorm(A_mat, 2)

    insts_hex = Dict(1 => OrbitalInstance(1, 1, (0, 0, 0)),
                      2 => OrbitalInstance(2, 1, (2, 0, 0)))
    b_hex = assemble_lattice_batch([dg_hex], insts_hex, [(1, 1), (1, 2)]; support_rtol = 1e-6)

    # the isotropic fallback _estimate_source_spacing (min nearest-neighbour distance)
    # a source with no lattice_basis would fall back to -- probed on a basis-less copy
    # of the same points, NOT through eval_batch_core.
    vs_probe = VolumeSource(copy(b_hex.positions), copy(b_hex.weights), b_hex.densities[:, 1])
    h_iso = BoundaryIntegral._estimate_source_spacing(vs_probe)
    @info "eval_batch_core skewed-grid regression test: h_aniso=$h_aniso h_iso=$h_iso ratio=$(h_aniso / h_iso)"
    @test h_aniso / h_iso > 1.05   # genuinely different -- the property near-cubic fixtures lack

    boxes_hex = [(center = (1.5, 1.5, 1.5), Lx = 8.0, Ly = 8.0, Lz = 8.0)]
    epses_hex = [2.0]
    eps_out_hex = 1.0
    res_hex = solve_dielectric_lattice_batch(boxes_hex, epses_hex, eps_out_hex, b_hex;
        n_quad = 4, rhs_atol = 1e-2, l_ec = 2.0, fmm_tol = 1e-6, gmres_rtol = 1e-8)

    br_hex = BoundaryIntegral.BatchResult(BoundaryIntegral.BATCH_FORMAT_VERSION, 1,
        b_hex.pair_ids, b_hex.gidx, b_hex.weights, b_hex.densities,
        res_hex.interface, res_hex.sigma, Dict{String,Any}())
    c_hex = CampaignInput("hex_test", "", String[], OrbitalSpec[], Inf, nothing,
        eps_out_hex, boxes_hex, epses_hex,
        Dict("lhs_tol" => 1e-6, "volume_tol" => 1e-8), 1, 5.0, "")

    targets_hex, store_hex = BoundaryIntegral.consolidate_core([br_hex], dg_hex, c_hex)
    pair_ids_out, V_hex = BoundaryIntegral.eval_batch_core(br_hex, targets_hex, store_hex, dg_hex, c_hex)

    # Reference: hand-build the VolumeSources with the SAME correctly-derived
    # lattice_basis, independently of eval_batch_core's own construction (this is
    # the "assert on the VolumeSources it constructs" the review asked for, done via
    # the observable V rather than by reaching into eval_batch_core's local
    # variables). If tasks.jl's `lattice_basis = lb` regresses to `lattice_basis =
    # nothing`, eval_batch_core's V will diverge from this reference -- a different
    # h means a different Section 3 Fourier box (h_n, L, dk) means a different Phi
    # -- and the comparison below will fail well outside the tight tolerance.
    K = length(br_hex.pair_ids)
    pos_hex = BoundaryIntegral.grid_positions(dg_hex, br_hex.gidx)
    sources_ref = [VolumeSource(copy(pos_hex), copy(br_hex.weights), br_hex.densities[:, k];
                                lattice_basis = lb_correct) for k in 1:K]
    Φ_ref = evaluate_batch_potential(br_hex.interface, br_hex.sigma, sources_ref, targets_hex.positions;
        lhs_tol = 1e-6, volume_tol = 1e-8, c_pad = c_hex.c_pad,
        screen_boxes = c_hex.boxes, screen_epses = c_hex.epses, screen_eps_out = c_hex.eps_out)
    nP = length(store_hex.pair_ids)
    V_ref = Matrix{Float64}(undef, nP, K)
    for kl in 1:nP, a in 1:K
        V_ref[kl, a] = LinearAlgebra.dot(store_hex.tw[kl], view(Φ_ref, store_hex.t_idx[kl], a))
    end

    @test pair_ids_out == br_hex.pair_ids
    @test maximum(abs.(V_hex .- V_ref)) < 1e-9 * maximum(abs.(V_ref))
end
