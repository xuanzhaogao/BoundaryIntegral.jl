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
        # V blocks are v2 and now cover EVERY row (the symmetry-restricted variant was removed
        # as a net loss), so placing by `rows` must fill the matrix outright. The mirror below
        # is kept only to reproduce assemble_v's reconstruction exactly.
        Vb = fill(NaN, length(pid), length(pid))
        for b in read_manifest(manifest_path(c))
            vr = BoundaryIntegral.load_v_rows(v_path(c, b.batch_id))
            @test length(vr.rows) == size(vr.V, 1)
            @test vr.target_pairs == pid[vr.rows]
            for (k, sp) in enumerate(vr.source_pairs)
                Vb[vr.rows, col[sp]] = vr.V[:, k]
            end
        end
        n_before = count(isnan, Vb)
        for i in 1:length(pid), j in 1:length(pid)
            isnan(Vb[i, j]) && !isnan(Vb[j, i]) && (Vb[i, j] = Vb[j, i])
        end
        @test !any(isnan, Vb)
        @test n_before == 0           # every entry computed directly; nothing left to mirror
        @test res.pair_ids == pid
        @test maximum(abs.(res.V .- Vb)) < 1e-8 * max(maximum(abs.(Vb)), eps())
        # Symmetry is now a REAL check rather than a tautology: V[i,j] and V[j,i] come from
        # different batches, hence different interfaces and sigmas, so they agree only to solver
        # accuracy. (Under the removed symmetry-restricted path one of the two was a copy of the
        # other, and this held to 1e-8 by construction -- it was testing nothing.)
        @test maximum(abs.(res.V .- transpose(res.V))) < 1e-2 * max(maximum(abs.(res.V)), eps())
    end
end

# "eval_batch_core propagates lattice_basis on a skewed grid" moved to
# test/campaign/tasks.jl (unconditional -- this file only runs under
# BI_RUN_FULL_TESTS=1, and that regression guard needs to run in CI).
