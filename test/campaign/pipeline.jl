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
        @test rep.max_rel_asym < 0.2   # coarse solve; 5.7% observed; sanity only
        @test isfile(joinpath(c.root, "V_full.jls"))

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
