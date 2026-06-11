# test/solver/batch_io.jl
using BoundaryIntegral
using Test

@testset "batch_io" begin
    br = BatchResult(BoundaryIntegral.BATCH_FORMAT_VERSION, 7,
        [(1, 1), (1, 2)],
        [(1, 1, 1), (2, 1, 1)],
        [0.1, 0.1],
        [1.0 0.0; 0.5 0.25],
        nothing,                          # interface: any serializable object
        ones(4, 2),
        Dict{String,Any}("niter" => 3, "t_solve" => 1.5))

    mktempdir() do dir
        p = joinpath(dir, "batch_0007.jls")
        @test !is_complete_batch(p)                       # missing file
        save_batch_result(p, br)
        @test is_complete_batch(p)
        @test isempty(filter(f -> occursin(".tmp", f), readdir(dir)))   # tmp cleaned up
        br2 = load_batch_result(p)
        @test br2.batch_id == 7
        @test br2.pair_ids == br.pair_ids
        @test br2.gidx == br.gidx
        @test br2.densities == br.densities
        @test br2.sigma == br.sigma
        @test br2.stats["niter"] == 3

        # corrupted file -> incomplete, load throws
        open(p, "w") do io; write(io, "garbage"); end
        @test !is_complete_batch(p)
        @test_throws Exception load_batch_result(p)

        # wrong version -> incomplete
        bad = BatchResult(BoundaryIntegral.BATCH_FORMAT_VERSION + 1, br.batch_id,
            br.pair_ids, br.gidx, br.weights, br.densities, br.interface, br.sigma, br.stats)
        save_batch_result(p, bad)
        @test !is_complete_batch(p)
    end
end
