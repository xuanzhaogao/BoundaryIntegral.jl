using BoundaryIntegral
using Test

@testset "manifest" begin
    centers = BoundaryIntegral.CenterInfo[
        BoundaryIntegral.CenterInfo(1, 1, (0,0,0),   (0.0, 0.0, 0.0)),
        BoundaryIntegral.CenterInfo(2, 2, (0,0,0),   (0.5, 0.5, 0.0)),
        BoundaryIntegral.CenterInfo(3, 1, (2,0,0),   (1.0, 0.0, 0.0)),
    ]
    pairs = enumerate_pairs(centers, 0.8)        # on-site + the d=0.707 (1,2) pair; not (1,3)@1.0
    @test all(p -> p[1] <= p[2], pairs)
    @test (1, 1) in pairs && (2, 2) in pairs && (3, 3) in pairs && (1, 2) in pairs
    @test !((1, 3) in pairs)

    batches = build_batches(pairs, 1)
    @test sum(b -> length(b.pairs), batches) == length(pairs)
    @test allunique(getfield.(batches, :batch_id))

    mktempdir() do dir
        cp = joinpath(dir, "centers.tsv"); mp = joinpath(dir, "manifest.tsv")
        write_centers(cp, centers); write_manifest(mp, batches)
        @test read_centers(cp) == centers
        @test read_manifest(mp) == batches
    end
end
