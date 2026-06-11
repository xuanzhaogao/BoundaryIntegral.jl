using BoundaryIntegral, Distributed, SlurmClusterManager, Test
include("fixture_campaign.jl")

@testset "run_phase local workers" begin
    mktempdir() do dir
        c = load_campaign(write_fixture_campaign(dir))
        prepare(c)
        run_phase(c, :solve; workers = 2)         # local addprocs(2)
        @test isempty(pending_batches(c, :solve))
        consolidate(c)
        run_phase(c, :eval; workers = 2)
        @test isempty(pending_batches(c, :eval))
        rmprocs(workers())                         # clean up
    end
end
