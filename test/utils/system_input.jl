using BoundaryIntegral
using Test

@testset "system_input parser" begin
    fixdir = joinpath(@__DIR__, "..", "fixtures")
    si = read_system_input(joinpath(fixdir, "system_small.bie"))

    @test si.unit_scale == 1.0

    @test length(si.boxes) == 1
    @test si.boxes[1].center == (0.0, 0.0, 0.0)
    @test si.boxes[1].Lx == 4.0
    @test si.epses == [11.7]
    @test si.eps_out == 1.0

    @test si.orbitals[1].center == (0.0, 0.0, 0.0)
    @test si.orbitals[2].center == (5.0, 0.0, 0.0)
    @test endswith(si.orbitals[1].xsf_path, "orb_a.xsf")

    @test sort(si.groups[1]) == [1]
    @test sort(si.groups[2]) == [2]

    @test_throws ErrorException read_system_input(
        joinpath(fixdir, "system_multiatom_bad.bie"))
end
