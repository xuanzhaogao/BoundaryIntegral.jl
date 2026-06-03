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
end

@testset "density_centroid + centerless orbital" begin
    fixdir = joinpath(@__DIR__, "..", "fixtures")
    _, dg = BoundaryIntegral.read_xsf(joinpath(fixdir, "orb_a.xsf"))
    c = BoundaryIntegral.density_centroid(dg)
    @test all(isapprox.(c, (1.0, 1.0, 1.0); atol = 1e-12))   # all-ones on {0,2}^3 grid -> mean (1,1,1)
    si = read_system_input(joinpath(fixdir, "system_centroid.bie"))
    @test all(isapprox.(si.orbitals[1].center, (1.0, 1.0, 1.0); atol = 1e-12))
end

@testset "lattice-shifted orbital image" begin
    fixdir = joinpath(@__DIR__, "..", "fixtures")
    # orb_lat: 4x4x4, supercell mult 2 -> 2 grid steps per a1; spike at the (1,1,1) corner.
    si = read_system_input(joinpath(fixdir, "system_lat.bie"))
    @test si.orbitals[1].grid_shift == (0, 0, 0)
    @test si.orbitals[1].center == (0.0, 0.0, 0.0)         # spike at origin
    @test si.orbitals[2].grid_shift == (2, 0, 0)           # LATTICE 1 0 0 -> 2 grid steps
    @test all(isapprox.(si.orbitals[2].center, (1.0, 0.0, 0.0); atol = 1e-12))  # shifted by a1
    @test sort(si.groups[1]) == [1, 2]                     # both within cutoff
end
