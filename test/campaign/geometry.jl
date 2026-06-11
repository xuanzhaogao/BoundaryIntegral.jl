using BoundaryIntegral
using Test

@testset "geometry snap" begin
    fixdir = joinpath(@__DIR__, "..", "fixtures")
    # orb_lat: 4x4x4, true cell 2.0, step 0.5; primvec = I => 2 steps per unit length.
    st, dg = BoundaryIntegral.read_xsf(joinpath(fixdir, "orb_lat.xsf"))
    c0 = BoundaryIntegral.density_centroid(dg)   # spike at origin -> (0,0,0)

    # exact lattice point: centroid + (1,0,0) -> 2 steps along x
    @test BoundaryIntegral.snap_orbital(dg, c0, (c0[1] + 1.0, c0[2], c0[3])) == (2, 0, 0)
    @test BoundaryIntegral.snap_orbital(dg, c0, c0) == (0, 0, 0)
    # off-lattice by < half a step (0.24 < 0.5/2 = 0.25 in Cartesian) snaps to nearest
    @test BoundaryIntegral.snap_orbital(dg, c0, (c0[1] + 0.24, c0[2], c0[3])) == (0, 0, 0)
    @test BoundaryIntegral.snap_orbital(dg, c0, (c0[1] + 0.26, c0[2], c0[3])) == (1, 0, 0)
    @test BoundaryIntegral.snap_orbital(dg, c0, (c0[1], c0[2], c0[3] - 1.0)) == (0, 0, -2)
end

@testset "geometry snap: orb_smooth identity basis" begin
    fixdir = joinpath(@__DIR__, "..", "fixtures")
    st, dg = BoundaryIntegral.read_xsf(joinpath(fixdir, "orb_smooth.xsf"))
    c0 = BoundaryIntegral.density_centroid(dg)
    # orb_smooth: true cell 3.0, nx=6 -> step 0.5 -> a displacement of +0.5 = 1 step.
    @test BoundaryIntegral.snap_orbital(dg, c0, (c0[1] + 0.5, c0[2], c0[3])) == (1, 0, 0)
end
