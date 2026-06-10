# test/solver/lattice_batch.jl
using BoundaryIntegral
using Test

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
    end
end
