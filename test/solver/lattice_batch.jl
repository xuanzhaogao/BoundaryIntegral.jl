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
        @test BoundaryIntegral._frame_overlap(4, 0, -2) == 1:2   # negative offset
        @test BoundaryIntegral._frame_overlap(4, -2, 0) == 1:2   # symmetric
        @test BoundaryIntegral._frame_overlap(4, 0, 3) == 4:4    # single-point overlap
    end

    @testset "assemble_lattice_batch == assemble_rhs_group (no-wrap regime)" begin
        si = read_system_input(joinpath(fixdir, "system_smooth_lat.bie"))
        g = assemble_rhs_group(si, 1; support_rtol = 1e-6)   # pairs (1,1),(1,2) via circshift

        st, dg = BoundaryIntegral.read_xsf(joinpath(fixdir, "orb_smooth.xsf"))
        templates = [dg]
        insts = Dict(1 => OrbitalInstance(1, 1, (0, 0, 0)),
                     2 => OrbitalInstance(2, 1, lattice_grid_steps(dg, st.primvec, (1, 0, 0))))
        b = assemble_lattice_batch(templates, insts, [(1, 1), (1, 2)]; support_rtol = 1e-6)

        @test b.pair_ids == [(1, 1), (1, 2)]
        @test size(b.densities) == (length(b.gidx), 2)
        @test size(b.positions) == (3, length(b.gidx))
        @test length(b.weights) == length(b.gidx)
        # same physics as the circshift group (blob tails that wrap are ~exp(-31), below rtol):
        # compare per-position values
        function posmap(P, D)
            d = Dict{NTuple{3,Float64},Vector{Float64}}()
            for s in 1:size(P, 2)
                d[(round(P[1,s]; digits=9), round(P[2,s]; digits=9), round(P[3,s]; digits=9))] = D[s, :]
            end
            d
        end
        da, db = posmap(g.positions, g.densities), posmap(b.positions, b.densities)
        @test Set(keys(da)) == Set(keys(db))
        @test maximum(maximum(abs.(da[k] .- db[k])) for k in keys(da)) < 1e-10
        # weights are the uniform cell weight
        @test all(b.weights .≈ g.weights[1])
    end

    @testset "assemble_lattice_batch multi-anchor + disjoint pair" begin
        st, dg = BoundaryIntegral.read_xsf(joinpath(fixdir, "orb_smooth.xsf"))
        s1 = lattice_grid_steps(dg, st.primvec, (1, 0, 0))
        insts = Dict(1 => OrbitalInstance(1, 1, (0, 0, 0)),
                     2 => OrbitalInstance(2, 1, s1),
                     3 => OrbitalInstance(3, 1, (40, 0, 0)))   # far away: zero overlap with 1
        b = assemble_lattice_batch([dg], insts, [(1, 1), (2, 2), (1, 3)]; support_rtol = 1e-6)
        @test size(b.densities, 2) == 3
        @test all(b.densities[:, 3] .== 0.0)                    # disjoint frames -> zero column
        # column 2 is column 1 translated by s1: same multiset of values
        @test sort(b.densities[findall(!iszero, b.densities[:, 1]), 1]) ≈
              sort(b.densities[findall(!iszero, b.densities[:, 2]), 2])
        # gidx sorted & unique
        @test issorted(b.gidx) && allunique(b.gidx)
        # support_rtol = 0: shifted pair occupies SHIFTED global indices, union grows
        b0 = assemble_lattice_batch([dg], insts, [(1, 1), (2, 2)]; support_rtol = 0.0)
        gx1 = sort(unique(g[1] for (s, g) in enumerate(b0.gidx) if b0.densities[s, 1] != 0))
        gx2 = sort(unique(g[1] for (s, g) in enumerate(b0.gidx) if b0.densities[s, 2] != 0))
        @test gx2 == gx1 .+ 2          # pair (2,2) is pair (1,1) translated by s1=(2,0,0)
        @test length(b0.gidx) > count(!iszero, b0.densities[:, 1])   # union grew
    end
end
