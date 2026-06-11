using BoundaryIntegral
using Test

@testset "toml_input" begin
    fixdir = joinpath(@__DIR__, "..", "fixtures")
    c = load_campaign(joinpath(fixdir, "system_small.toml"))

    @test c.name == "system_small"
    @test length(c.templates) == 2
    @test endswith(c.templates[1], "orb_a.xsf")
    @test length(c.orbitals) == 2
    @test c.orbitals[1].type == 1 && c.orbitals[1].pos == (1.0, 1.0, 1.0)
    @test c.orbitals[2].type == 2 && c.orbitals[2].pos == (5.0, 1.0, 1.0)
    @test c.neighbor_cutoff == 3.0
    @test c.pair_overrides === nothing
    @test length(c.boxes) == 1 && c.boxes[1].Lx == 4.0 && c.epses == [11.7]
    @test c.eps_out == 1.0
    @test c.solve["n_quad"] == 4 && c.solve["l_ec"] == 2.0
    @test c.n_centers_per_batch == 1
    @test c.far_pad_steps == 2.0

    # template paths are resolved relative to the toml's directory; toml_path recorded
    @test isabspath(c.templates[1])
    @test isabspath(c.toml_path) && endswith(c.toml_path, "system_small.toml")

    # path helpers
    @test endswith(batch_path(c, 7), "batches/batch_0007.jls")
    @test endswith(v_path(c, 12), "V/V_0012.jls")
    @test endswith(manifest_path(c), "manifest.tsv")
    @test BoundaryIntegral.campaign_l_ec(c) ≈ 2.0     # l_ec given directly
end

@testset "pair_overrides parse" begin
    fixdir = joinpath(@__DIR__, "..", "fixtures")
    c = load_campaign(joinpath(fixdir, "system_overrides.toml"))
    @test c.pair_overrides == [(1, 1), (1, 2)]
end
