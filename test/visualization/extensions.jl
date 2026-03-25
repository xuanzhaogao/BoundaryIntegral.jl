pkg_root = pkgdir(BoundaryIntegral)

@testset "Visualization extension wiring" begin
    project_toml = read(joinpath(pkg_root, "Project.toml"), String)
    @test occursin("\nMakie = ", project_toml)
    @test occursin("MakieExt = [\"Makie\"]", project_toml)

    extension_path = joinpath(pkg_root, "ext", "MakieExt.jl")

    @test isfile(extension_path)
    @test !isfile(joinpath(pkg_root, "ext", "GLMakieExt.jl"))

    if isfile(extension_path)
        extension_source = read(extension_path, String)
        @test occursin("module MakieExt", extension_source)
        @test occursin("using Makie", extension_source)
        @test !occursin("using GLMakie", extension_source)
    end
end

@testset "Visualization fallback warnings" begin
    @test_logs (:warn, r"Makie.*backend") BoundaryIntegral.viz_2d(nothing)
    @test_logs (:warn, r"Makie.*backend") BoundaryIntegral.viz_3d(nothing)
    @test_logs (:warn, r"Makie.*backend") BoundaryIntegral.viz_3d_surface(nothing)
    @test_logs (:warn, r"Makie.*backend") BoundaryIntegral.viz_3d_interface_solution(nothing)
    @test_logs (:warn, r"Makie.*backend") BoundaryIntegral.viz_3d_zslice(nothing)
end

@testset "Visualization README guidance" begin
    readme = read(joinpath(pkg_root, "README.md"), String)
    @test occursin("Optional Makie visualization extension.", readme)
    @test occursin("install Makie and a backend", readme)
    @test occursin("Pkg.add([\"Makie\", \"CairoMakie\"])", readme)
    @test !occursin("To enable GLMakie-based plotting", readme)
end
