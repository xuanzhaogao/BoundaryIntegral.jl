using BoundaryIntegral
import BoundaryIntegral as BI
using LinearAlgebra
using Test

@testset "multi_box3d helpers" begin
    @testset "_box3d_faces_at_center" begin
        center = (1.0, 2.0, 3.0)
        Lx, Ly, Lz = 2.0, 4.0, 6.0
        faces = BI._box3d_faces_at_center(center, Lx, Ly, Lz)
        @test length(faces) == 6

        # Each face is (a, b, c, d, normal) where a,b,c,d are NTuple{3,Float64}
        # Check that face centers are offset by center
        for (a, b, c, d, normal) in faces
            face_center = (a .+ b .+ c .+ d) ./ 4
            # face center should differ from box center only along the normal axis
            diff = face_center .- center
            @test abs(dot(diff, normal)) > 0  # offset along normal
            # tangential components should be zero
            tangential = diff .- normal .* dot(diff, normal)
            @test norm(tangential) < 1e-14
        end

        # Check normals are unit vectors
        for (a, b, c, d, normal) in faces
            @test norm(normal) ≈ 1.0
        end

        # Check face areas: two faces of each size
        areas = Float64[]
        for (a, b, c, d, normal) in faces
            Lab = norm(b .- a)
            Lda = norm(a .- d)
            push!(areas, Lab * Lda)
        end
        sort!(areas)
        @test areas[1] ≈ areas[2] ≈ Lx * Ly  # z-faces
        @test areas[3] ≈ areas[4] ≈ Lx * Lz  # y-faces
        @test areas[5] ≈ areas[6] ≈ Ly * Lz  # x-faces
    end
end
