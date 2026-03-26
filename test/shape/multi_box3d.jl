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

@testset "_rect_overlap_3d" begin
    # Two faces sharing a full face on x=0.5 plane
    # Box1 at origin, Box2 at (1,0,0), both unit cubes
    # Box1 has face at x=+0.5, Box2 has face at x=-0.5
    face1_a = (0.5, -0.5, -0.5)
    face1_b = (0.5,  0.5, -0.5)
    face1_c = (0.5,  0.5,  0.5)
    face1_d = (0.5, -0.5,  0.5)
    face1_n = (1.0, 0.0, 0.0)

    face2_a = (0.5, -0.5, -0.5)
    face2_b = (0.5,  0.5, -0.5)
    face2_c = (0.5,  0.5,  0.5)
    face2_d = (0.5, -0.5,  0.5)
    face2_n = (-1.0, 0.0, 0.0)

    has_overlap, region = BI._rect_overlap_3d(
        face1_a, face1_b, face1_c, face1_d, face1_n,
        face2_a, face2_b, face2_c, face2_d, face2_n
    )
    @test has_overlap
    # The overlap region should be the full face
    oa, ob, oc, od = region
    Lab = norm(ob .- oa)
    Lda = norm(oa .- od)
    @test Lab * Lda ≈ 1.0  # 1x1 face

    # Non-coplanar faces should not overlap
    face3_a = (0.0, -0.5, 0.5)
    face3_b = (0.0,  0.5, 0.5)
    face3_c = (1.0,  0.5, 0.5)
    face3_d = (1.0, -0.5, 0.5)
    face3_n = (0.0, 0.0, 1.0)

    has_overlap2, _ = BI._rect_overlap_3d(
        face1_a, face1_b, face1_c, face1_d, face1_n,
        face3_a, face3_b, face3_c, face3_d, face3_n
    )
    @test !has_overlap2

    # Partial overlap: box1 unit cube at origin, box2 unit cube at (0, 0.5, 0)
    # They share part of the y=+0.5 face of box1 and y=-0.5 face of box2 is at y=0.0
    # Actually box2's y=-0.5 face is at y=0.5-0.5=0.0, box1's y=+0.5 face is at y=0.5
    # These are NOT co-planar, so no overlap. Let me fix:
    # box1 at origin, box2 at (0, 1.0, 0) - they touch at y=0.5
    face_b1_yp_a = (0.5, 0.5, -0.5)  # box1 y=+0.5 face  (but this face has normal (0,1,0))
    face_b1_yp_b = (-0.5, 0.5, -0.5)
    face_b1_yp_c = (-0.5, 0.5, 0.5)
    face_b1_yp_d = (0.5, 0.5, 0.5)
    face_b1_yp_n = (0.0, 1.0, 0.0)

    face_b2_ym_a = (-0.5, 0.5, -0.5)  # box2 at (0,1,0) y=-0.5 face at y=0.5
    face_b2_ym_b = (0.5, 0.5, -0.5)
    face_b2_ym_c = (0.5, 0.5, 0.5)
    face_b2_ym_d = (-0.5, 0.5, 0.5)
    face_b2_ym_n = (0.0, -1.0, 0.0)

    has_overlap3, region3 = BI._rect_overlap_3d(
        face_b1_yp_a, face_b1_yp_b, face_b1_yp_c, face_b1_yp_d, face_b1_yp_n,
        face_b2_ym_a, face_b2_ym_b, face_b2_ym_c, face_b2_ym_d, face_b2_ym_n
    )
    @test has_overlap3
    oa3, ob3, oc3, od3 = region3
    @test norm(ob3 .- oa3) * norm(oa3 .- od3) ≈ 1.0
end
