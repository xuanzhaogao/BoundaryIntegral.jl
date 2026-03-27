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

    # Full overlap on y=0.5 plane: box1 at origin, box2 at (0,1,0), both unit cubes
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

@testset "_detect_shared_faces_3d" begin
    # Two unit cubes touching at x=0.5
    boxes = [
        (center = (0.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
        (center = (1.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
    ]
    shared = BI._detect_shared_faces_3d(boxes)
    @test length(shared) == 1  # one shared face

    # Each entry is (region, id_lo, id_hi, normal_from_hi_to_lo)
    region, id1, id2, normal = shared[1]
    @test id1 < id2
    a, b, c, d = region
    area = norm(b .- a) * norm(a .- d)
    @test area ≈ 1.0  # full 1x1 face

    # Three boxes in a line
    boxes3 = [
        (center = (0.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
        (center = (1.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
        (center = (2.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
    ]
    shared3 = BI._detect_shared_faces_3d(boxes3)
    @test length(shared3) == 2  # box1-box2, box2-box3

    # Non-touching boxes
    boxes_far = [
        (center = (0.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
        (center = (5.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
    ]
    shared_far = BI._detect_shared_faces_3d(boxes_far)
    @test length(shared_far) == 0
end

@testset "_subtract_rects_from_face_3d" begin
    # Face on x=0.5 plane, 1x1 face from y=-0.5..0.5, z=-0.5..0.5
    face_a = (0.5, -0.5, -0.5)
    face_b = (0.5,  0.5, -0.5)
    face_c = (0.5,  0.5,  0.5)
    face_d = (0.5, -0.5,  0.5)
    face_n = (1.0, 0.0, 0.0)

    # No shared regions: should return the whole face with all edges/corners true
    remaining = BI._subtract_rects_from_face_3d(face_a, face_b, face_c, face_d, face_n, NTuple{4, NTuple{3, Float64}}[])
    @test length(remaining) == 1
    ra, rb, rc, rd, ie, ic = remaining[1]
    @test norm(rb .- ra) * norm(ra .- rd) ≈ 1.0
    @test ie == (true, true, true, true)
    @test ic == (true, true, true, true)

    # Full face shared: should return empty
    shared_full = [(face_a, face_b, face_c, face_d)]
    remaining_full = BI._subtract_rects_from_face_3d(face_a, face_b, face_c, face_d, face_n, shared_full)
    @test isempty(remaining_full)

    # Half face shared: bottom half z=-0.5..0.0
    shared_half = [((0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.0), (0.5, -0.5, 0.0))]
    remaining_half = BI._subtract_rects_from_face_3d(face_a, face_b, face_c, face_d, face_n, shared_half)
    total_area_remaining = sum(norm(r[2] .- r[1]) * norm(r[1] .- r[4]) for r in remaining_half)
    @test total_area_remaining ≈ 0.5
    # The remaining piece: edge_ab at z=0 is on a shared region boundary → physical
    # edge_cd at z=0.5 is on the original face boundary → physical
    _, _, _, _, ie_half, ic_half = remaining_half[1]
    @test ie_half[1] == true   # ab edge: at z=0, on shared region boundary
    @test ie_half[3] == true   # cd edge: at z=0.5, on original face boundary

    # Center hole: 1x1 shared in center of 2x2 face (on the x=1 plane)
    face2_a = (1.0, -1.0, -1.0)
    face2_b = (1.0,  1.0, -1.0)
    face2_c = (1.0,  1.0,  1.0)
    face2_d = (1.0, -1.0,  1.0)
    face2_n = (1.0, 0.0, 0.0)
    shared_center = [((1.0, -0.5, -0.5), (1.0, 0.5, -0.5), (1.0, 0.5, 0.5), (1.0, -0.5, 0.5))]
    remaining_center = BI._subtract_rects_from_face_3d(face2_a, face2_b, face2_c, face2_d, face2_n, shared_center)
    @test length(remaining_center) == 8
    total_area_center = sum(norm(r[2] .- r[1]) * norm(r[1] .- r[4]) for r in remaining_center)
    @test total_area_center ≈ 3.0  # 4 - 1

    # All edges are physical: they lie on either the face boundary or the shared region boundary
    # Corner pieces (4): all 4 edges are physical (2 face boundary + 2 shared boundary)
    # Side pieces (4): all 4 edges are physical (1 face boundary + 2 shared boundary + 1 shared boundary)
    for r in remaining_center
        @test all(r[5])  # all edges physical
        @test all(r[6])  # all corners physical
    end
end

@testset "multi_dielectric_box3d" begin
    @testset "single box equivalence" begin
        Lx, Ly, Lz = 1.0, 1.0, 1.0
        n_quad = 4
        l_ec = 0.3
        eps_in = 2.0
        eps_out = 1.0

        single = BI.single_dielectric_box3d(Lx, Ly, Lz, n_quad, l_ec, eps_in, eps_out, Float64)

        boxes = [(center = (0.0, 0.0, 0.0), Lx = Lx, Ly = Ly, Lz = Lz)]
        epses = [eps_in]
        multi = BI.multi_dielectric_box3d(n_quad, l_ec, boxes, epses, eps_out)

        @test length(multi.panels) == length(single.panels)
        @test BI.num_points(multi) == BI.num_points(single)
        @test all(multi.eps_in .== eps_in)
        @test all(multi.eps_out .== eps_out)
    end

    @testset "two touching boxes" begin
        boxes = [
            (center = (0.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
            (center = (1.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
        ]
        epses = [2.0, 4.0]
        eps_out = 1.0
        interface = BI.multi_dielectric_box3d(4, 0.3, boxes, epses, eps_out)

        @test length(interface.panels) > 0
        @test BI.num_points(interface) > 0

        # Check that eps values are correct
        unique_eps_pairs = Set{Tuple{Float64, Float64}}()
        for i in 1:length(interface.panels)
            push!(unique_eps_pairs, (interface.eps_in[i], interface.eps_out[i]))
        end
        # Should have: (2.0, 1.0) for box1-vacuum, (4.0, 1.0) for box2-vacuum,
        # and (4.0, 2.0) for the shared face (normal from box2 to box1)
        @test (2.0, 1.0) in unique_eps_pairs  # box1 external
        @test (4.0, 1.0) in unique_eps_pairs  # box2 external
        @test (4.0, 2.0) in unique_eps_pairs  # shared face

        # Total surface area check:
        # Two unit cubes touching: total exposed area = 2*6 - 2*1 = 10 external + 1 shared = 11 face-areas
        total_weight = sum(BI.all_weights(interface))
        @test total_weight ≈ 11.0 atol = 0.01
    end

    @testset "three boxes L-shape" begin
        boxes = [
            (center = (0.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
            (center = (1.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
            (center = (0.0, 1.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
        ]
        epses = [2.0, 3.0, 4.0]
        interface = BI.multi_dielectric_box3d(4, 0.3, boxes, epses)

        @test length(interface.panels) > 0

        # 3 cubes, 2 shared faces, total weight = 3*6 - 2*2 + 2 = 16
        # Actually: 3 cubes have 18 face-areas total. 2 shared faces remove 2 external face-areas
        # and add 2 shared face-areas (which are the same area).
        # Total area integrated = 18 - 2*2 + 2 = 16... let me recalculate:
        # Each shared face replaces 2 external faces (one from each box) with 1 shared face.
        # So total face-areas = 18 - 2 = 16 (each shared face removes 1 net face since
        # 2 external become 1 shared). Wait:
        # - 3 boxes * 6 faces = 18 face-units
        # - 2 shared faces: each shared face means 2 external faces are replaced by 1 shared face
        # But the shared face is still a face with panels, so total face count = 18 - 2 = 16
        # Total area = 16 * 1.0 = 16.0
        total_weight = sum(BI.all_weights(interface))
        @test total_weight ≈ 16.0 atol = 0.01
    end

    @testset "different-sized touching boxes" begin
        boxes = [
            (center = (0.0, 0.0, 0.0), Lx = 2.0, Ly = 2.0, Lz = 2.0),
            (center = (1.5, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
        ]
        epses = [2.0, 4.0]
        eps_out = 1.0
        interface = BI.multi_dielectric_box3d(4, 0.3, boxes, epses, eps_out)

        @test length(interface.panels) > 0

        # Big box: 6 faces of 2x2=4, minus 1x1 shared = 23 external
        # Small box: 6 faces of 1x1=1, minus 1x1 shared = 5 external
        # Shared: 1x1 = 1
        # Total = 23 + 5 + 1 = 29
        total_weight = sum(BI.all_weights(interface))
        @test total_weight ≈ 29.0 atol = 0.1

        # Verify eps pairs
        eps_pairs = Set{Tuple{Float64, Float64}}()
        for i in 1:length(interface.panels)
            push!(eps_pairs, (interface.eps_in[i], interface.eps_out[i]))
        end
        @test (2.0, 1.0) in eps_pairs
        @test (4.0, 1.0) in eps_pairs
        @test (4.0, 2.0) in eps_pairs
    end
end

@testset "multi_dielectric_box3d solver integration" begin
    boxes = [
        (center = (0.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
        (center = (1.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
    ]
    epses = [2.0, 4.0]
    eps_out = 1.0
    interface = BI.multi_dielectric_box3d(4, 0.3, boxes, epses, eps_out)

    # Point source outside both boxes
    ps = BI.PointSource((3.0, 0.0, 0.0), 1.0)
    eps_src = eps_out

    # Build LHS and RHS
    lhs = BI.lhs_dielectric_box3d(interface)
    rhs = BI.rhs_dielectric_box3d(interface, ps, eps_src)

    @test size(lhs, 1) == BI.num_points(interface)
    @test size(lhs, 2) == BI.num_points(interface)
    @test length(rhs) == BI.num_points(interface)

    # Solve
    sigma = lhs \ rhs
    @test length(sigma) == BI.num_points(interface)
    @test all(isfinite.(sigma))
end

@testset "multi_dielectric_box3d_rhs_adaptive" begin
    @testset "PointSource with different-sized boxes" begin
        boxes = [
            (center = (0.0, 0.0, 0.0), Lx = 2.0, Ly = 2.0, Lz = 2.0),
            (center = (1.5, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
        ]
        epses = [2.0, 4.0]
        eps_out = 1.0
        ps = BI.PointSource((0.1, 0.1, 0.1), 1.0)
        eps_src = epses[1]

        interface = BI.multi_dielectric_box3d_rhs_adaptive(
            4, 0.3, boxes, epses, ps, eps_src, 1e-4, eps_out;
            max_depth = 4,
        )

        @test length(interface.panels) > 0
        total_weight = sum(BI.all_weights(interface))
        @test total_weight ≈ 29.0 atol = 0.1

        lhs = BI.lhs_dielectric_box3d(interface)
        rhs = BI.rhs_dielectric_box3d(interface, ps, eps_src)
        sigma = lhs \ rhs
        @test all(isfinite.(sigma))
    end

    @testset "rhs::Function overload" begin
        boxes = [
            (center = (0.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
            (center = (1.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0),
        ]
        epses = [2.0, 4.0]
        ps = BI.PointSource((0.1, 0.1, 0.1), 1.0)
        eps_src = 2.0
        rhs_func(p, n) = -ps.charge * BI.laplace3d_grad(ps.point, p, n) / eps_src

        interface = BI.multi_dielectric_box3d_rhs_adaptive(
            4, 0.3, boxes, epses, rhs_func, 1e-4;
            max_depth = 4,
        )

        @test length(interface.panels) > 0
        total_weight = sum(BI.all_weights(interface))
        @test total_weight ≈ 11.0 atol = 0.1
    end
end
