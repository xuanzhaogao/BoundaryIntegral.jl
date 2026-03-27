function _box3d_geometry(Lx::T, Ly::T, Lz::T) where T
    t1 = one(T)
    t0 = zero(T)

    vertices = NTuple{3, T}[
        ( Lx / 2,  Ly / 2,  Lz / 2),  # 1
        (-Lx / 2,  Ly / 2,  Lz / 2),  # 2
        (-Lx / 2, -Ly / 2,  Lz / 2),  # 3
        ( Lx / 2, -Ly / 2,  Lz / 2),  # 4
        ( Lx / 2,  Ly / 2, -Lz / 2),  # 5
        (-Lx / 2,  Ly / 2, -Lz / 2),  # 6
        (-Lx / 2, -Ly / 2, -Lz / 2),  # 7
        ( Lx / 2, -Ly / 2, -Lz / 2),  # 8
    ]

    faces = NTuple{4, Int}[
        (1, 2, 3, 4),  # z = +Lz/2
        (5, 8, 7, 6),  # z = -Lz/2
        (8, 5, 1, 4),  # x = +Lx/2
        (7, 3, 2, 6),  # x = -Lx/2
        (6, 2, 1, 5),  # y = +Ly/2
        (7, 8, 4, 3),  # y = -Ly/2
    ]

    normals = NTuple{3, T}[
        ( t0,  t0,  t1),
        ( t0,  t0, -t1),
        ( t1,  t0,  t0),
        (-t1,  t0,  t0),
        ( t0,  t1,  t0),
        ( t0, -t1,  t0),
    ]

    return vertices, faces, normals
end

function _box3d_face_quads(Lx::T, Ly::T, Lz::T) where T
    vertices, faces, normals = _box3d_geometry(Lx, Ly, Lz)
    quads = NTuple{4, NTuple{3, T}}[]
    face_normals = NTuple{3, T}[]
    for i in 1:6
        face = faces[i]
        push!(quads, (vertices[face[1]], vertices[face[2]], vertices[face[3]], vertices[face[4]]))
        push!(face_normals, normals[i])
    end
    return quads, face_normals
end

function single_dielectric_box3d(Lx::T, Ly::T, Lz::T, n_quad::Int, l_ec::T, eps_in::T, eps_out::T, ::Type{T} = Float64; alpha::T = sqrt(T(2))) where T
    ns, ws = gausslegendre(n_quad)
    quads, normals = _box3d_face_quads(Lx, Ly, Lz)

    panels = Vector{FlatPanel{T, 3}}()
    for i in eachindex(quads)
        a, b, c, d = quads[i]
        normal = normals[i]
        append!(panels, rect_panel3d_adaptive_panels(
            a, b, c, d,
            ns,
            ws,
            normal,
            (true, true, true, true),
            (true, true, true, true),
            alpha,
            l_ec,
        ))
    end

    return DielectricInterface(panels, fill(eps_in, length(panels)), fill(eps_out, length(panels)))
end
