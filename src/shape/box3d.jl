# internal panel type for 3d rectangle surface panel generation
struct TempPanel3D{T}
    a::NTuple{3, T}
    b::NTuple{3, T}
    c::NTuple{3, T}
    d::NTuple{3, T}

    is_a_corner::Bool
    is_b_corner::Bool
    is_c_corner::Bool
    is_d_corner::Bool

    is_ab_edge::Bool
    is_bc_edge::Bool
    is_cd_edge::Bool
    is_da_edge::Bool

    normal::NTuple{3, T}
end

# mesh a rectangle surface panel with tensor product Gauss-Legendre quadrature points
function rect_panel3d_discretize(
    a::NTuple{3, T},
    b::NTuple{3, T},
    c::NTuple{3, T},
    d::NTuple{3, T},
    ns::Vector{T},
    ws::Vector{T},
    normal::NTuple{3, T};
    is_edge::Bool = false,
) where T

    # check edge lengths
    Lab = norm(b .- a)
    Lbc = norm(c .- b)
    Lcd = norm(d .- c)
    Lda = norm(a .- d)
    @assert (Lab ≈ Lcd) && (Lbc ≈ Lda) "Edges of the square are not equal"

    # check perpendicularity
    @assert (abs(dot(normal, b .- a)) < 1e-10) && (abs(dot(normal, c .- b)) < 1e-10) && (abs(dot(normal, d .- c)) < 1e-10) && (abs(dot(normal, a .- d)) < 1e-10) "Normal is not perpendicular to the edges"
    @assert (abs(dot(b .- a, c .- b)) < 1e-10) && (abs(dot(c .- b, d .- c)) < 1e-10) && (abs(dot(d .- c, a .- d)) < 1e-10) && (abs(dot(a .- d, b .- a)) < 1e-10) "Edges are not perpendicular"

    cc = (a .+ b .+ c .+ d) ./ 4

    points = Vector{NTuple{3, T}}()
    for i in 1:length(ns)
        for j in 1:length(ns)
            p = cc .+ (b .- a) .* (ns[i] / 2) .+ (d .- a) .* (ns[j] / 2)
            push!(points, p)
        end
    end

    weights = Vector{T}()
    for i in 1:length(ns)
        for j in 1:length(ns)
            push!(weights, ws[i] * ws[j] * Lab * Lbc / 4)
        end
    end

    corners = [a, b, c, d]

    return FlatPanel(normal, corners, is_edge, length(ns), ns, ws, points, weights)
end

function divide_temp_panel3d(tpl::TempPanel3D{T}, n_divide_x::Int, n_divide_y::Int) where T
    # @assert n_divide_x >= 2 "n_divide_x must be greater than or equal to 2"
    # @assert n_divide_y >= 2 "n_divide_y must be greater than or equal to 2"

    panels = Vector{TempPanel3D{T}}(undef, n_divide_x * n_divide_y) # the panels are arranged in a row-major order

    for i in 1:n_divide_x
        for j in 1:n_divide_y
            u0 = (i - 1) / n_divide_x
            u1 = i / n_divide_x
            v0 = (j - 1) / n_divide_y
            v1 = j / n_divide_y

            a_ij = tpl.a .+ (tpl.b .- tpl.a) .* u0 .+ (tpl.d .- tpl.a) .* v0
            b_ij = tpl.a .+ (tpl.b .- tpl.a) .* u1 .+ (tpl.d .- tpl.a) .* v0
            c_ij = tpl.a .+ (tpl.b .- tpl.a) .* u1 .+ (tpl.d .- tpl.a) .* v1
            d_ij = tpl.a .+ (tpl.b .- tpl.a) .* u0 .+ (tpl.d .- tpl.a) .* v1

            is_a_corner = (i == 1 && j == 1) ? tpl.is_a_corner : false
            is_b_corner = (i == n_divide_x && j == 1) ? tpl.is_b_corner : false
            is_c_corner = (i == n_divide_x && j == n_divide_y) ? tpl.is_c_corner : false
            is_d_corner = (i == 1 && j == n_divide_y) ? tpl.is_d_corner : false

            is_ab_edge = (j == 1) ? tpl.is_ab_edge : false
            is_bc_edge = (i == n_divide_x) ? tpl.is_bc_edge : false
            is_cd_edge = (j == n_divide_y) ? tpl.is_cd_edge : false
            is_da_edge = (i == 1) ? tpl.is_da_edge : false

            panels[(i - 1) * n_divide_y + j] = TempPanel3D(a_ij, b_ij, c_ij, d_ij, is_a_corner, is_b_corner, is_c_corner, is_d_corner, is_ab_edge, is_bc_edge, is_cd_edge, is_da_edge, tpl.normal)
        end
    end
    return panels
end

# alpha controls the coarse grid aspect ratio; l_ec is the maximum length of an edge/corner panel
function rect_panel3d_adaptive_panels(a::NTuple{3, T}, b::NTuple{3, T}, c::NTuple{3, T}, d::NTuple{3, T}, ns::Vector{T}, ws::Vector{T}, normal::NTuple{3, T}, is_edge::NTuple{4, Bool}, is_corner::NTuple{4, Bool}, alpha::T, l_ec::T) where T
    Lab = norm(b .- a)
    Lda = norm(a .- d)
    n_divide_x, n_divide_y = best_grid_mn(Lab, Lda, alpha)

    rough = divide_temp_panel3d(
        TempPanel3D(a, b, c, d, is_corner[1], is_corner[2], is_corner[3], is_corner[4], is_edge[1], is_edge[2], is_edge[3], is_edge[4], normal),
        n_divide_x,
        n_divide_y,
    )

    fine = TempPanel3D{T}[]
    while !isempty(rough)
        tpl = popfirst!(rough)
        has_ec = tpl.is_a_corner || tpl.is_b_corner || tpl.is_c_corner || tpl.is_d_corner ||
            tpl.is_ab_edge || tpl.is_bc_edge || tpl.is_cd_edge || tpl.is_da_edge
        L_ab = norm(tpl.b .- tpl.a)
        L_da = norm(tpl.a .- tpl.d)
        if has_ec && max(L_ab, L_da) > l_ec
            append!(rough, divide_temp_panel3d(tpl, 2, 2))
        else
            push!(fine, tpl)
        end
    end

    panels = Vector{FlatPanel{T, 3}}()
    for tpl in fine
        is_edge = tpl.is_ab_edge || tpl.is_bc_edge || tpl.is_cd_edge || tpl.is_da_edge || tpl.is_a_corner || tpl.is_b_corner || tpl.is_c_corner || tpl.is_d_corner
        push!(panels, rect_panel3d_discretize(tpl.a, tpl.b, tpl.c, tpl.d, ns, ws, tpl.normal; is_edge = is_edge))
    end

    return panels
end
