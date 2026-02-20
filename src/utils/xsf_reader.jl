struct XSFStructure
    primvec::Matrix{Float64}          # 3x3, rows are lattice vectors (Å or bohr depending on source)
    species::Vector{String}           # length nat
    positions::Matrix{Float64}        # nat x 3 Cartesian coordinates
end

struct XSFDatagrid3D
    nx::Int
    ny::Int
    nz::Int
    origin::SVector{3,Float64}
    A::SVector{3,Float64}
    B::SVector{3,Float64}
    C::SVector{3,Float64}
    values::Array{Float64,3}          # size (nx, ny, nz), i-fastest
end

S3(x::AbstractVector{<:Real}) = StaticArrays.SVector{3,Float64}(Float64(x[1]), Float64(x[2]), Float64(x[3]))

function _read_vec3(io)::Vector{Float64}
    while !eof(io)
        s = strip(readline(io))
        isempty(s) && continue
        return parse.(Float64, split(replace(s, "D" => "E")))
    end
    error("Unexpected EOF while reading vec3")
end

function read_xsf(path::AbstractString)
    primvec = nothing
    species = String[]
    positions = Matrix{Float64}(undef, 0, 3)

    datagrid = nothing

    open(path, "r") do io
        lines = readlines(io)
        nlines = length(lines)

        i = 1
        while i <= nlines
            s = strip(lines[i])

            if s == "PRIMVEC"
                v1 = parse.(Float64, split(strip(lines[i+1])))
                v2 = parse.(Float64, split(strip(lines[i+2])))
                v3 = parse.(Float64, split(strip(lines[i+3])))
                primvec = vcat(v1', v2', v3')  # 3x3 rows = vectors
                i += 4
                continue
            end

            if s == "PRIMCOORD"
                nat = parse(Int, split(strip(lines[i+1]))[1])
                species = Vector{String}(undef, nat)
                positions = Matrix{Float64}(undef, nat, 3)
                for a = 1:nat
                    parts = split(strip(lines[i+1+a]))
                    species[a] = parts[1]
                    positions[a, :] = parse.(Float64, parts[2:4])
                end
                i += 2 + nat
                continue
            end

            if startswith(s, "BEGIN_DATAGRID_3D")
                # header
                dims = parse.(Int, split(strip(lines[i+1])))
                nx, ny, nz = dims
                origin = S3(parse.(Float64, split(strip(lines[i+2]))))
                A = S3(parse.(Float64, split(strip(lines[i+3]))))
                B = S3(parse.(Float64, split(strip(lines[i+4]))))
                C = S3(parse.(Float64, split(strip(lines[i+5]))))

                n = nx * ny * nz
                data = Vector{Float64}(undef, n)
                p = 1
                j = i + 6
                while j <= nlines
                    t = strip(lines[j])
                    if startswith(t, "END_DATAGRID_3D")
                        break
                    end
                    if isempty(t)
                        j += 1
                        continue
                    end
                    parts = split(replace(t, "D" => "E"))   # handle Fortran exponent
                    vals = parse.(Float64, parts)
                    data[p:p+length(vals)-1] .= vals
                    p += length(vals)
                    j += 1
                    p > n && break
                end
                p == n+1 || error("DATAGRID expected $n values, got $(p-1)")

                values = reshape(data, (nx, ny, nz)) # i (x) fastest, then j (y), then k (z)
                datagrid = (nx=nx, ny=ny, nz=nz, origin=origin, A=A, B=B, C=C, values=values)

                i = j + 1
                continue
            end

            i += 1
        end
    end

    primvec === nothing && error("No PRIMVEC found")
    isempty(species) && error("No PRIMCOORD found")
    datagrid === nothing && error("No DATAGRID_3D found")

    structure = XSFStructure(primvec, species, positions)
    return structure, datagrid
end

function true_cell_vectors(datagrid)
    nx, ny, nz = datagrid.nx, datagrid.ny, datagrid.nz
    A = collect(Tuple(datagrid.A))
    B = collect(Tuple(datagrid.B))
    C = collect(Tuple(datagrid.C))
    At = A .* (nx / (nx - 1))
    Bt = B .* (ny / (ny - 1))
    Ct = C .* (nz / (nz - 1))
    return At, Bt, Ct
end

function grid_point(datagrid, i::Int, j::Int, k::Int)
    # Wannier90 half-open convention: i,j,k are 1-based -> (i-1)/N
    nx, ny, nz = datagrid.nx, datagrid.ny, datagrid.nz
    u = (i-1) / nx
    v = (j-1) / ny
    w = (k-1) / nz
    o = Tuple(datagrid.origin)
    At, Bt, Ct = true_cell_vectors(datagrid)
    return (
        o[1] + u*At[1] + v*Bt[1] + w*Ct[1],
        o[2] + u*At[2] + v*Bt[2] + w*Ct[2],
        o[3] + u*At[3] + v*Bt[3] + w*Ct[3],
    )
end

function VolumeSource(datagrid; shift::NTuple{3,<:Real}=(0.0, 0.0, 0.0))
    nx, ny, nz = datagrid.nx, datagrid.ny, datagrid.nz
    T = Float64
    axes = (
        collect(T((i - 1) / nx) for i in 1:nx),
        collect(T((j - 1) / ny) for j in 1:ny),
        collect(T((k - 1) / nz) for k in 1:nz),
    )
    At, Bt, Ct = true_cell_vectors(datagrid)
    basis = (
        (At[1], At[2], At[3]),
        (Bt[1], Bt[2], Bt[3]),
        (Ct[1], Ct[2], Ct[3]),
    )
    jac = abs(det(hcat(collect(At), collect(Bt), collect(Ct))))
    weights = fill(jac / (nx * ny * nz), nx, ny, nz)
    density = copy(datagrid.values)
    shift_f = (Float64(shift[1]), Float64(shift[2]), Float64(shift[3]))
    origin = (datagrid.origin[1], datagrid.origin[2], datagrid.origin[3]) .+ shift_f
    return VolumeSource(axes, weights, density, origin, basis)
end

function _datagrid_affine(datagrid)
    o = datagrid.origin
    At, Bt, Ct = true_cell_vectors(datagrid)
    M = hcat(collect(At), collect(Bt), collect(Ct))
    Minv = inv(M)
    u_max = (datagrid.nx - 1) / datagrid.nx
    v_max = (datagrid.ny - 1) / datagrid.ny
    w_max = (datagrid.nz - 1) / datagrid.nz
    return o, M, Minv, u_max, v_max, w_max
end

function _is_axis_aligned_datagrid(datagrid)
    At, Bt, Ct = true_cell_vectors(datagrid)
    tol = 1e-12
    return abs(At[2]) <= tol && abs(At[3]) <= tol &&
        abs(Bt[1]) <= tol && abs(Bt[3]) <= tol &&
        abs(Ct[1]) <= tol && abs(Ct[2]) <= tol
end

function _is_inside_datagrid(Minv::AbstractMatrix, p::NTuple{3, <:Real}, datagrid; tol = 1e-10)
    o = datagrid.origin
    uvw = Minv * [p[1] - o[1], p[2] - o[2], p[3] - o[3]]
    u, v, w = uvw
    u_max = (datagrid.nx - 1) / datagrid.nx
    v_max = (datagrid.ny - 1) / datagrid.ny
    w_max = (datagrid.nz - 1) / datagrid.nz
    return -tol <= u <= u_max + tol &&
        -tol <= v <= v_max + tol &&
        -tol <= w <= w_max + tol
end

function _datagrid_trilinear_value(datagrid, Minv::AbstractMatrix, p::NTuple{3, <:Real}; tol = 1e-10)
    nx, ny, nz = datagrid.nx, datagrid.ny, datagrid.nz
    o = datagrid.origin
    duvw = Minv * [p[1] - o[1], p[2] - o[2], p[3] - o[3]]
    u, v, w = duvw
    u_min, v_min, w_min = 0.0, 0.0, 0.0
    u_max = (nx - 1) / nx
    v_max = (ny - 1) / ny
    w_max = (nz - 1) / nz
    if u < u_min - tol || u > u_max + tol || v < v_min - tol || v > v_max + tol || w < w_min - tol || w > w_max + tol
        return NaN
    end

    uf = clamp(u, u_min, u_max)
    vf = clamp(v, v_min, v_max)
    wf = clamp(w, w_min, w_max)
    sx = uf * nx + 1
    sy = vf * ny + 1
    sz = wf * nz + 1

    ix0 = clamp(floor(Int, sx), 1, nx - 1)
    iy0 = clamp(floor(Int, sy), 1, ny - 1)
    iz0 = clamp(floor(Int, sz), 1, nz - 1)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1

    tx = clamp(sx - ix0, 0.0, 1.0)
    ty = clamp(sy - iy0, 0.0, 1.0)
    tz = clamp(sz - iz0, 0.0, 1.0)

    vals = datagrid.values
    v000 = vals[ix0, iy0, iz0]
    v100 = vals[ix1, iy0, iz0]
    v010 = vals[ix0, iy1, iz0]
    v110 = vals[ix1, iy1, iz0]
    v001 = vals[ix0, iy0, iz1]
    v101 = vals[ix1, iy0, iz1]
    v011 = vals[ix0, iy1, iz1]
    v111 = vals[ix1, iy1, iz1]

    c00 = v000 * (1 - tx) + v100 * tx
    c10 = v010 * (1 - tx) + v110 * tx
    c01 = v001 * (1 - tx) + v101 * tx
    c11 = v011 * (1 - tx) + v111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    return c0 * (1 - tz) + c1 * tz
end

function datagrid_zslice(
    datagrid;
    iz::Union{Nothing, Int} = nothing,
    z::Union{Nothing, Real} = nothing,
    nx_sample::Union{Nothing, Int} = nothing,
    ny_sample::Union{Nothing, Int} = nothing,
    interpolation::Symbol = :trilinear,
    log_density::Bool = false,
    log_floor::Real = 1e-16,
)
    isnothing(iz) || isnothing(z) || throw(ArgumentError("Specify only one of iz or z"))
    interpolation in (:trilinear, :index) || throw(ArgumentError("interpolation must be :trilinear or :index"))
    nx, ny, nz = datagrid.nx, datagrid.ny, datagrid.nz
    k = if !isnothing(iz)
        1 <= iz <= nz || throw(ArgumentError("iz must be between 1 and $nz"))
        iz
    elseif !isnothing(z)
        zf = float(z)
        zs = [grid_point(datagrid, 1, 1, kk)[3] for kk in 1:nz]
        argmin(abs.(zs .- zf))
    else
        cld(nz, 2)
    end

    if isnothing(z)
        xs = [grid_point(datagrid, i, 1, k)[1] for i in 1:nx]
        ys = [grid_point(datagrid, 1, j, k)[2] for j in 1:ny]
        vals = datagrid.values[:, :, k]
        vals_f = log_density ? log10.(abs.(float.(vals)) .+ float(log_floor)) : float.(vals)
        z_level = grid_point(datagrid, 1, 1, k)[3]
        return (x = xs, y = ys, values = vals_f, iz = k, z = z_level)
    end

    # For axis-aligned grids, z=const matches a single k-layer exactly.
    if interpolation == :index || _is_axis_aligned_datagrid(datagrid)
        xs = [grid_point(datagrid, i, 1, k)[1] for i in 1:nx]
        ys = [grid_point(datagrid, 1, j, k)[2] for j in 1:ny]
        vals = datagrid.values[:, :, k]
        vals_f = log_density ? log10.(abs.(float.(vals)) .+ float(log_floor)) : float.(vals)
        z_level = isnothing(z) ? grid_point(datagrid, 1, 1, k)[3] : float(z)
        return (x = xs, y = ys, values = vals_f, iz = k, z = z_level)
    end

    nxs = isnothing(nx_sample) ? nx : nx_sample
    nys = isnothing(ny_sample) ? ny : ny_sample
    nxs >= 2 || throw(ArgumentError("nx_sample must be >= 2"))
    nys >= 2 || throw(ArgumentError("ny_sample must be >= 2"))

    o, M, Minv, u_max, v_max, w_max = _datagrid_affine(datagrid)
    corners = NTuple{3, Float64}[]
    for u in (0.0, u_max), v in (0.0, v_max), w in (0.0, w_max)
        p = M * [u, v, w]
        push!(corners, (o[1] + p[1], o[2] + p[2], o[3] + p[3]))
    end
    xmin = minimum(c[1] for c in corners)
    xmax = maximum(c[1] for c in corners)
    ymin = minimum(c[2] for c in corners)
    ymax = maximum(c[2] for c in corners)

    xs = collect(LinRange(xmin, xmax, nxs))
    ys = collect(LinRange(ymin, ymax, nys))
    z0 = float(z)
    vals = Matrix{Float64}(undef, nxs, nys)
    for i in 1:nxs, j in 1:nys
        p = (xs[i], ys[j], z0)
        if _is_inside_datagrid(Minv, p, datagrid)
            vals[i, j] = _datagrid_trilinear_value(datagrid, Minv, p)
        else
            vals[i, j] = NaN
        end
    end
    vals_f = log_density ? log10.(abs.(vals) .+ float(log_floor)) : vals
    return (x = xs, y = ys, values = vals_f, iz = k, z = z0)
end
