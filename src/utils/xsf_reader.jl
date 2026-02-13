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

function VolumeSource(datagrid; shift::NTuple{3,Float64}=(0,0,0))
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
    origin = (datagrid.origin[1], datagrid.origin[2], datagrid.origin[3]) .+ shift
    return VolumeSource(axes, weights, density, origin, basis)
end
