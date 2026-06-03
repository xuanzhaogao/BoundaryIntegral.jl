const BoxGeom = NamedTuple{(:center, :Lx, :Ly, :Lz),
    Tuple{NTuple{3,Float64}, Float64, Float64, Float64}}

struct OrbitalEntry
    id::Int
    xsf_path::String
    center::NTuple{3,Float64}
end

struct SystemInput
    unit_scale::Float64
    boxes::Vector{BoxGeom}
    epses::Vector{Float64}
    eps_out::Float64
    orbitals::Dict{Int,OrbitalEntry}
    groups::Dict{Int,Vector{Int}}
end

_clean(line) = strip(first(split(line, '#')))

function read_system_input(path::AbstractString)
    base = dirname(abspath(path))
    raw = readlines(path)
    lines = [(_clean(l)) for l in raw]

    unit_scale = 1.0
    boxes = BoxGeom[]
    epses = Float64[]
    eps_out = 1.0
    orb_rows = Tuple{Int,String,Int}[]
    cutoff = Inf
    overrides = Dict{Int,Vector{Int}}()

    i = 1
    while i <= length(lines)
        s = lines[i]
        if isempty(s)
            i += 1; continue
        elseif startswith(s, "UNITS")
            u = lowercase(split(s)[2])
            unit_scale = u == "bohr" ? 1.0 :
                         u == "angstrom" ? 1.8897259886 :
                         error("Unknown UNITS: $u")
            i += 1
        elseif s == "BEGIN_DIELECTRICS"
            i += 1
            while i <= length(lines) && lines[i] != "END_DIELECTRICS"
                row = lines[i]
                if isempty(row)
                    i += 1; continue
                elseif startswith(row, "EPS_OUT")
                    eps_out = parse(Float64, split(row)[2])
                else
                    p = parse.(Float64, split(row))
                    length(p) == 7 || error("DIELECTRIC row needs 7 numbers, got: $row")
                    push!(boxes, (center = (p[1], p[2], p[3]),
                                  Lx = p[4], Ly = p[5], Lz = p[6]))
                    push!(epses, p[7])
                end
                i += 1
            end
            i += 1
        elseif s == "BEGIN_ORBITALS"
            i += 1
            while i <= length(lines) && lines[i] != "END_ORBITALS"
                row = lines[i]
                if isempty(row); i += 1; continue; end
                parts = split(row)
                id = parse(Int, parts[1])
                xsf = parts[2]
                atom_idx = length(parts) >= 3 ? parse(Int, parts[3]) : 0
                push!(orb_rows, (id, xsf, atom_idx))
                i += 1
            end
            i += 1
        elseif s == "BEGIN_GROUPING"
            i += 1
            while i <= length(lines) && lines[i] != "END_GROUPING"
                row = lines[i]
                if isempty(row); i += 1; continue; end
                if startswith(row, "CUTOFF")
                    cutoff = parse(Float64, split(row)[2])
                elseif occursin(':', row)
                    lhs, rhs = split(row, ':')
                    ci = parse(Int, strip(lhs))
                    overrides[ci] = parse.(Int, split(strip(rhs)))
                else
                    error("Unrecognized GROUPING line: $row")
                end
                i += 1
            end
            i += 1
        else
            error("Unrecognized top-level line: $s")
        end
    end

    orbitals = Dict{Int,OrbitalEntry}()
    for (id, xsf, atom_idx) in orb_rows
        full = isabspath(xsf) ? xsf : joinpath(base, xsf)
        structure, _ = read_xsf(full)
        nat = size(structure.positions, 1)
        a = atom_idx == 0 ? (nat == 1 ? 1 :
            error("orbital $id: PRIMCOORD has $nat atoms; specify atom_index")) :
            atom_idx
        (1 <= a <= nat) || error("orbital $id: atom_index $a out of range 1:$nat")
        c = ntuple(d -> structure.positions[a, d] * unit_scale, 3)
        orbitals[id] = OrbitalEntry(id, full, c)
    end

    boxes = BoxGeom[(center = b.center .* unit_scale,
                     Lx = b.Lx * unit_scale, Ly = b.Ly * unit_scale,
                     Lz = b.Lz * unit_scale) for b in boxes]
    cutoff_scaled = cutoff * unit_scale

    ids = sort(collect(keys(orbitals)))
    groups = Dict{Int,Vector{Int}}()
    for i_id in ids
        ci = orbitals[i_id].center
        neigh = Int[]
        for j_id in ids
            cj = orbitals[j_id].center
            d = sqrt(sum(abs2, ci .- cj))
            d <= cutoff_scaled && push!(neigh, j_id)
        end
        groups[i_id] = neigh
    end
    for (ci, lst) in overrides
        groups[ci] = copy(lst)
    end

    return SystemInput(unit_scale, boxes, epses, eps_out, orbitals, groups)
end
