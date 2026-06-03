const BoxGeom = NamedTuple{(:center, :Lx, :Ly, :Lz),
    Tuple{NTuple{3,Float64}, Float64, Float64, Float64}}

struct OrbitalEntry
    id::Int
    xsf_path::String
    center::NTuple{3,Float64}
    grid_shift::NTuple{3,Int}     # integer circshift (lattice-image translation; (0,0,0) = none)
end

"""
    SolveParams

Solver parameters parsed from the `.bie` `BEGIN_SOLVE` block (all optional; defaults shown).
`l_ec` takes precedence over `edge_refine_level`; if neither is given, `resolved_l_ec` uses
level 4 against the thinnest box (matching the monolayer reference convention).
"""
Base.@kwdef struct SolveParams
    n_quad::Int = 6
    edge_refine_level::Union{Int,Nothing} = nothing
    l_ec::Union{Float64,Nothing} = nothing
    rhs_tol::Float64 = 1e-3
    lhs_tol::Float64 = 1e-5
    gmres_rtol::Float64 = 1e-5
    support_rtol::Float64 = 1e-6
    volume_tol::Float64 = 1e-6
    max_order::Int = 8
    max_depth::Int = 128
end

struct SystemInput
    unit_scale::Float64
    boxes::Vector{BoxGeom}
    epses::Vector{Float64}
    eps_out::Float64
    orbitals::Dict{Int,OrbitalEntry}
    groups::Dict{Int,Vector{Int}}
    solve::SolveParams
end

function _build_solve_params(d::AbstractDict)
    pInt(k, dflt)   = haskey(d, k) ? parse(Int, d[k]) : dflt
    pFloat(k, dflt) = haskey(d, k) ? parse(Float64, d[k]) : dflt
    return SolveParams(
        n_quad            = pInt("N_QUAD", 6),
        edge_refine_level = haskey(d, "EDGE_REFINE_LEVEL") ? parse(Int, d["EDGE_REFINE_LEVEL"]) : nothing,
        l_ec              = haskey(d, "L_EC") ? parse(Float64, d["L_EC"]) : nothing,
        rhs_tol           = pFloat("RHS_TOL", 1e-3),
        lhs_tol           = pFloat("LHS_TOL", 1e-5),
        gmres_rtol        = pFloat("GMRES_RTOL", 1e-5),
        support_rtol      = pFloat("SUPPORT_RTOL", 1e-6),
        volume_tol        = pFloat("VOLUME_TOL", 1e-6),
        max_order         = pInt("MAX_ORDER", 8),
        max_depth         = pInt("MAX_DEPTH", 128),
    )
end

"""
    resolved_l_ec(si) -> Float64

Edge/corner refinement target size: `si.solve.l_ec` if set, else
`min(box.Lz) / 2^edge_refine_level * 1.01` (level defaults to 4).
"""
function resolved_l_ec(si::SystemInput)
    si.solve.l_ec !== nothing && return si.solve.l_ec
    level = si.solve.edge_refine_level === nothing ? 4 : si.solve.edge_refine_level
    minLz = minimum(b.Lz for b in si.boxes)
    return minLz / 2.0^level * 1.01
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
    orb_rows = Tuple{Int,String,Tuple{Symbol,Any}}[]
    cutoff = Inf
    overrides = Dict{Int,Vector{Int}}()
    solve_dict = Dict{String,String}()

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
            i <= length(lines) || error("unexpected EOF inside BEGIN_DIELECTRICS")
            i += 1  # skip END_DIELECTRICS
        elseif s == "BEGIN_ORBITALS"
            i += 1
            while i <= length(lines) && lines[i] != "END_ORBITALS"
                row = lines[i]
                if isempty(row); i += 1; continue; end
                parts = split(row)
                id = parse(Int, parts[1])
                xsf = parts[2]
                # spec :: (:none,) | (:center, (cx,cy,cz)) | (:lattice, (n1,n2,n3))
                spec = if length(parts) >= 3 && uppercase(parts[3]) == "LATTICE"
                    length(parts) >= 6 || error("ORBITAL ... LATTICE needs n1 n2 n3, got: $row")
                    (:lattice, (parse(Int, parts[4]), parse(Int, parts[5]), parse(Int, parts[6])))
                elseif length(parts) >= 5
                    (:center, (parse(Float64, parts[3]), parse(Float64, parts[4]), parse(Float64, parts[5])))
                elseif length(parts) == 2
                    (:none, nothing)
                else
                    error("ORBITAL row: `id xsf` | `id xsf cx cy cz` | `id xsf LATTICE n1 n2 n3`, got: $row")
                end
                push!(orb_rows, (id, xsf, spec))
                i += 1
            end
            i <= length(lines) || error("unexpected EOF inside BEGIN_ORBITALS")
            i += 1  # skip END_ORBITALS
        elseif s == "BEGIN_GROUPING"
            i += 1
            while i <= length(lines) && lines[i] != "END_GROUPING"
                row = lines[i]
                if isempty(row); i += 1; continue; end
                if startswith(row, "CUTOFF")
                    cutoff = parse(Float64, split(row)[2])
                elseif occursin(':', row)
                    lhs, rhs = split(row, ':'; limit = 2)
                    ci = parse(Int, strip(lhs))
                    overrides[ci] = parse.(Int, split(strip(rhs)))
                else
                    error("Unrecognized GROUPING line: $row")
                end
                i += 1
            end
            i <= length(lines) || error("unexpected EOF inside BEGIN_GROUPING")
            i += 1  # skip END_GROUPING
        elseif s == "BEGIN_SOLVE"
            i += 1
            while i <= length(lines) && lines[i] != "END_SOLVE"
                row = lines[i]
                if !isempty(row)
                    parts = split(row)
                    length(parts) >= 2 || error("SOLVE line needs `KEY value`, got: $row")
                    solve_dict[uppercase(parts[1])] = parts[2]
                end
                i += 1
            end
            i <= length(lines) || error("unexpected EOF inside BEGIN_SOLVE")
            i += 1  # skip END_SOLVE
        else
            error("Unrecognized top-level line: $s")
        end
    end

    orbitals = Dict{Int,OrbitalEntry}()
    for (id, xsf, spec) in orb_rows
        full = isabspath(xsf) ? xsf : joinpath(base, xsf)
        grid_shift = (0, 0, 0)
        c0 = if spec[1] == :center
            spec[2]
        else
            st, dg = read_xsf(full)
            if spec[1] == :lattice
                grid_shift = _lattice_grid_shift(dg, st.primvec, spec[2])
                dg = merge(dg, (; values = circshift(dg.values, grid_shift)))
            end
            density_centroid(dg)
        end
        c = ntuple(d -> c0[d] * unit_scale, 3)
        orbitals[id] = OrbitalEntry(id, full, c, grid_shift)
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
        haskey(orbitals, ci) || error("GROUPING override references unknown orbital id $ci")
        groups[ci] = copy(lst)
    end

    return SystemInput(unit_scale, boxes, epses, eps_out, orbitals, groups, _build_solve_params(solve_dict))
end
