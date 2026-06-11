# Unified .toml input for the four-index campaign (spec
# docs/.../2026-06-10-toml-campaign-unification-design.md). Replaces the .bie format.

using TOML

# BoxGeom is also defined in utils/system_input.jl (to be deleted in Task 5).
# We re-declare it here so that toml_input.jl does not depend on system_input.jl,
# and Task 5 can delete the duplicate without any other change.
# NOTE: if system_input.jl is still loaded first, Julia will see the same const value
# and not error (identical NamedTuple type alias).  To avoid a "cannot redefine" error
# we guard with isdefined.
if !@isdefined(BoxGeom)
    const BoxGeom = NamedTuple{(:center, :Lx, :Ly, :Lz),
        Tuple{NTuple{3,Float64}, Float64, Float64, Float64}}
end

"""
    OrbitalSpec(type, pos)

One orbital from the .toml `[[orbital]]` list: `type` indexes `CampaignInput.templates`
(1-based), `pos` is its Cartesian center in the templates' frame.
"""
struct OrbitalSpec
    type::Int
    pos::NTuple{3,Float64}
end

"""
    CampaignInput

Parsed `.toml`: the full system (orbitals, dielectrics, solve params) plus campaign
batching/eval settings. Successor to both `Campaign` and the deleted `SystemInput`.
Template grids are NOT loaded here (see `load_templates!`). Orbital id = 1-based index
in `orbitals`.
"""
struct CampaignInput
    name::String
    root::String
    templates::Vector{String}                 # type index → xsf path (absolute)
    orbitals::Vector{OrbitalSpec}
    neighbor_cutoff::Float64
    pair_overrides::Union{Nothing,Vector{Tuple{Int,Int}}}
    eps_out::Float64
    boxes::Vector{BoxGeom}
    epses::Vector{Float64}
    solve::Dict{String,Float64}
    n_centers_per_batch::Int
    far_pad_steps::Float64
    toml_path::String                         # absolute path of the source .toml (workers reload from it)
end

function load_campaign(toml_path::AbstractString)
    toml_path = abspath(toml_path)
    d = TOML.parsefile(toml_path)
    base = dirname(toml_path)
    haskey(d, "batching") || error("$toml_path: missing [batching] section")
    haskey(d, "orbital") || error("$toml_path: needs at least one [[orbital]] entry")

    templates = String[isabspath(t) ? t : joinpath(base, t) for t in d["templates"]]
    orbitals = OrbitalSpec[]
    for o in d["orbital"]
        push!(orbitals, OrbitalSpec(Int(o["type"]),
            (Float64(o["x"]), Float64(o["y"]), Float64(o["z"]))))
    end

    pr = get(d, "pairing", Dict())
    cutoff = Float64(get(pr, "neighbor_cutoff", Inf))
    overrides = haskey(pr, "pairs") ?
        Tuple{Int,Int}[(Int(p[1]), Int(p[2])) for p in pr["pairs"]] : nothing

    di = d["dielectrics"]
    boxes = BoxGeom[]; epses = Float64[]
    for row in di["boxes"]
        length(row) == 7 || error("dielectrics.boxes rows are [cx cy cz Lx Ly Lz eps]")
        push!(boxes, (center = (Float64(row[1]), Float64(row[2]), Float64(row[3])),
                      Lx = Float64(row[4]), Ly = Float64(row[5]), Lz = Float64(row[6])))
        push!(epses, Float64(row[7]))
    end

    solve = Dict{String,Float64}(k => Float64(v) for (k, v) in d["solve"])
    return CampaignInput(d["name"], d["root"], templates, orbitals, cutoff, overrides,
        Float64(get(di, "eps_out", 1.0)), boxes, epses, solve,
        Int(d["batching"]["n_centers_per_batch"]),
        Float64(get(get(d, "eval", Dict()), "far_pad_steps", 2.0)), toml_path)
end

manifest_path(c::CampaignInput)  = joinpath(c.root, "manifest.tsv")
centers_path(c::CampaignInput)   = joinpath(c.root, "centers.tsv")
targets_path(c::CampaignInput)   = joinpath(c.root, "targets.jls")
rho_store_path(c::CampaignInput) = joinpath(c.root, "rho_store.jls")
logs_dir(c::CampaignInput)       = joinpath(c.root, "logs")
batch_path(c::CampaignInput, id::Int) = joinpath(c.root, "batches", _b4("batch", id))
v_path(c::CampaignInput, id::Int)     = joinpath(c.root, "V", _b4("V", id))
_b4(stem, id) = string(stem, "_", lpad(id, 4, '0'), ".jls")

"Edge/corner refinement target size: `solve.l_ec` if set, else min box Lz / 2^level · 1.01."
function campaign_l_ec(c::CampaignInput)
    haskey(c.solve, "l_ec") && return c.solve["l_ec"]
    level = Int(get(c.solve, "edge_refine_level", 4))
    minimum(b.Lz for b in c.boxes) / 2.0^level * 1.01
end

# worker-local template cache: path => (structure, datagrid)
const TEMPLATE_CACHE = Dict{String,Any}()
function load_templates!(c::CampaignInput)
    return [get!(TEMPLATE_CACHE, p) do
                read_xsf(p)
            end for p in c.templates]
end
