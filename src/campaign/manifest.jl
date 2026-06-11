"""
    CenterInfo(id, template_id, steps, center)

One orbital site: 1-based `id`, `template_id` (index into `CampaignInput.templates`), the
integer global-grid offset `steps`, and the realized Cartesian `center`.
"""
struct CenterInfo
    id::Int
    template_id::Int
    steps::NTuple{3,Int}
    center::NTuple{3,Float64}
end

struct BatchSpec
    batch_id::Int
    anchors::Vector{Int}
    pairs::Vector{Tuple{Int,Int}}
end

Base.:(==)(a::CenterInfo, b::CenterInfo) =
    a.id == b.id && a.template_id == b.template_id && a.steps == b.steps && a.center == b.center
Base.:(==)(a::BatchSpec, b::BatchSpec) =
    a.batch_id == b.batch_id && a.anchors == b.anchors && a.pairs == b.pairs

"""
    enumerate_centers(c::CampaignInput) -> Vector{CenterInfo}

Snap each explicit `[[orbital]]` to its integer grid offset (against its template's grid)
and record the realized Cartesian center. Orbital id = 1-based index in `c.orbitals`.
"""
function enumerate_centers(c::CampaignInput)
    temps = load_templates!(c)
    out = CenterInfo[]
    for (id, o) in enumerate(c.orbitals)
        dg = temps[o.type][2]
        c0 = ntuple(d -> Float64(density_centroid(dg)[d]), 3)
        steps = snap_orbital(dg, c0, o.pos)
        At, Bt, Ct = true_cell_vectors(dg)
        G = hcat(collect(At) ./ dg.nx, collect(Bt) ./ dg.ny, collect(Ct) ./ dg.nz)
        realized = collect(c0) .+ G * collect(steps)
        push!(out, CenterInfo(id, o.type, steps, (realized[1], realized[2], realized[3])))
    end
    return out
end

"Unique pairs (i ≤ j) with center distance ≤ cutoff. On-site pairs (i,i) included."
function enumerate_pairs(centers::Vector{CenterInfo}, cutoff::Real)
    byid = sort(centers; by = c -> c.id)
    pairs = Tuple{Int,Int}[]
    for (m, ci) in enumerate(byid), cj in byid[m:end]
        sqrt(sum(abs2, ci.center .- cj.center)) <= cutoff && push!(pairs, (ci.id, cj.id))
    end
    return pairs
end

"""
    build_batches(pairs, n_centers_per_batch) -> Vector{BatchSpec}

Each pair belongs to its anchor (= min id); consecutive anchors merged n at a time.
"""
function build_batches(pairs::Vector{Tuple{Int,Int}}, n_centers_per_batch::Int)
    by_anchor = Dict{Int,Vector{Tuple{Int,Int}}}()
    for p in pairs
        push!(get!(by_anchor, min(p[1], p[2]), Tuple{Int,Int}[]), p)
    end
    anchors = sort(collect(keys(by_anchor)))
    out = BatchSpec[]; bid = 0
    for grp in Iterators.partition(anchors, n_centers_per_batch)
        bid += 1
        push!(out, BatchSpec(bid, collect(grp), reduce(vcat, (sort(by_anchor[a]) for a in grp))))
    end
    return out
end

function write_centers(path::AbstractString, centers::Vector{CenterInfo})
    d = dirname(path); isempty(d) || mkpath(d)
    open(path, "w") do io
        println(io, "id\ttemplate\tsx\tsy\tsz\tcx\tcy\tcz")
        for c in sort(centers; by = c -> c.id)
            int = join([c.id, c.template_id, c.steps...], '\t')
            flt = join(repr.(c.center), '\t')
            println(io, int, '\t', flt)
        end
    end
end

function read_centers(path::AbstractString)
    out = CenterInfo[]
    for (n, line) in enumerate(eachline(path))
        n == 1 && continue
        f = split(line, '\t')
        push!(out, CenterInfo(parse(Int, f[1]), parse(Int, f[2]),
            (parse(Int, f[3]), parse(Int, f[4]), parse(Int, f[5])),
            (parse(Float64, f[6]), parse(Float64, f[7]), parse(Float64, f[8]))))
    end
    return out
end

function write_manifest(path::AbstractString, batches::Vector{BatchSpec})
    d = dirname(path); isempty(d) || mkpath(d)
    open(path, "w") do io
        println(io, "batch_id\tanchors\tK\tpairs")
        for b in sort(batches; by = b -> b.batch_id)
            ps = join(("$(i):$(j)" for (i, j) in b.pairs), ';')
            println(io, join([b.batch_id, join(b.anchors, ','), length(b.pairs), ps], '\t'))
        end
    end
end

function read_manifest(path::AbstractString)
    out = BatchSpec[]
    for (n, line) in enumerate(eachline(path))
        n == 1 && continue
        f = split(line, '\t')
        pairs = isempty(strip(f[4])) ? Tuple{Int,Int}[] :
            [(parse(Int, split(p, ':')[1]), parse(Int, split(p, ':')[2])) for p in split(f[4], ';')]
        push!(out, BatchSpec(parse(Int, f[1]), parse.(Int, split(f[2], ',')), pairs))
    end
    return out
end
