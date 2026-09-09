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

This groups by a fixed ANCHOR COUNT, so the batch size K (= pairs per batch) is whatever the
neighbour cutoff happens to give that anchor: on the 198-orbital / 2623-pair benchmark, K runs
from 1 to 17 with a mean of 13.2. Prefer the `centers`/`k_target` method below, which groups to
a fixed K instead -- every batch pays a full interface build, corrected LHS operator and pottrg
map, so the batch COUNT is what those fixed costs scale with.
"""
function build_batches(pairs::Vector{Tuple{Int,Int}}, n_centers_per_batch::Int)
    by_anchor = _pairs_by_anchor(pairs)
    anchors = sort(collect(keys(by_anchor)))
    out = BatchSpec[]; bid = 0
    for grp in Iterators.partition(anchors, n_centers_per_batch)
        bid += 1
        push!(out, BatchSpec(bid, collect(grp), reduce(vcat, (sort(by_anchor[a]) for a in grp))))
    end
    return out
end

function _pairs_by_anchor(pairs::Vector{Tuple{Int,Int}})
    by_anchor = Dict{Int,Vector{Tuple{Int,Int}}}()
    for p in pairs
        push!(get!(by_anchor, min(p[1], p[2]), Tuple{Int,Int}[]), p)
    end
    return by_anchor
end

"""
    build_batches(pairs, centers, k_target) -> Vector{BatchSpec}

Partition the pairs into batches of approximately `k_target` pairs each, keeping the pairs of a
batch spatially together.

Why K and not an anchor count: a batch's cost is (fixed work) + K x (per-pair work). The fixed
part -- one adaptive interface, one corrected LHS operator, one pottrg map over the evaluation
targets -- is paid once per batch regardless of K, so the batch COUNT drives it. Grouping by
anchor count leaves K to the cutoff geometry (1 to 17 on the 198-orbital benchmark, mean 13.2),
giving both more batches than necessary and a wide spread of per-batch cost, which unbalances a
distributed run.

This is a graph-partitioning problem: with orbitals as vertices and pairs as edges, the pairs
form a line graph whose vertices we want split into equal parts with few cut edges -- two pairs
sharing an orbital share source support, so keeping them together shrinks the batch envelope.
For this geometry (a planar lattice; every orbital lies in one plane) recursive coordinate
bisection on the pair midpoints realizes that objective directly and deterministically, with no
graph library: pairs sharing an orbital have nearby midpoints, so geometric compactness and a
small edge cut coincide. Parts come out within one pair of equal size.

Locality matters because the batch's interface is refined on the envelope of its sources: pairs
that are far apart inflate the envelope and with it the interface, costing more in the solve
than the batching saves.

Note the pairs of one anchor are NOT forced into the same batch -- `BatchSpec.anchors` is
metadata (the manifest TSV and equality) and nothing computational depends on that grouping.
Freeing it is what lets the parts hit `k_target` closely.
"""
function build_batches(pairs::Vector{Tuple{Int,Int}}, centers::Vector{CenterInfo},
                       k_target::Int)
    k_target >= 1 || throw(ArgumentError("k_target must be >= 1, got $(k_target)"))
    isempty(pairs) && return BatchSpec[]

    pos = Dict(c.id => c.center for c in centers)
    for p in pairs
        (haskey(pos, p[1]) && haskey(pos, p[2])) ||
            throw(ArgumentError("build_batches: pair $p references an unknown center"))
    end
    mid = [((pos[p[1]][1] + pos[p[2]][1]) / 2,
            (pos[p[1]][2] + pos[p[2]][2]) / 2,
            (pos[p[1]][3] + pos[p[2]][3]) / 2) for p in pairs]

    nparts = max(1, round(Int, length(pairs) / k_target))
    parts = Vector{Vector{Int}}()
    _rcb!(parts, collect(1:length(pairs)), mid, nparts)

    out = BatchSpec[]
    for (bid, part) in enumerate(parts)
        ps = sort(pairs[part])
        anchors = sort(unique(min(p[1], p[2]) for p in ps))
        push!(out, BatchSpec(bid, anchors, ps))
    end
    return out
end

# Recursive coordinate bisection: split along the axis of greatest extent at the position that
# divides the parts-to-be as evenly as the remaining part count requires. Sizes land within one
# element of equal. `mid` holds one midpoint per pair; `idx` indexes into it.
function _rcb!(out::Vector{Vector{Int}}, idx::Vector{Int},
               mid::Vector{NTuple{3,Float64}}, nparts::Int)
    if nparts <= 1 || length(idx) <= 1
        push!(out, idx)
        return
    end
    ext = ntuple(d -> begin
        v = (mid[i][d] for i in idx)
        maximum(v) - minimum(v)
    end, 3)
    ax = argmax(ext)
    perm = sortperm(idx; by = i -> (mid[i][ax], i))   # tie-break on index: deterministic
    sorted = idx[perm]
    n1 = nparts ÷ 2
    cut = clamp(round(Int, length(idx) * n1 / nparts), 1, length(idx) - 1)
    _rcb!(out, sorted[1:cut], mid, n1)
    _rcb!(out, sorted[cut+1:end], mid, nparts - n1)
    return
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
