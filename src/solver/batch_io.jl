# src/solver/batch_io.jl
# Solve-phase output container + atomic file IO (spec §3). Files are written
# tmp-then-rename on the SAME filesystem, so a batch file either exists complete or
# not at all; status scans never see partial writes.

using Serialization

const BATCH_FORMAT_VERSION = 1

"""
    BatchResult

Everything the post-eval phase needs from one solved batch: the raw truncated pair
densities (global-grid indexed), the adapted interface, the layer densities Σ, and
solve stats. Positions are NOT stored — they are recomputed from `gidx` + the template.
"""
struct BatchResult
    version::Int
    batch_id::Int
    pair_ids::Vector{Tuple{Int,Int}}
    gidx::Vector{NTuple{3,Int}}
    weights::Vector{Float64}
    densities::Matrix{Float64}        # n × K raw pair densities
    interface::Any                    # DielectricInterface (serialized as-is)
    sigma::Matrix{Float64}            # N × K
    stats::Dict{String,Any}
end

"Atomic write: serialize to `<path>.tmp.<pid>`, then rename onto `path`."
function save_batch_result(path::AbstractString, br::BatchResult)
    mkpath(dirname(path))
    tmp = string(path, ".tmp.", getpid())
    open(tmp, "w") do io
        serialize(io, br)
    end
    mv(tmp, path; force = true)
    return path
end

"""
    load_batch_result(path) -> BatchResult

Deserialize and validate (format version, internal shape consistency). Throws on any
problem — callers treating a batch as done must use `is_complete_batch` first.
"""
function load_batch_result(path::AbstractString)
    br = open(deserialize, path)
    br isa BatchResult || error("$path: not a BatchResult")
    br.version == BATCH_FORMAT_VERSION ||
        error("$path: format version $(br.version) ≠ $(BATCH_FORMAT_VERSION)")
    n, K = size(br.densities)
    length(br.gidx) == n || error("$path: gidx/densities mismatch")
    length(br.weights) == n || error("$path: weights/densities mismatch")
    length(br.pair_ids) == K || error("$path: pair_ids/densities mismatch")
    size(br.sigma, 2) == K || error("$path: sigma/densities mismatch")
    return br
end

"True iff `path` exists and loads as a valid, current-version BatchResult."
function is_complete_batch(path::AbstractString)
    isfile(path) || return false
    try
        load_batch_result(path)
        return true
    catch
        return false
    end
end
