"""
    write_v_table(path, pair_ids, V)

Write the four-index result as a text table: header `i\\tj\\tk\\tl\\tV`, then one row per
entry `i j k l value` where `(i,j) = pair_ids[row]`, `(k,l) = pair_ids[col]`,
`value = V[row, col]`. All entries (dense). Atomic (tmp + rename).
"""
function write_v_table(path::AbstractString, pair_ids::Vector{Tuple{Int,Int}}, V::AbstractMatrix)
    nr, nc = size(V)
    (nr == length(pair_ids) && nc == length(pair_ids)) ||
        throw(DimensionMismatch("V is $(size(V)) but pair_ids has $(length(pair_ids))"))
    d = dirname(path); isempty(d) || mkpath(d)
    tmp = string(path, ".tmp.", getpid())
    open(tmp, "w") do io
        println(io, "i\tj\tk\tl\tV")
        for r in 1:nr, cc in 1:nc
            (i, j) = pair_ids[r]; (k, l) = pair_ids[cc]
            println(io, i, '\t', j, '\t', k, '\t', l, '\t', V[r, cc])
        end
    end
    mv(tmp, path; force = true)
    return path
end
