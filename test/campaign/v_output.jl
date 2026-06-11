using BoundaryIntegral, Test

@testset "v_output round-trip" begin
    pair_ids = [(1,1),(1,2),(2,2)]
    V = [1.0 2.0 3.0; 2.0 4.0 5.0; 3.0 5.0 6.0]
    mktempdir() do dir
        p = joinpath(dir, "V_full.tsv")
        BoundaryIntegral.write_v_table(p, pair_ids, V)
        lines = readlines(p)
        @test lines[1] == "i\tj\tk\tl\tV"
        @test length(lines) == 1 + 9          # header + 3×3 entries
        # parse back
        got = Dict{NTuple{4,Int},Float64}()
        for ln in lines[2:end]
            f = split(ln, '\t'); got[(parse(Int,f[1]),parse(Int,f[2]),parse(Int,f[3]),parse(Int,f[4]))] = parse(Float64, f[5])
        end
        @test got[(1,1,1,2)] == 2.0    # (i,j)=pair_ids[1]=(1,1), (k,l)=pair_ids[2]=(1,2)
        @test got[(2,2,2,2)] == 6.0
        @test length(got) == 9
    end
end
