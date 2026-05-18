using BoundaryIntegral
import BoundaryIntegral as BI
using Test

@testset "AdaptiveConfig defaults" begin
    cfg = BI.AdaptiveConfig(atol = 1e-8)
    @test cfg.atol == 1e-8
    @test cfg.rtol == sqrt(eps(Float64))
    @test cfg.n_GL == 0
    @test cfg.max_depth == 20

    cfg2 = BI.AdaptiveConfig(atol = 1e-6, rtol = 1e-10, n_GL = 8, max_depth = 12)
    @test cfg2.n_GL == 8
    @test cfg2.max_depth == 12
end
