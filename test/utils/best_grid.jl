@testset "best_grid" begin
    @test BoundaryIntegral.best_grid_mn(20.0, 1.0, sqrt(2)) == (15, 1)
    @test BoundaryIntegral.best_grid_mn(2.0, 1.0, sqrt(2)) == (2, 1)
    @test BoundaryIntegral.best_grid_mn(1.0, 2.0, sqrt(2)) == (1, 2)
    @test BoundaryIntegral.best_grid_mn(1.0, 1.0, 1.0) == (1, 1)
    @test BoundaryIntegral.best_grid_mn(3.0, 2.0, 1.0) == (3, 2)
    @test_throws ArgumentError BoundaryIntegral.best_grid_mn(0.0, 1.0, 1.0)
    @test_throws ArgumentError BoundaryIntegral.best_grid_mn(1.0, 0.0, 1.0)
    @test_throws ArgumentError BoundaryIntegral.best_grid_mn(1.0, 1.0, 0.99)
end
