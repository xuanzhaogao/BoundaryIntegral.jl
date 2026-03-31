using Test
using BoundaryIntegral
const BI = BoundaryIntegral

@testset "singular panel creation" begin
    box = BI.single_dielectric_box2d(1.0, 1.0, 8, 0.2, 0.05, 5.0, 1.0, Float64; use_singular=true)
    n_singular = count(p -> p.is_singular, box.panels)
    @test n_singular > 0
    @test n_singular == 8  # 4 corners x 2 edges per corner
    for p in box.panels
        if p.is_singular
            @test p.singular_exponent != 0.0
        else
            @test p.singular_exponent == 0.0
        end
    end
    # Non-singular box still works
    box_reg = BI.single_dielectric_box2d(1.0, 1.0, 8, 0.2, 0.05, 5.0, 1.0, Float64; use_singular=false)
    @test count(p -> p.is_singular, box_reg.panels) == 0
end
