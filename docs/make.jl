using BoundaryIntegral
using Documenter

DocMeta.setdocmeta!(BoundaryIntegral, :DocTestSetup, :(using BoundaryIntegral); recursive=true)

makedocs(;
    modules=[BoundaryIntegral],
    authors="Xuanzhao Gao <xgao@flatironinstitute.org> and contributors",
    sitename="BoundaryIntegral.jl",
    format=Documenter.HTML(;
        canonical="https://xuanzhaogao.github.io/BoundaryIntegral.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "User guide" => "guide.md",
        "Lattice campaigns" => "campaign.md",
        "API reference" => "api.md",
        "Development" => "development.md",
    ],
)

deploydocs(;
    repo="github.com/xuanzhaogao/BoundaryIntegral.jl",
    devbranch="main",
)
