# Copyright (c) 2026 Josef Jouaux
# Copyright (c) 2022-present, CompactNSearch contributors

using Documenter
using Octopus

DocMeta.setdocmeta!(Octopus, :DocTestSetup, :(using Octopus); recursive = true)

makedocs(;
    modules = [Octopus],
    authors = "Josef Jouaux",
    sitename = "Octopus.jl",
    format = Documenter.HTML(;
        canonical = "https://una-auxme.github.io/Octopus.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md",
        "GPU (CUDA)" => "gpu.md",
        "Differentiable edges" => "gnn.md",
        "API reference" => "api.md",
    ],
    # Every exported symbol must appear in the API reference.
    checkdocs = :exports,
)

deploydocs(;
    repo = "github.com/una-auxme/Octopus.jl",
    devbranch = "main",
    push_preview = true,
)
