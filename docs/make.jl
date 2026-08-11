#
# Copyright (c) 2026 Josef Jouaux
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
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
    # Run the `jldoctest` blocks as tests. On by default, but stated explicitly:
    # a silent flip to `false` would leave the examples rotting unnoticed.
    doctest = true,
    # Turn doc warnings (broken @ref links, missing docstrings, failed
    # doctests) into a failed build rather than console noise CI ignores.
    warnonly = false,
)

deploydocs(;
    repo = "github.com/una-auxme/Octopus.jl",
    devbranch = "main",
    push_preview = true,
)
