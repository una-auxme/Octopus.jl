```@meta
CurrentModule = Octopus
```

# Octopus.jl 🐙

Fast octree neighborhood search for Julia — on CPU and NVIDIA GPUs.

> *Eight octants, eight arms.* An octree carves space into **8** children at
> every level; an octopus reaches out with **8** arms to grab everything within
> reach. That is exactly what this package does: for every point it grabs all
> neighbors inside a fixed radius, fast, with zero allocations on the hot path.

Octopus.jl is a Julia port of the fast octree neighborhood search for SPH
simulations (Fernández-Fernández et al., SIGGRAPH Asia 2022,
[DOI 10.1145/3550454.3555523](https://dl.acm.org/doi/10.1145/3550454.3555523)).
It mirrors the C++ [TreeNSearch](https://github.com/InteractiveComputerGraphics/TreeNSearch)
reference API and adds Julia-idiomatic and GPU-friendly conveniences on top,
including differentiable edge construction for graph neural networks.

## Why Octopus.jl?

- **Fast.** A cache-friendly Morton/octree build with an almost-sorted refit
  path for time-stepping simulations where points barely move between steps.
- **Allocation-free hot loop.** Iterate neighbors through a callback rather
  than materializing neighbor lists — no garbage on the critical path.
- **CPU and GPU from one API.** The same calls run multithreaded on the CPU
  (Polyester + SIMD) or on NVIDIA GPUs via a CUDA weak extension, where the
  tree build is fully device-resident.
- **GNN-ready.** Build flat-COO edge buffers and differentiate through them
  (Zygote / ChainRules), so neighbor graphs can sit inside a learned model.

## Installation

```julia
using Pkg
Pkg.add("Octopus")
```

Octopus.jl requires Julia 1.12 or newer. The GPU path additionally needs
`CUDA.jl`, and the differentiable path needs `ChainRulesCore.jl` — both are
weak dependencies, loaded on demand:

```julia
Pkg.add("CUDA")             # optional: NVIDIA GPU backend
Pkg.add("ChainRulesCore")   # optional: gradients through build_edges_diff
```

## Quick start

```julia
using Octopus

xyz = rand(Float32, 3, 100_000)

tns = TNS(Float32; ndims = 3)
set_search_radius!(tns, 0.02f0)
id = add_point_set!(tns, xyz)
set_active_search!(tns, id, id)
run!(tns)

for_each_neighbor(tns, id, id, 1) do j
    # use neighbor j of point 1
end
```

## Where to go next

- The [Guide](guide.md) walks through the full workflow: point sets, active
  pairs, iteration, materialized lists, refit mode and z-sorting.
- [GPU (CUDA)](gpu.md) covers the device-resident build and in-kernel neighbor
  iteration.
- [Differentiable edges](gnn.md) shows how to backpropagate through the
  neighbor graph.
- The [API reference](api.md) documents every exported symbol.

## Scope

- 2D and 3D, fixed radius. Pass `ndims = 2` to [`TNS`](@ref) for a quadtree;
  the default `ndims = 3` gives an octree. Coordinates are `(NDIMS, N)`
  matrices.
- `TNS{Float32}` is the tested and benchmarked path. `TNS{Float64}` works but
  is unbenchmarked.
- CPU and CUDA (NVIDIA). AMDGPU and Metal require kernel rewrites and are out
  of scope.
- Refit-mode temporal coherence for time-stepping simulations.

## Citation

If you use this package, please cite the original paper:

```bibtex
@article{FernandezFernandez2022,
  author = {Fern{\'{a}}ndez-Fern{\'{a}}ndez, Jos{\'{e}} Antonio and Westhofen, Lukas and L{\"{o}}schner, Fabian and Jeske, Stefan Rhys and Longva, Andreas and Bender, Jan},
  title = {Fast Octree Neighborhood Search for SPH Simulations},
  journal = {ACM Transactions on Graphics},
  volume = {41},
  number = {6},
  year = {2022},
  doi = {10.1145/3550454.3555523},
}
```

## License

Copyright (c) 2026 Josef Jouaux.
Released under the MIT License.
