# Octopus.jl 🐙

[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://una-auxme.github.io/Octopus.jl/dev/)
[![CI](https://github.com/una-auxme/Octopus.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/una-auxme/Octopus.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/una-auxme/Octopus.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/una-auxme/Octopus.jl)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Fast octree neighborhood search for Julia — on CPU and NVIDIA GPUs.

> *Eight octants, eight arms.* An octree carves space into **8** children at every
> level; an octopus reaches out with **8** arms to grab everything within reach.
> That is exactly what this package does: for every point it grabs all neighbors
> inside a fixed radius, fast, with zero allocations on the hot path.

Julia port of the fast octree neighborhood search for SPH simulations
(Fernández-Fernández et al., SIGGRAPH Asia 2022,
[DOI 10.1145/3550454.3555523](https://dl.acm.org/doi/10.1145/3550454.3555523)).
It mirrors the C++ [TreeNSearch](https://github.com/InteractiveComputerGraphics/TreeNSearch)
reference API, and adds Julia-idiomatic and GPU-friendly conveniences on top
(including differentiable edge construction for graph neural networks).

Zero-allocation iteration via `for_each_neighbor` (host) or
`for_each_neighbor_device` (inside the user's `@cuda` kernel, NVIDIA only in v0.1).

**[📖 Documentation](https://una-auxme.github.io/Octopus.jl/dev/)**

## Why Octopus.jl?

- **Fast.** A cache-friendly Morton/octree build with an almost-sorted refit path
  for time-stepping simulations where points barely move between steps.
- **Allocation-free hot loop.** Iterate neighbors through a callback rather than
  materializing neighbor lists — no garbage on the critical path.
- **CPU and GPU from one API.** The same calls run on the CPU or on NVIDIA GPUs
  via a CUDA weak extension, where the tree build is fully device-resident. On
  the CPU the binning pass is multithreaded (Polyester); the tree build, refit,
  and edge construction are currently serial.
- **GNN-ready.** Build flat-COO edge buffers and differentiate through them
  (Zygote / ChainRules), so neighbor graphs can sit inside a learned model.

## Installation

```julia
using Pkg
Pkg.add("Octopus")
```

Requires Julia 1.12 or newer. The GPU and autodiff paths are weak dependencies,
loaded only when you import them:

```julia
Pkg.add("CUDA")             # optional: NVIDIA GPU backend
Pkg.add("ChainRulesCore")   # optional: gradients via build_edges_diff
```

## Quick start

```julia
using Octopus

xyz = rand(Float32, 3, 100_000)
tns = TNS(Float32; ndims=3)
set_search_radius!(tns, 0.02f0)
id = add_point_set!(tns, xyz)
set_active_search!(tns, id, id)
run!(tns)

for_each_neighbor(tns, id, id, 1) do j
    # use neighbor j of point 1
end
```

## GPU usage (CUDA)

The GPU path lives in a weak extension that loads automatically when `CUDA` is
imported. Pass `CuArray` coordinates and the same API runs on the device: the
origin reduction, Morton encoding, sort, topology pass and bounds refit all run
in device kernels, so coordinates never round-trip through the host. NVIDIA
only. See the [GPU guide](https://una-auxme.github.io/Octopus.jl/dev/gpu/) for
in-kernel iteration and tuning.

```julia
using Octopus, CUDA            # importing CUDA activates the GPU extension

xyz = CuArray(rand(Float32, 3, 100_000))   # coords on the device
tns = TNS(Float32; ndims=3)
set_search_radius!(tns, 0.02f0)
id  = add_point_set!(tns, xyz)              # CuArray coords ⇒ :cuda backend
set_active_search!(tns, id, id)
run!(tns)

# High-level: flat-COO edge buffers, all CuArrays, ready to drop into a GNN.
e = build_edges(tns, id, id)
# e.senders, e.receivers :: CuArray{Int32}
# e.rel_displacement     :: CuArray{Float32} (NDIMS, n_edges)   (self-pair excludes j == i)
# e.rel_dist_norm        :: CuArray{Float32} (1, n_edges)
```

For zero-allocation neighbor iteration inside your own `@cuda` kernel, grab a
`device_view` (an isbits handle holding the device arrays) and call
`for_each_neighbor_device` per query point. Note the callback writes into an
array slot rather than a mutable local — a captured scalar would be boxed and
rejected by the GPU compiler:

```julia
dv = device_view(tns, id, id)

function count_neighbors!(counts, dv)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(counts) && return
    for_each_neighbor_device(dv, i) do _    # callback arg is the neighbor index j
        @inbounds counts[i] += Int32(1)     # write the slot, not a boxed local
    end
    return
end

counts = CUDA.zeros(Int32, size(xyz, 2))
threads = 128
@cuda threads=threads blocks=cld(length(counts), threads) count_neighbors!(counts, dv)
```

For the hot path, `@for_each_neighbor_device_inline dv i j begin … end` (and the
`_2d` variant) inlines the traversal so the cursor stays in a register — measurably
faster than the closure form when emitting many edges per point.

## Differentiable edges (GNN)

`build_edges_diff` backpropagates through the edge features, so a radius graph
can sit inside a learned model. Gradients reach the coordinates via
`rel_displacement` and `rel_dist_norm`; the tree topology and the edge indices
are treated as constants.

```julia
using Octopus, ChainRulesCore, Zygote   # ChainRulesCore activates the extension

radius = 0.25f0

function loss(coords)
    tns = TNS(Float32)
    set_search_radius!(tns, radius)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)

    e = build_edges_diff(coords, tns, id, radius)
    return sum(abs2, e.rel_displacement) + 0.5f0 * sum(abs2, e.rel_dist_norm)
end

g = Zygote.gradient(loss, rand(Float32, 3, 1_000))[1]   # (3, 1000)
```

Add `using CUDA` for the GPU rule. Details and correctness notes in the
[differentiable-edges guide](https://una-auxme.github.io/Octopus.jl/dev/gnn/).

## Documentation

- [Guide](https://una-auxme.github.io/Octopus.jl/dev/guide/) — the full
  workflow: point sets, active pairs, iteration, materialized lists, refit
  mode, z-sorting.
- [GPU (CUDA)](https://una-auxme.github.io/Octopus.jl/dev/gpu/) — device
  build, in-kernel iteration, tuning.
- [Differentiable edges](https://una-auxme.github.io/Octopus.jl/dev/gnn/) —
  gradients through the neighbor graph.
- [API reference](https://una-auxme.github.io/Octopus.jl/dev/api/) — every
  exported symbol.

## Development

```console
$ git clone https://github.com/una-auxme/Octopus.jl
$ cd Octopus.jl
$ julia --project=. -e 'using Pkg; Pkg.test()'
```

Two test groups are opt-in via environment variables, since they need extra
precompilation or hardware:

| Variable | Enables | Needs |
|---|---|---|
| `JULIA_OCTOPUS_TEST_CHAINRULES=1` | ChainRules / Zygote gradient tests | — (runs in CI) |
| `JULIA_OCTOPUS_TEST_CUDA=1` | CUDA backend tests | an NVIDIA GPU |

To build the docs locally:

```console
$ julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
$ julia --project=docs docs/make.jl
```

## Scope (v0.1)

- 2D and 3D, fixed radius. Pass `ndims=2` to `TNS` for a quadtree; the
  default is `ndims=3` (octree). Coords are `(NDIMS, N)` matrices.
  `TNS{Float32}` is the tested and benchmarked path; `TNS{Float64}` works
  but is unbenchmarked.
- CPU + CUDA (NVIDIA). AMDGPU/Metal require kernel rewrites and are out of scope.
- Refit-mode temporal coherence for time-stepping simulations.

## Paper-aligned API

The following names mirror the paper and the C++ TreeNSearch reference
implementation directly: `TNS`, `set_search_radius!`, `add_point_set!`,
`resize_point_set!`, `set_active_search!`, `run!`, `for_each_neighbor`,
`get_neighborlist`, `prepare_zsort`, `apply_zsort`.

The two z-sort entry points dropped their `!` in v0.2 — neither mutates an
argument, so the bang was misleading under the Julia convention. The v0.1
spellings `prepare_zsort!` / `apply_zsort!` still work and are deprecated.

## Julia-only extensions

The following are convenience layers added on top of the paper's API; they
are not part of the upstream C++ TreeNSearch surface and exist to fit
Julia idioms or GPU usage:

- `for_each_neighbor_device`, `device_view`,
  `@for_each_neighbor_device_inline[_2d]` — in-kernel iteration for CUDA.
- `set_refit_mode!` — exposes the paper's almost-sorted refit path as a
  user-controlled toggle.
- `update_point_set!` — strict same-N update with a dimension check; users
  who need to change N call `resize_point_set!` instead.
- `materialize_all_neighbors!`, `get_neighborlist` (Vector return),
  `build_edges` / `build_edges!` / `EdgeBuffer` — opt-in CSR / flat-COO
  materialisations of the neighbor relation, useful for callers that want
  a dense list or a graph-edge buffer rather than the callback iteration
  primitive.
- `build_edges_diff` — differentiable edge construction (gradients flow to
  coordinates) for graph-neural-network pipelines, via the ChainRulesCore
  weak extension.

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

Released under the MIT License — see [LICENSE](LICENSE).
