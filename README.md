# TreeNSearch.jl

Julia port of the fast octree neighborhood search for SPH simulations
(Fernández-Fernández et al., SIGGRAPH Asia 2022, [DOI 10.1145/3550454.3555523](https://dl.acm.org/doi/10.1145/3550454.3555523)).

Mirrors the C++ [TreeNSearch](https://github.com/InteractiveComputerGraphics/TreeNSearch) API.
Zero-allocation iteration via `for_each_neighbor` (host) or `for_each_neighbor_device`
(inside the user's `@cuda` kernel, NVIDIA only in v0.1).

## Quick start

```julia
using TreeNSearch

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
`get_neighborlist`, `prepare_zsort!`, `apply_zsort!`.

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

## Citation

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
