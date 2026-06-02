# TreeNSearch.jl

Julia port of the fast octree neighborhood search for SPH simulations
(Fernández-Fernández et al., SIGGRAPH Asia 2022, [DOI 10.1145/3550454.3555523](https://dl.acm.org/doi/10.1145/3550454.3555523)).

Mirrors the C++ [TreeNSearch](https://github.com/InteractiveComputerGraphics/TreeNSearch) API.
Memory-efficient: ~6 B/pt tree overhead vs 12–18 B/pt for `PointNeighbors.jl` cell-list backends,
and zero-allocation iteration via `for_each_neighbor` (host) or `for_each_neighbor_device`
(inside the user's `@cuda` kernel, NVIDIA only in v0.1).

## Quick start

```julia
using TreeNSearch

xyz = rand(Float32, 3, 100_000)
tns = TNS(Float32; ndims=3)
set_search_radius!(tns, 0.02f0)
id = add_point_set!(tns, xyz)
set_symmetric_search!(tns, id, id)
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
- Host iteration, device iteration, and opt-in CSR materialization
  (`materialize_all_neighbors!` is CPU-only in v0.1; on GPU use
  `for_each_neighbor_device` / `@for_each_neighbor_device_inline` /
  `build_edges`).
- Refit-mode temporal coherence for time-stepping simulations.

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
