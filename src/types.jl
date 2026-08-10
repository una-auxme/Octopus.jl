#
# Copyright (c) 2026 Josef Jouaux
# Copyright (c) 2022-present, CompactNSearch contributors
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# Core data structures, shared by the CPU and CUDA paths (the CUDA ext swaps
# Array for CuArray through the type parameters).

using StaticArrays

struct PointSet{T,NDIMS,A<:AbstractMatrix{T}}
    coords::A              # (NDIMS, N), user-owned reference, never copied
    n::Int32
end

function PointSet(coords::AbstractMatrix{T}) where {T}
    NDIMS = size(coords, 1)
    PointSet{T,NDIMS,typeof(coords)}(coords, Int32(size(coords, 2)))
end

# Flat SoA octree (2^NDIMS-way). A node is a leaf iff children[1] == -1.
mutable struct Octree{T,NDIMS,
                     VI<:AbstractVector{Int32},
                     MF<:AbstractMatrix{T},
                     MI<:AbstractMatrix{Int32}}
    node_bounds_min::MF    # (NDIMS, n_nodes)
    node_bounds_max::MF    # (NDIMS, n_nodes)
    node_first::VI         # inclusive first idx into permutation
    node_last::VI          # inclusive last idx
    node_children::MI      # (2^NDIMS, n_nodes); children[1,i]==-1 marks a leaf
    n_nodes::Int32
end

function Octree{T,NDIMS}(backend_like::AbstractArray) where {T,NDIMS}
    # Empty skeleton, grown on first build. 2^NDIMS children (4=quadtree, 8=octree).
    MF = similar(backend_like, T, (NDIMS, 0))
    MI = similar(backend_like, Int32, (1 << NDIMS, 0))
    VI = similar(backend_like, Int32, (0,))
    Octree{T,NDIMS,typeof(VI),typeof(MF),typeof(MI)}(
        MF, similar(MF), VI, similar(VI), MI, Int32(0))
end

mutable struct NeighborBuffer{VI<:AbstractVector{Int32}}
    flat::VI               # CSR values; empty until materialized
    offsets::VI            # length n_query_points + 1
    materialized::Bool
end

NeighborBuffer(like::AbstractArray) = NeighborBuffer(
    similar(like, Int32, (0,)),
    similar(like, Int32, (1,)),  # offsets[1] = 0
    false,
)

# Flat-COO edges for downstream GNN consumers (not part of upstream TreeNSearch).
# senders[k] = target j, receivers[k] = query i,
# rel_displacement = (coords_q[:,i] - coords_t[:,j]) / radius,
# rel_dist_norm    = ‖coords_q[:,i] - coords_t[:,j]‖ / radius.
"""
    EdgeBuffer

Flat-COO storage for the neighbor relation of one active pair, shaped for
graph-neural-network consumers. Populated by `build_edges!`; `build_edges`
returns the same arrays as a NamedTuple.

Fields, with `k` ranging over the `n_edges` emitted edges:

- `senders::AbstractVector{Int32}` — target index `j`.
- `receivers::AbstractVector{Int32}` — query index `i`.
- `rel_displacement::AbstractMatrix{T}` — `(NDIMS, n_edges)`, the radius-normalized
  displacement `(coords_q[:, i] - coords_t[:, j]) / radius`.
- `rel_dist_norm::AbstractMatrix{T}` — `(1, n_edges)`, the radius-normalized distance
  `‖coords_q[:, i] - coords_t[:, j]‖ / radius`.
- `n_edges::Int32`, `materialized::Bool` — fill state.

The arrays live on the same device as the point set, so a CUDA `TNS` yields
`CuArray`s ready to hand to a model. Storage is reused across calls: a later
`build_edges!` on the same pair overwrites it. Take copies if you need the
values to outlive the next call — `build_edges_diff` does exactly this.

Julia-only addition, not part of the upstream C++ TreeNSearch API.
"""
mutable struct EdgeBuffer{T,VI<:AbstractVector{Int32},MF<:AbstractMatrix{T}}
    senders::VI            # length n_edges
    receivers::VI          # length n_edges
    rel_displacement::MF   # (NDIMS, n_edges)
    rel_dist_norm::MF      # (1, n_edges)
    n_edges::Int32
    materialized::Bool
end

function make_edge_buffer(::Type{T}, ndims::Integer, like::AbstractArray) where {T}
    senders   = similar(like, Int32, (0,))
    receivers = similar(like, Int32, (0,))
    rel_disp  = similar(like, T, (Int(ndims), 0))
    rel_dist  = similar(like, T, (1, 0))
    EdgeBuffer{T,typeof(senders),typeof(rel_disp)}(
        senders, receivers, rel_disp, rel_dist, Int32(0), false)
end

# Main handle. `dev ∈ (:cpu, :cuda)` is inferred from the first add_point_set!.
"""
    TNS(T = Float32; ndims = 3) -> TNS{T,ndims}

The main handle: holds the registered point sets, their trees, the active
search pairs and all reusable scratch. `T` is the coordinate element type and
`ndims` selects the spatial dimension — `3` builds an octree, `2` a quadtree.

`TNS{Float32}` is the tested and benchmarked configuration. `TNS{Float64}`
works but is unbenchmarked.

The backend is not chosen here. It is inferred on the first
[`add_point_set!`](@ref) from the array type of the coordinates: a plain
`Array` selects the CPU path, a GPU array (e.g. `CuArray`) the CUDA path. All
point sets on one handle must share a backend.

A typical setup, in order:

```julia
tns = TNS(Float32; ndims = 3)
set_search_radius!(tns, 0.02f0)
id = add_point_set!(tns, xyz)      # xyz is (3, N)
set_active_search!(tns, id, id)
run!(tns)
```

Per-thread traversal stacks are preallocated for `Threads.maxthreadid()`
threads and grown on demand, so queries are safe to issue from inside threaded
loops.
"""
mutable struct TNS{T,NDIMS}
    dev::Symbol
    radius::T
    target_leaf_size::Int32
    refit_mode::Bool

    # These five are heterogeneous by design: the backend is not known until the
    # first `add_point_set!`, so a CPU handle stores `Vector{Int32}`/`Matrix{T}`
    # here and a CUDA handle stores the `CuArray` equivalents. Encoding that in
    # the type parameters would mean fixing the backend at construction and
    # giving up the inference described in `add_point_set!`.
    #
    # They are `Vector{Any}` rather than the weaker `Vector`, which is the
    # abstract `Vector{U} where U`: with `Vector`, even `length`/`getindex` on
    # the container dispatch dynamically. `Vector{Any}` is a concrete container,
    # so only the *elements* stay dynamic — and every hot path crosses a
    # function barrier (`_for_each_neighbor_barrier`, `build_edges_cpu!`,
    # `materialize_cpu!`) that specializes on the concrete element types. The
    # allocation tests in test/memory.jl pin the result: 16 B at the entry
    # point, 0 B inside the barrier.
    point_sets::Vector{Any}
    active_pairs::Vector{Tuple{Int32,Int32}}
    trees::Vector{Any}

    morton_codes::Vector{Any}
    permutation::Vector{Any}

    neighbor_buffers::Vector{NeighborBuffer}
    edge_buffers::Vector{EdgeBuffer}
    build_scratch::Vector{Int32}          # reused Pass-A scratch for build_edges_cpu!
    query_stacks::Vector{Vector{Int32}}   # one stack per thread, indexed by threadid()

    dirty::Vector{Bool}
end

function TNS(::Type{T}=Float32; ndims::Int=3) where {T<:AbstractFloat}
    # Size to maxthreadid so Julia 1.12+ dynamic thread migration can't index OOB.
    nstacks = max(Threads.maxthreadid(), Threads.nthreads())
    # Each traversal stack holds the worst-case DFS frontier (148 entries in 3D,
    # 94 in 2D — under 600 B/thread), so the `@inbounds` pushes in the traversal
    # are bounded by construction rather than by assumption.
    depth = _max_stack_depth(Val(ndims))
    stacks = [zeros(Int32, depth) for _ in 1:nstacks]
    TNS{T,ndims}(
        :uninitialized,
        zero(T), Int32(32), false,
        Any[], Tuple{Int32,Int32}[], Any[],
        Any[], Any[],
        NeighborBuffer[], EdgeBuffer[], Int32[], stacks,
        Bool[],
    )
end

# Adapt plumbing for users who move a TNS across devices by hand; kernel entry
# uses device_view for isbits safety instead.
import Adapt

Adapt.@adapt_structure PointSet
Adapt.@adapt_structure Octree
Adapt.@adapt_structure NeighborBuffer
Adapt.@adapt_structure EdgeBuffer
