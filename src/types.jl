# Core data structures. CPU and CUDA paths share these shapes; the CUDA ext
# parameterizes the array types differently (CuArray vs Array).

using StaticArrays

struct PointSet{T,NDIMS,A<:AbstractMatrix{T}}
    coords::A              # (NDIMS, N) user-owned reference, never copied
    n::Int32
end

function PointSet(coords::AbstractMatrix{T}) where {T}
    NDIMS = size(coords, 1)
    PointSet{T,NDIMS,typeof(coords)}(coords, Int32(size(coords, 2)))
end

# Flat SoA octree. Leaves are internal nodes whose children[1] == -1.
mutable struct Octree{T,NDIMS,
                     VI<:AbstractVector{Int32},
                     MF<:AbstractMatrix{T},
                     MI<:AbstractMatrix{Int32}}
    node_bounds_min::MF    # (NDIMS, n_nodes)
    node_bounds_max::MF    # (NDIMS, n_nodes)
    node_first::VI         # inclusive first idx into permutation
    node_last::VI          # inclusive last idx
    node_children::MI      # (2^NDIMS, n_nodes), children[1,i]==-1 marks leaf
    n_nodes::Int32
end

function Octree{T,NDIMS}(backend_like::AbstractArray) where {T,NDIMS}
    # Empty skeleton; grown on first build. Children dimension is 2^NDIMS:
    # 4 for a quadtree, 8 for an octree.
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

# Julia-only extension on top of the paper's API: a flat-COO edge representation
# for downstream consumers (e.g. graph neural networks). Not part of the upstream
# TreeNSearch interface. `senders[k]` is the target index j; `receivers[k]` is
# the query index i. `rel_displacement` is the normalized
# (coords_q[:,i] - coords_t[:,j]) / radius (query minus target). `rel_dist_norm`
# is ‖coords_q[:,i] - coords_t[:,j]‖ / radius.
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

# Main handle. `dev ∈ (:cpu, :cuda)` is inferred from the first
# `add_point_set!` call.
mutable struct TNS{T,NDIMS}
    dev::Symbol
    radius::T
    cell_size::T
    target_leaf_size::Int32
    refit_mode::Bool
    rebuild_threshold::Float32

    point_sets::Vector
    active_pairs::Vector{Tuple{Int32,Int32}}
    trees::Vector

    morton_codes::Vector
    permutation::Vector

    neighbor_buffers::Vector{NeighborBuffer}
    edge_buffers::Vector{EdgeBuffer}
    build_scratch::Vector{Int32}
    # One traversal stack per thread so parallel callers don't race. Indexed
    # by `Threads.threadid()`; sized at construction to `Threads.nthreads()`.
    query_stacks::Vector{Vector{Int32}}

    dirty::Vector{Bool}
    origin::Vector
end

function TNS(::Type{T}=Float32; ndims::Int=3) where {T<:AbstractFloat}
    # Size to maxthreadid so Julia 1.12+'s dynamic thread migration can't
    # index out of bounds. Each stack is 256 bytes × threads — trivial.
    nstacks = max(Threads.maxthreadid(), Threads.nthreads())
    stacks = [zeros(Int32, 64) for _ in 1:nstacks]
    TNS{T,ndims}(
        :uninitialized,
        zero(T), zero(T), Int32(32), false, 0.1f0,
        Any[], Tuple{Int32,Int32}[], Any[],
        Any[], Any[],
        NeighborBuffer[], EdgeBuffer[], Int32[], stacks,
        Bool[], Any[],
    )
end

# ---- Adapt plumbing (for users who want to move a TNS across devices by hand).
# This is a best-effort recursive adapt; device-side access uses `device_view`
# for isbits-safe kernel entry instead.
import Adapt

Adapt.@adapt_structure PointSet
Adapt.@adapt_structure Octree
Adapt.@adapt_structure NeighborBuffer
Adapt.@adapt_structure EdgeBuffer
