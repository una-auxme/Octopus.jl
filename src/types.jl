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
mutable struct TNS{T,NDIMS}
    dev::Symbol
    radius::T
    target_leaf_size::Int32
    refit_mode::Bool

    point_sets::Vector
    active_pairs::Vector{Tuple{Int32,Int32}}
    trees::Vector

    morton_codes::Vector
    permutation::Vector

    neighbor_buffers::Vector{NeighborBuffer}
    edge_buffers::Vector{EdgeBuffer}
    build_scratch::Vector{Int32}          # reused Pass-A scratch for build_edges_cpu!
    query_stacks::Vector{Vector{Int32}}   # one stack per thread, indexed by threadid()

    dirty::Vector{Bool}
end

function TNS(::Type{T}=Float32; ndims::Int=3) where {T<:AbstractFloat}
    # Size to maxthreadid so Julia 1.12+ dynamic thread migration can't index OOB.
    nstacks = max(Threads.maxthreadid(), Threads.nthreads())
    stacks = [zeros(Int32, 64) for _ in 1:nstacks]
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
