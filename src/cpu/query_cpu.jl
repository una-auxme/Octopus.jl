#
# Copyright (c) 2026 Josef Jouaux
# Copyright (c) 2022-present, CompactNSearch contributors
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# Opt-in CSR materialization. Two-pass:
#   A: count neighbors per querying point -> offsets (exclusive scan)
#   B: write into pre-sized flat[] at offset + cursor
# Self-pair (qid == tid) excludes j == i.
#
# Same shape — and the same parallelism — as build_edges_cpu! next door: pass A
# writes only `offsets[i+1]`, pass B only the disjoint slice
# `[offsets[i]+1 .. offsets[i]+count_i]`. Chunks are parallel over query points
# under `@batch`, each with its own traversal stack; the scan between them
# stays serial. See util.jl for the chunking, and edges_cpu.jl for the reason
# the per-chunk work sits in named worker functions.
#
# 3D and 2D workers dispatch on the tree type.

# ---------------- pass A workers (count) ----------------------------------

# Typed Refs, one per chunk: a reassigned closure-captured local boxes as
# Core.Box{Any} and re-allocates a heap Int per neighbor in the leaf loop.
function _count_neighbors_range!(
    offsets::AbstractVector{Int32}, tree::Octree{T,3},
    coords_q::AbstractMatrix{T}, coords_t::AbstractMatrix{T},
    perm_t::AbstractVector{Int32}, stack::AbstractVector{Int32},
    lo::Int, hi::Int, r_sq::T, exclude_self::Bool,
) where {T}
    cref = Ref{Int32}(0)
    @inbounds for i in lo:hi
        px = coords_q[1, i]; py = coords_q[2, i]; pz = coords_q[3, i]
        cref[] = Int32(0)
        i32 = Int32(i)
        if exclude_self
            _traverse!(tree, coords_t, perm_t, stack, px, py, pz, r_sq) do j
                if Int32(j) != i32
                    cref[] += Int32(1)
                end
            end
        else
            _traverse!(tree, coords_t, perm_t, stack, px, py, pz, r_sq) do _
                cref[] += Int32(1)
            end
        end
        offsets[i + 1] = cref[]
    end
    return nothing
end

function _count_neighbors_range!(
    offsets::AbstractVector{Int32}, tree::Octree{T,2},
    coords_q::AbstractMatrix{T}, coords_t::AbstractMatrix{T},
    perm_t::AbstractVector{Int32}, stack::AbstractVector{Int32},
    lo::Int, hi::Int, r_sq::T, exclude_self::Bool,
) where {T}
    cref = Ref{Int32}(0)
    @inbounds for i in lo:hi
        px = coords_q[1, i]; py = coords_q[2, i]
        cref[] = Int32(0)
        i32 = Int32(i)
        if exclude_self
            _traverse!(tree, coords_t, perm_t, stack, px, py, r_sq) do j
                if Int32(j) != i32
                    cref[] += Int32(1)
                end
            end
        else
            _traverse!(tree, coords_t, perm_t, stack, px, py, r_sq) do _
                cref[] += Int32(1)
            end
        end
        offsets[i + 1] = cref[]
    end
    return nothing
end

# ---------------- pass B workers (write) ----------------------------------

function _write_neighbors_range!(
    flat::AbstractVector{Int32}, offsets::AbstractVector{Int32}, tree::Octree{T,3},
    coords_q::AbstractMatrix{T}, coords_t::AbstractMatrix{T},
    perm_t::AbstractVector{Int32}, stack::AbstractVector{Int32},
    lo::Int, hi::Int, r_sq::T, exclude_self::Bool,
) where {T}
    cur = Ref{Int32}(0)
    @inbounds for i in lo:hi
        px = coords_q[1, i]; py = coords_q[2, i]; pz = coords_q[3, i]
        cur[] = offsets[i]
        i32 = Int32(i)
        if exclude_self
            _traverse!(tree, coords_t, perm_t, stack, px, py, pz, r_sq) do j
                if Int32(j) != i32
                    k = cur[] + Int32(1); cur[] = k
                    flat[k] = Int32(j)
                end
            end
        else
            _traverse!(tree, coords_t, perm_t, stack, px, py, pz, r_sq) do j
                k = cur[] + Int32(1); cur[] = k
                flat[k] = Int32(j)
            end
        end
    end
    return nothing
end

function _write_neighbors_range!(
    flat::AbstractVector{Int32}, offsets::AbstractVector{Int32}, tree::Octree{T,2},
    coords_q::AbstractMatrix{T}, coords_t::AbstractMatrix{T},
    perm_t::AbstractVector{Int32}, stack::AbstractVector{Int32},
    lo::Int, hi::Int, r_sq::T, exclude_self::Bool,
) where {T}
    cur = Ref{Int32}(0)
    @inbounds for i in lo:hi
        px = coords_q[1, i]; py = coords_q[2, i]
        cur[] = offsets[i]
        i32 = Int32(i)
        if exclude_self
            _traverse!(tree, coords_t, perm_t, stack, px, py, r_sq) do j
                if Int32(j) != i32
                    k = cur[] + Int32(1); cur[] = k
                    flat[k] = Int32(j)
                end
            end
        else
            _traverse!(tree, coords_t, perm_t, stack, px, py, r_sq) do j
                k = cur[] + Int32(1); cur[] = k
                flat[k] = Int32(j)
            end
        end
    end
    return nothing
end

# ---------------- driver ---------------------------------------------------

function materialize_cpu!(
    buf::NeighborBuffer,
    tree::Octree{T,NDIMS},
    perm_t::AbstractVector{Int32},
    coords_q::AbstractMatrix{T},
    coords_t::AbstractMatrix{T},
    stacks::Vector{Vector{Int32}},
    r::T,
    exclude_self::Bool,
) where {T,NDIMS}
    n_q = size(coords_q, 2)
    r_sq = r * r

    # Pass A — counts
    resize!(buf.offsets, n_q + 1)
    fill!(buf.offsets, Int32(0))
    offsets = buf.offsets

    nchunks = n_query_chunks(n_q, length(stacks))
    if nchunks == 1
        _count_neighbors_range!(offsets, tree, coords_q, coords_t, perm_t,
                                @inbounds(stacks[1]), 1, n_q, r_sq, exclude_self)
    else
        @batch for c in 1:nchunks
            lo, hi = chunk_bounds(n_q, nchunks, c)
            _count_neighbors_range!(offsets, tree, coords_q, coords_t, perm_t,
                                    @inbounds(stacks[c]), lo, hi, r_sq, exclude_self)
        end
    end

    # Exclusive scan -> CSR row offsets starting at 0.
    @inbounds for i in 1:n_q
        offsets[i + 1] += offsets[i]
    end

    total = @inbounds offsets[n_q + 1]
    resize!(buf.flat, total)
    flat = buf.flat

    # Pass B — writes into disjoint per-point slices.
    if nchunks == 1
        _write_neighbors_range!(flat, offsets, tree, coords_q, coords_t, perm_t,
                                @inbounds(stacks[1]), 1, n_q, r_sq, exclude_self)
    else
        @batch for c in 1:nchunks
            lo, hi = chunk_bounds(n_q, nchunks, c)
            _write_neighbors_range!(flat, offsets, tree, coords_q, coords_t, perm_t,
                                    @inbounds(stacks[c]), lo, hi, r_sq, exclude_self)
        end
    end

    buf.materialized = true
    return buf
end
