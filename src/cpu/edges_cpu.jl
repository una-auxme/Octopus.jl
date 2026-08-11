#
# Copyright (c) 2026 Josef Jouaux
# Copyright (c) 2022-present, CompactNSearch contributors
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# Two-pass edge construction. Pass A counts neighbors per query point; pass B
# fills flat-COO senders/receivers and per-edge displacement/distance using the
# diff already produced in the leaf inner loop. Sign convention:
#   rel_displacement[:,k] = (coords_q[:,i] - coords_t[:,j]) / r   (query - target)
#   rel_dist_norm[1,k]    = sqrt(d²) / r
# Self-pair (qid == tid) excludes j == i.
#
# 3D and 2D overloads dispatch on the tree type.
#
# Both passes are parallel over query points and run under Polyester `@batch`:
# pass A writes only `counts[i+1]`, and pass B writes only the disjoint slice
# `[counts[i]+1 .. counts[i+1]]`, so no atomics or locks are needed. The tree,
# permutation and coordinates are read-only throughout. Each chunk takes its
# own traversal stack from `stacks` (see `n_query_chunks` in util.jl); the
# exclusive scan between the passes stays serial, being O(n_q) integer adds
# against a traversal that dominates by orders of magnitude.
#
# The per-chunk work lives in named worker functions rather than inline in the
# `@batch` body: it keeps the closures in ordinary functions where the
# `Ref{Int32}` trick below still specializes, and leaves `@batch` with nothing
# but a call to distribute.

# ---------------- pass A workers (count) ----------------------------------

# `cref` is a reused typed Ref, not a closure-captured plain local: a
# reassigned captured local boxes as Core.Box{Any} and re-allocates on every
# increment inside the leaf loop (one heap Int per edge). A typed Ref{Int32}
# stores in place — same inner-loop cost, zero allocation. It is created per
# chunk, so concurrent chunks never share it.
function _count_edges_range!(
    counts::AbstractVector{Int32}, tree::Octree{T,3},
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
            _traverse_with_diff!(tree, coords_t, perm_t, stack, px, py, pz, r_sq) do j, _, _, _, _
                if Int32(j) != i32
                    cref[] += Int32(1)
                end
            end
        else
            _traverse_with_diff!(tree, coords_t, perm_t, stack, px, py, pz, r_sq) do _, _, _, _, _
                cref[] += Int32(1)
            end
        end
        counts[i + 1] = cref[]
    end
    return nothing
end

function _count_edges_range!(
    counts::AbstractVector{Int32}, tree::Octree{T,2},
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
            _traverse_with_diff!(tree, coords_t, perm_t, stack, px, py, r_sq) do j, _, _, _
                if Int32(j) != i32
                    cref[] += Int32(1)
                end
            end
        else
            _traverse_with_diff!(tree, coords_t, perm_t, stack, px, py, r_sq) do _, _, _, _
                cref[] += Int32(1)
            end
        end
        counts[i + 1] = cref[]
    end
    return nothing
end

# ---------------- pass B workers (fill) -----------------------------------

function _fill_edges_range!(
    senders::AbstractVector{Int32}, receivers::AbstractVector{Int32},
    rdisp::AbstractMatrix{T}, rdist::AbstractMatrix{T},
    counts::AbstractVector{Int32}, tree::Octree{T,3},
    coords_q::AbstractMatrix{T}, coords_t::AbstractMatrix{T},
    perm_t::AbstractVector{Int32}, stack::AbstractVector{Int32},
    lo::Int, hi::Int, r_sq::T, inv_r::T, exclude_self::Bool,
) where {T}
    cur = Ref{Int32}(0)
    @inbounds for i in lo:hi
        px = coords_q[1, i]; py = coords_q[2, i]; pz = coords_q[3, i]
        cur[] = counts[i]
        i32 = Int32(i)
        if exclude_self
            _traverse_with_diff!(tree, coords_t, perm_t, stack, px, py, pz, r_sq) do j, dx, dy, dz, d2
                if Int32(j) != i32
                    k = cur[] + Int32(1); cur[] = k
                    senders[k]   = Int32(j)
                    receivers[k] = i32
                    rdisp[1, k] = -dx * inv_r
                    rdisp[2, k] = -dy * inv_r
                    rdisp[3, k] = -dz * inv_r
                    rdist[1, k] = sqrt(d2) * inv_r
                end
            end
        else
            _traverse_with_diff!(tree, coords_t, perm_t, stack, px, py, pz, r_sq) do j, dx, dy, dz, d2
                k = cur[] + Int32(1); cur[] = k
                senders[k]   = Int32(j)
                receivers[k] = i32
                rdisp[1, k] = -dx * inv_r
                rdisp[2, k] = -dy * inv_r
                rdisp[3, k] = -dz * inv_r
                rdist[1, k] = sqrt(d2) * inv_r
            end
        end
    end
    return nothing
end

function _fill_edges_range!(
    senders::AbstractVector{Int32}, receivers::AbstractVector{Int32},
    rdisp::AbstractMatrix{T}, rdist::AbstractMatrix{T},
    counts::AbstractVector{Int32}, tree::Octree{T,2},
    coords_q::AbstractMatrix{T}, coords_t::AbstractMatrix{T},
    perm_t::AbstractVector{Int32}, stack::AbstractVector{Int32},
    lo::Int, hi::Int, r_sq::T, inv_r::T, exclude_self::Bool,
) where {T}
    cur = Ref{Int32}(0)
    @inbounds for i in lo:hi
        px = coords_q[1, i]; py = coords_q[2, i]
        cur[] = counts[i]
        i32 = Int32(i)
        if exclude_self
            _traverse_with_diff!(tree, coords_t, perm_t, stack, px, py, r_sq) do j, dx, dy, d2
                if Int32(j) != i32
                    k = cur[] + Int32(1); cur[] = k
                    senders[k]   = Int32(j)
                    receivers[k] = i32
                    rdisp[1, k] = -dx * inv_r
                    rdisp[2, k] = -dy * inv_r
                    rdist[1, k] = sqrt(d2) * inv_r
                end
            end
        else
            _traverse_with_diff!(tree, coords_t, perm_t, stack, px, py, r_sq) do j, dx, dy, d2
                k = cur[] + Int32(1); cur[] = k
                senders[k]   = Int32(j)
                receivers[k] = i32
                rdisp[1, k] = -dx * inv_r
                rdisp[2, k] = -dy * inv_r
                rdist[1, k] = sqrt(d2) * inv_r
            end
        end
    end
    return nothing
end

# ---------------- driver ---------------------------------------------------

function build_edges_cpu!(
    buf::EdgeBuffer{T},
    tree::Octree{T,NDIMS},
    perm_t::AbstractVector{Int32},
    coords_q::AbstractMatrix{T},
    coords_t::AbstractMatrix{T},
    stacks::Vector{Vector{Int32}},
    counts::AbstractVector{Int32},
    r::T,
    exclude_self::Bool,
) where {T,NDIMS}
    n_q = size(coords_q, 2)
    r_sq = r * r
    inv_r = inv(r)

    # Pass A — counts. Reused scratch buffer (offsets are not stored on
    # EdgeBuffer); the rrule snapshots the output arrays, so this transient
    # may be mutated/reused across calls without affecting gradients.
    ensure_capacity!(counts, n_q + 1)
    resize!(counts, n_q + 1)
    @inbounds counts[1] = Int32(0)

    nchunks = n_query_chunks(n_q, length(stacks))
    if nchunks == 1
        _count_edges_range!(counts, tree, coords_q, coords_t, perm_t,
                            @inbounds(stacks[1]), 1, n_q, r_sq, exclude_self)
    else
        @batch for c in 1:nchunks
            lo, hi = chunk_bounds(n_q, nchunks, c)
            _count_edges_range!(counts, tree, coords_q, coords_t, perm_t,
                                @inbounds(stacks[c]), lo, hi, r_sq, exclude_self)
        end
    end

    # Exclusive scan in place: counts[i+1] becomes the edge offset of point i+1.
    # Serial by design — O(n_q) adds next to a traversal that costs orders of
    # magnitude more, and a parallel scan would need a second pass over counts.
    @inbounds for i in 1:n_q
        counts[i + 1] += counts[i]
    end
    n_edges = @inbounds counts[n_q + 1]

    resize!(buf.senders, n_edges)
    resize!(buf.receivers, n_edges)
    if size(buf.rel_displacement, 2) != n_edges
        buf.rel_displacement = similar(buf.rel_displacement, T, (NDIMS, n_edges))
    end
    if size(buf.rel_dist_norm, 2) != n_edges
        buf.rel_dist_norm = similar(buf.rel_dist_norm, T, (1, n_edges))
    end
    buf.n_edges = n_edges

    # Pass B — fill. Each query point writes a disjoint slice
    # [counts[i]+1 .. counts[i+1]], so chunks never overlap and edge order is
    # identical to the serial build.
    senders   = buf.senders
    receivers = buf.receivers
    rdisp     = buf.rel_displacement
    rdist     = buf.rel_dist_norm

    if nchunks == 1
        _fill_edges_range!(senders, receivers, rdisp, rdist, counts, tree,
                           coords_q, coords_t, perm_t, @inbounds(stacks[1]),
                           1, n_q, r_sq, inv_r, exclude_self)
    else
        @batch for c in 1:nchunks
            lo, hi = chunk_bounds(n_q, nchunks, c)
            _fill_edges_range!(senders, receivers, rdisp, rdist, counts, tree,
                               coords_q, coords_t, perm_t, @inbounds(stacks[c]),
                               lo, hi, r_sq, inv_r, exclude_self)
        end
    end

    buf.materialized = true
    return buf
end
