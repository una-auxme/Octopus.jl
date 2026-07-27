#
# Copyright (c) 2026 Josef Jouaux, Chair of Mechatronics, University of Augsburg
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

function build_edges_cpu!(
    buf::EdgeBuffer{T},
    tree::Octree{T,3},
    perm_t::Vector{Int32},
    coords_q::AbstractMatrix{T},
    coords_t::AbstractMatrix{T},
    stack::Vector{Int32},
    counts::Vector{Int32},
    r::T,
    exclude_self::Bool,
) where {T}
    n_q = size(coords_q, 2)
    r_sq = r * r
    inv_r = inv(r)

    # Pass A — counts. Reused scratch buffer (offsets are not stored on
    # EdgeBuffer); the rrule snapshots the output arrays, so this transient
    # may be mutated/reused across calls without affecting gradients.
    ensure_capacity!(counts, n_q + 1)
    resize!(counts, n_q + 1)
    @inbounds counts[1] = Int32(0)

    # `cref`/`cur` are reused Refs, not closure-captured plain locals: a
    # reassigned captured local boxes as Core.Box{Any} and re-allocates on
    # every increment inside the leaf loop (one heap Int per edge). A typed
    # Ref{Int32} stores in place — same inner-loop cost, zero allocation.
    cref = Ref{Int32}(0)
    @inbounds for i in 1:n_q
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

    # Exclusive scan in place: counts[i+1] becomes the edge offset of point i+1.
    @inbounds for i in 1:n_q
        counts[i + 1] += counts[i]
    end
    n_edges = @inbounds counts[n_q + 1]

    resize!(buf.senders, n_edges)
    resize!(buf.receivers, n_edges)
    if size(buf.rel_displacement, 2) != n_edges
        buf.rel_displacement = similar(buf.rel_displacement, T, (3, n_edges))
    end
    if size(buf.rel_dist_norm, 2) != n_edges
        buf.rel_dist_norm = similar(buf.rel_dist_norm, T, (1, n_edges))
    end
    buf.n_edges = n_edges

    # Pass B — fill. Each query point writes a disjoint slice [counts[i]+1 .. counts[i+1]].
    senders   = buf.senders
    receivers = buf.receivers
    rdisp     = buf.rel_displacement
    rdist     = buf.rel_dist_norm

    cur = Ref{Int32}(0)
    @inbounds for i in 1:n_q
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

    buf.materialized = true
    return buf
end

function build_edges_cpu!(
    buf::EdgeBuffer{T},
    tree::Octree{T,2},
    perm_t::Vector{Int32},
    coords_q::AbstractMatrix{T},
    coords_t::AbstractMatrix{T},
    stack::Vector{Int32},
    counts::Vector{Int32},
    r::T,
    exclude_self::Bool,
) where {T}
    n_q = size(coords_q, 2)
    r_sq = r * r
    inv_r = inv(r)

    ensure_capacity!(counts, n_q + 1)
    resize!(counts, n_q + 1)
    @inbounds counts[1] = Int32(0)

    # See the 3D path: typed Refs avoid Core.Box{Any} per-edge re-allocation.
    cref = Ref{Int32}(0)
    @inbounds for i in 1:n_q
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

    @inbounds for i in 1:n_q
        counts[i + 1] += counts[i]
    end
    n_edges = @inbounds counts[n_q + 1]

    resize!(buf.senders, n_edges)
    resize!(buf.receivers, n_edges)
    if size(buf.rel_displacement, 2) != n_edges
        buf.rel_displacement = similar(buf.rel_displacement, T, (2, n_edges))
    end
    if size(buf.rel_dist_norm, 2) != n_edges
        buf.rel_dist_norm = similar(buf.rel_dist_norm, T, (1, n_edges))
    end
    buf.n_edges = n_edges

    senders   = buf.senders
    receivers = buf.receivers
    rdisp     = buf.rel_displacement
    rdist     = buf.rel_dist_norm

    cur = Ref{Int32}(0)
    @inbounds for i in 1:n_q
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

    buf.materialized = true
    return buf
end
