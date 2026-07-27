#
# Copyright (c) 2026 Josef Jouaux, Chair of Mechatronics, University of Augsburg
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# Phase 4: parallel top-down tree build over the sorted Morton order.
#
# The tree is 2^NDIMS-way (octree for 3D, quadtree for 2D). Each node holds a
# contiguous range [first, last] into the permutation; particles in that range
# share the upper bits of the Morton code. Construction is BFS by level; one
# node is partitioned into up to NCH children by the NDIMS Morton bits at the
# current level.
#
# Parallelism: within a BFS round, node partitioning is independent per node
# (each reads a disjoint slice of the permutation). We do serial emission
# into the shared node arrays to keep indexing simple; contention would need
# atomic bump counters — easy upgrade later.

# Bits-per-axis and matching tree depth budget.
@inline _bits_per_axis(::Val{2}) = 31  # 62-bit code, top bit zero for lcp_tiebreak
@inline _bits_per_axis(::Val{3}) = 21  # 63-bit code, top bit zero
@inline _max_tree_levels(v::Val) = _bits_per_axis(v)

# Bit position of the lowest of the NDIMS Morton bits at level L (L=0 = root split).
@inline function _level_shift(L::Integer, ::Val{NDIMS}) where {NDIMS}
    bits = _bits_per_axis(Val(NDIMS))
    return NDIMS * bits - NDIMS * (Int(L) + 1)
end

@inline _octant_mask(::Val{NDIMS}) where {NDIMS} = UInt64((1 << NDIMS) - 1)

# Partition the parent's permutation slice into up to NCH=2^NDIMS octants.
# Caller passes MVectors sized to NCH and a Val{NDIMS}.
@inline function _partition_octants!(
    oc_first::MVector{NCH,Int32},
    oc_last::MVector{NCH,Int32},
    morton_codes::Vector{UInt64},
    perm::Vector{Int32},
    first::Int32, last::Int32, L::Integer,
    ::Val{NDIMS},
) where {NCH,NDIMS}
    @inbounds for o in 1:NCH
        oc_first[o] = Int32(-1)
        oc_last[o] = Int32(-2)
    end
    shift = _level_shift(L, Val(NDIMS))
    mask  = _octant_mask(Val(NDIMS))
    @inbounds begin
        prev_o = Int32(-1)
        for i in first:last
            o = Int32((morton_codes[perm[i]] >> shift) & mask) + Int32(1)
            if o != prev_o
                prev_o != Int32(-1) && (oc_last[prev_o] = i - Int32(1))
                oc_first[o] = i
                prev_o = o
            end
        end
        prev_o != Int32(-1) && (oc_last[prev_o] = last)
    end
    return nothing
end

# Small helper to grow the SoA node arrays in lockstep.
function _grow_node_arrays!(node_first, node_last, node_level, node_children, needed::Integer, ::Val{NCH}) where {NCH}
    if needed > length(node_first)
        newcap = max(Int(needed) + 16, (length(node_first) * 3) >> 1)
        resize!(node_first, newcap)
        resize!(node_last, newcap)
        resize!(node_level, newcap)
        newkids = Matrix{Int32}(undef, NCH, newcap)
        @inbounds for k in axes(node_children, 2), r in 1:NCH
            newkids[r, k] = node_children[r, k]
        end
        return (node_first, node_last, node_level, newkids)
    end
    return (node_first, node_last, node_level, node_children)
end

function build_cpu!(
    tree::Octree{T,NDIMS},
    morton_codes::Vector{UInt64},
    perm::Vector{Int32},
    target_leaf_size::Int32,
) where {T,NDIMS}
    NCH = 1 << NDIMS
    n = length(perm)
    if n == 0
        tree.n_nodes = Int32(0)
        tree.node_first = Int32[]
        tree.node_last = Int32[]
        tree.node_children = Matrix{Int32}(undef, NCH, 0)
        tree.node_bounds_min = Matrix{T}(undef, NDIMS, 0)
        tree.node_bounds_max = Matrix{T}(undef, NDIMS, 0)
        return tree
    end

    max_levels = _max_tree_levels(Val(NDIMS))

    # Safe upper bound on node count with target_leaf_size ≥ 1.
    est = max(Int32(NCH * cld(n, max(Int(target_leaf_size), 1)) + 16), Int32(16))
    node_first    = Vector{Int32}(undef, est)
    node_last     = Vector{Int32}(undef, est)
    node_level    = Vector{Int32}(undef, est)
    node_children = Matrix{Int32}(undef, NCH, est)

    # Root
    node_first[1] = Int32(1)
    node_last[1]  = Int32(n)
    node_level[1] = Int32(0)
    n_nodes = Int32(1)
    next_open = Int32(1)

    oc_first = MVector{NCH,Int32}(undef)
    oc_last  = MVector{NCH,Int32}(undef)

    while next_open <= n_nodes
        frontier_end = n_nodes

        for nid in next_open:frontier_end
            f = node_first[nid]
            l = node_last[nid]
            lvl = node_level[nid]
            count = l - f + Int32(1)

            if count <= target_leaf_size || lvl >= Int32(max_levels)
                @inbounds for o in 1:NCH
                    node_children[o, nid] = Int32(-1)
                end
                continue
            end

            _partition_octants!(oc_first, oc_last, morton_codes, perm, f, l, lvl, Val(NDIMS))

            # Emit one child per non-empty octant; empty octants get the -1
            # sentinel. Fully degenerate input (all particles in one octant)
            # just produces a one-child chain, level by level, until the range
            # drops to a leaf or hits max_levels.
            @inbounds for o in 1:NCH
                if oc_first[o] == Int32(-1)
                    node_children[o, nid] = Int32(-1)
                else
                    needed = n_nodes + Int32(1)
                    (node_first, node_last, node_level, node_children) =
                        _grow_node_arrays!(node_first, node_last, node_level, node_children, needed, Val(NCH))
                    cid = needed
                    node_first[cid] = oc_first[o]
                    node_last[cid]  = oc_last[o]
                    node_level[cid] = lvl + Int32(1)
                    node_children[o, nid] = cid
                    n_nodes = cid
                end
            end
        end

        next_open = frontier_end + Int32(1)
    end

    # Trim to exact size. `node_children` must be a fresh (NCH, n_nodes) matrix.
    resize!(node_first, n_nodes)
    resize!(node_last, n_nodes)
    children_out = Matrix{Int32}(undef, NCH, n_nodes)
    @inbounds for k in 1:n_nodes, r in 1:NCH
        children_out[r, k] = node_children[r, k]
    end

    tree.node_first = node_first
    tree.node_last = node_last
    tree.node_children = children_out
    tree.node_bounds_min = Matrix{T}(undef, NDIMS, n_nodes)
    tree.node_bounds_max = Matrix{T}(undef, NDIMS, n_nodes)
    tree.n_nodes = n_nodes
    return tree
end
