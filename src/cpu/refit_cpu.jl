#
# Copyright (c) 2026 Josef Jouaux
# Copyright (c) 2022-present, CompactNSearch contributors
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# Phase 5: bottom-up AABB fit.
#
# Invariant from build: node ids are emitted in BFS order, so any child has a
# strictly greater id than its parent. Processing ids in reverse order
# therefore guarantees children are finalized before parents, without needing
# a second "level" array.
#
# For leaves (children[1,i] == -1) we fold over the particle range; for
# internal nodes we fold over non-sentinel children.
#
# NDIMS-parametric: the per-axis fold and per-child loop bounds are
# constant-folded when NDIMS is known concretely (always the case in
# practice, since the Octree carries NDIMS as a type parameter).

function refit_bounds_cpu!(
    tree::Octree{T,NDIMS},
    coords::AbstractMatrix{T},
    perm::Vector{Int32},
) where {T,NDIMS}
    n_nodes = tree.n_nodes
    n_nodes == 0 && return tree

    NCH = 1 << NDIMS
    bmin = tree.node_bounds_min
    bmax = tree.node_bounds_max
    kids = tree.node_children
    first_arr = tree.node_first
    last_arr = tree.node_last

    INF = typemax(T)
    NEG = typemin(T)

    @inbounds for nid in Int32(n_nodes):-Int32(1):Int32(1)
        if kids[1, nid] == Int32(-1)
            # Leaf: fold over particle coords in the permuted range.
            f = first_arr[nid]; l = last_arr[nid]
            for d in 1:NDIMS
                lo = INF; hi = NEG
                for k in f:l
                    p = perm[k]
                    x = coords[d, p]
                    lo = min(lo, x)
                    hi = max(hi, x)
                end
                bmin[d, nid] = lo
                bmax[d, nid] = hi
            end
        else
            # Internal: union of non-empty children.
            for d in 1:NDIMS
                lo = INF; hi = NEG
                for c in 1:NCH
                    cid = kids[c, nid]
                    cid == Int32(-1) && continue
                    lo = min(lo, bmin[d, cid])
                    hi = max(hi, bmax[d, cid])
                end
                bmin[d, nid] = lo
                bmax[d, nid] = hi
            end
        end
    end
    return tree
end
