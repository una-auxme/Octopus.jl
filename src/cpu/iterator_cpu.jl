#
# Copyright (c) 2026 Josef Jouaux
# Copyright (c) 2022-present, CompactNSearch contributors
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# Host-side lazy iteration. Zero allocation per call (stack-based traversal
# using an MVector). The inner distance loop is the paper's "over-approximate
# and just SIMD-crunch the distances" trick — branchless, cache-hot.
#
# 2D and 3D paths are split because their callbacks have different arities
# (the 3D form passes `dz`). Dispatch on `Octree{T,NDIMS}` picks the right
# kernel; callers hand a closure of the matching arity.

# Grow the per-thread stack vector lazily on first use by a new thread id.
# Protected by a lock so concurrent growths can't corrupt the backing array.
# Each stack is sized to `_max_stack_depth`, the proven worst-case DFS frontier,
# so the `@inbounds` pushes below can never run off the end.
const _QUERY_STACKS_LOCK = ReentrantLock()
function _grow_query_stacks!(tns::TNS{T,NDIMS}, tid::Int) where {T,NDIMS}
    depth = _max_stack_depth(Val(NDIMS))
    lock(_QUERY_STACKS_LOCK) do
        while length(tns.query_stacks) < tid
            push!(tns.query_stacks, zeros(Int32, depth))
        end
    end
    return nothing
end

# Squared distance from point to AABB (3D).
@inline function _aabb_dist_sq(
    px::T, py::T, pz::T,
    lo1::T, lo2::T, lo3::T,
    hi1::T, hi2::T, hi3::T,
)::T where {T}
    dx = max(lo1 - px, zero(T)); dx = max(dx, px - hi1)
    dy = max(lo2 - py, zero(T)); dy = max(dy, py - hi2)
    dz = max(lo3 - pz, zero(T)); dz = max(dz, pz - hi3)
    return dx * dx + dy * dy + dz * dz
end

# Squared distance from point to AABB (2D).
@inline function _aabb_dist_sq(
    px::T, py::T,
    lo1::T, lo2::T,
    hi1::T, hi2::T,
)::T where {T}
    dx = max(lo1 - px, zero(T)); dx = max(dx, px - hi1)
    dy = max(lo2 - py, zero(T)); dy = max(dy, py - hi2)
    return dx * dx + dy * dy
end

# 3D traverse. Callback shape: f(j, dx, dy, dz, d2).
@inline function _traverse_with_diff!(
    f::F,
    tree::Octree{T,3},
    target_coords::AbstractMatrix{T},
    perm::AbstractVector{Int32},
    stack::AbstractVector{Int32},
    px::T, py::T, pz::T,
    r_sq::T,
) where {F,T}
    tree.n_nodes == 0 && return nothing

    sp = 1
    @inbounds stack[sp] = Int32(1)  # root

    bmin = tree.node_bounds_min
    bmax = tree.node_bounds_max
    kids = tree.node_children
    first_arr = tree.node_first
    last_arr = tree.node_last

    while sp > 0
        nid = @inbounds stack[sp]
        sp -= 1

        @inbounds begin
            dsq = _aabb_dist_sq(
                px, py, pz,
                bmin[1, nid], bmin[2, nid], bmin[3, nid],
                bmax[1, nid], bmax[2, nid], bmax[3, nid],
            )
        end
        dsq > r_sq && continue

        if @inbounds(kids[1, nid]) == Int32(-1)
            # Leaf: flat-loop over particles in the permuted range.
            f_idx = @inbounds first_arr[nid]
            l_idx = @inbounds last_arr[nid]
            @inbounds for k in f_idx:l_idx
                j = perm[k]
                qx = target_coords[1, j]
                qy = target_coords[2, j]
                qz = target_coords[3, j]
                dx = qx - px; dy = qy - py; dz = qz - pz
                d2 = dx * dx + dy * dy + dz * dz
                if d2 <= r_sq
                    f(j, dx, dy, dz, d2)
                end
            end
        else
            @inbounds for c in 1:8
                cid = kids[c, nid]
                if cid != Int32(-1)
                    cdsq = _aabb_dist_sq(
                        px, py, pz,
                        bmin[1, cid], bmin[2, cid], bmin[3, cid],
                        bmax[1, cid], bmax[2, cid], bmax[3, cid],
                    )
                    if cdsq <= r_sq
                        sp += 1
                        stack[sp] = cid
                    end
                end
            end
        end
    end
    return nothing
end

# 2D traverse. Callback shape: f(j, dx, dy, d2).
@inline function _traverse_with_diff!(
    f::F,
    tree::Octree{T,2},
    target_coords::AbstractMatrix{T},
    perm::AbstractVector{Int32},
    stack::AbstractVector{Int32},
    px::T, py::T,
    r_sq::T,
) where {F,T}
    tree.n_nodes == 0 && return nothing

    sp = 1
    @inbounds stack[sp] = Int32(1)  # root

    bmin = tree.node_bounds_min
    bmax = tree.node_bounds_max
    kids = tree.node_children
    first_arr = tree.node_first
    last_arr = tree.node_last

    while sp > 0
        nid = @inbounds stack[sp]
        sp -= 1

        @inbounds begin
            dsq = _aabb_dist_sq(
                px, py,
                bmin[1, nid], bmin[2, nid],
                bmax[1, nid], bmax[2, nid],
            )
        end
        dsq > r_sq && continue

        if @inbounds(kids[1, nid]) == Int32(-1)
            f_idx = @inbounds first_arr[nid]
            l_idx = @inbounds last_arr[nid]
            @inbounds for k in f_idx:l_idx
                j = perm[k]
                qx = target_coords[1, j]
                qy = target_coords[2, j]
                dx = qx - px; dy = qy - py
                d2 = dx * dx + dy * dy
                if d2 <= r_sq
                    f(j, dx, dy, d2)
                end
            end
        else
            @inbounds for c in 1:4
                cid = kids[c, nid]
                if cid != Int32(-1)
                    cdsq = _aabb_dist_sq(
                        px, py,
                        bmin[1, cid], bmin[2, cid],
                        bmax[1, cid], bmax[2, cid],
                    )
                    if cdsq <= r_sq
                        sp += 1
                        stack[sp] = cid
                    end
                end
            end
        end
    end
    return nothing
end

# Index-only wrappers (3D and 2D).
@inline function _traverse!(
    f::F,
    tree::Octree{T,3},
    target_coords::AbstractMatrix{T},
    perm::AbstractVector{Int32},
    stack::AbstractVector{Int32},
    px::T, py::T, pz::T,
    r_sq::T,
) where {F,T}
    _traverse_with_diff!(tree, target_coords, perm, stack, px, py, pz, r_sq) do j, _, _, _, _
        f(j)
    end
end

@inline function _traverse!(
    f::F,
    tree::Octree{T,2},
    target_coords::AbstractMatrix{T},
    perm::AbstractVector{Int32},
    stack::AbstractVector{Int32},
    px::T, py::T,
    r_sq::T,
) where {F,T}
    _traverse_with_diff!(tree, target_coords, perm, stack, px, py, r_sq) do j, _, _, _
        f(j)
    end
end

# Public: iterate neighbors of point `i` in querying set qid, found in target
# set tid, calling `f(j)` for each matching target index.
"""
    for_each_neighbor(f, tns, qid, tid, i)

Call `f(j)` for every point `j` in target set `tid` lying within `radius` of
point `i` in query set `qid`. This is the zero-allocation iteration primitive —
neighbors are visited during traversal rather than collected, so nothing lands
on the heap in the hot loop.

Intended for `do`-block syntax:

```julia
for_each_neighbor(tns, qid, tid, i) do j
    # use neighbor j
end
```

Run `run!(tns)` first. Self-pairs are *not* filtered: when `qid == tid` the
callback also fires for `j == i`, so skip it yourself if unwanted (this is
what `get_neighborlist` does).

Thread-safe — each caller gets its own traversal stack, indexed by thread id.

Host-side and CPU-only; on a CUDA `TNS` use `for_each_neighbor_device` inside
your own kernel, or `build_edges` for a materialized edge list.
"""
function for_each_neighbor(f::F, tns::TNS{T,NDIMS}, qid::Integer, tid::Integer, i::Integer) where {F,T,NDIMS}
    tns.dev === :cpu || error("for_each_neighbor (host) requires a CPU TNS; use for_each_neighbor_device on GPU")
    # Per-thread stack so concurrent callers don't race on one shared buffer.
    # Lock-free resize if a new thread id exceeds our prealloc (happens in
    # Julia 1.12+ with dynamic thread spawn).
    thread_id = Threads.threadid()
    if thread_id > length(tns.query_stacks)
        _grow_query_stacks!(tns, thread_id)
    end
    stack = @inbounds tns.query_stacks[thread_id]
    # Function barrier: extraction from untyped Vectors is dynamic; the
    # barrier specializes on concrete PointSet/Octree types so the hot path
    # is fully type-inferred and allocation-free.
    _for_each_neighbor_barrier(f,
        tns.point_sets[qid], tns.point_sets[tid],
        tns.trees[tid], tns.permutation[tid],
        stack,
        tns.radius, Int(i))
end

@inline function _for_each_neighbor_barrier(
    f::F,
    qs::PointSet{T,3,A1},
    ts::PointSet{T,3,A2},
    tree::Octree{T,3},
    perm::AbstractVector{Int32},
    stack::AbstractVector{Int32},
    r::T, i::Int,
) where {F,T,A1,A2}
    coords_q = qs.coords
    coords_t = ts.coords
    r_sq = r * r
    px = @inbounds coords_q[1, i]
    py = @inbounds coords_q[2, i]
    pz = @inbounds coords_q[3, i]
    _traverse!(f, tree, coords_t, perm, stack, px, py, pz, r_sq)
    return nothing
end

@inline function _for_each_neighbor_barrier(
    f::F,
    qs::PointSet{T,2,A1},
    ts::PointSet{T,2,A2},
    tree::Octree{T,2},
    perm::AbstractVector{Int32},
    stack::AbstractVector{Int32},
    r::T, i::Int,
) where {F,T,A1,A2}
    coords_q = qs.coords
    coords_t = ts.coords
    r_sq = r * r
    px = @inbounds coords_q[1, i]
    py = @inbounds coords_q[2, i]
    _traverse!(f, tree, coords_t, perm, stack, px, py, r_sq)
    return nothing
end
