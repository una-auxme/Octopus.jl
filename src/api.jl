# Public API. Orchestrates the CPU phase sequence; dispatches to the CUDA
# extension via hooks declared in TreeNSearch.jl when the user passes GPU
# coords.

using StaticArrays

# ---------------- configuration -------------------------------------------

function set_search_radius!(tns::TNS{T,NDIMS}, r::Real) where {T,NDIMS}
    r > 0 || throw(ArgumentError("search radius must be positive"))
    tns.radius = T(r)
    tns.cell_size = T(r)
    _mark_all_dirty!(tns)
    return tns
end

function set_refit_mode!(tns::TNS, on::Bool)
    tns.refit_mode = on
    _mark_all_dirty!(tns)
    return tns
end

function _mark_all_dirty!(tns::TNS)
    @inbounds for i in eachindex(tns.dirty)
        tns.dirty[i] = true
    end
    return nothing
end

# ---------------- point set registration ----------------------------------

function _infer_dev!(tns::TNS, coords::AbstractMatrix)
    new_dev = coords isa AbstractGPUArray ? :cuda : :cpu
    if tns.dev === :uninitialized
        tns.dev = new_dev
    elseif tns.dev !== new_dev
        throw(ArgumentError(
            "TNS was initialized on :$(tns.dev) but got coords on :$new_dev; " *
            "all point sets must share a backend"))
    end
    if new_dev === :cuda && !isdefined(Main, :CUDA) && !_cuda_ext_loaded()
        throw(ArgumentError(
            "GPU coords detected; load CUDA.jl (`using CUDA`) to enable the CUDA backend"))
    end
    return new_dev
end

# Flipped by the CUDA ext's __init__. Using a Ref avoids the
# method-overwrite-during-precompile error.
const _CUDA_EXT_LOADED = Ref(false)
_cuda_ext_loaded() = _CUDA_EXT_LOADED[]

function add_point_set!(tns::TNS{T,NDIMS}, coords::AbstractMatrix{T}) where {T,NDIMS}
    size(coords, 1) == NDIMS || throw(DimensionMismatch("coords must be ($NDIMS, N)"))
    _infer_dev!(tns, coords)

    push!(tns.point_sets, PointSet(coords))
    push!(tns.morton_codes, UInt64[])
    push!(tns.permutation, Int32[])
    push!(tns.trees, Octree{T,NDIMS}(coords))
    push!(tns.dirty, true)
    push!(tns.origin, zero(SVector{NDIMS,T}))
    return length(tns.point_sets)  # set id
end

function resize_point_set!(tns::TNS{T,NDIMS}, set_id::Integer, coords::AbstractMatrix{T}) where {T,NDIMS}
    size(coords, 1) == NDIMS || throw(DimensionMismatch("coords must be ($NDIMS, N)"))
    tns.point_sets[set_id] = PointSet(coords)
    tns.dirty[set_id] = true
    return tns
end

"""
    update_point_set!(tns, set_id, coords) -> tns

Julia-only convenience over `resize_point_set!`: refresh the coordinates of
a point set under a strict same-`N` precondition. Throws `DimensionMismatch`
if `N` changes — callers who genuinely need a different `N` should use
`resize_point_set!` instead. Not part of the upstream C++ TreeNSearch API.
"""
function update_point_set!(tns::TNS{T,NDIMS}, set_id::Integer, coords::AbstractMatrix{T}) where {T,NDIMS}
    size(coords, 1) == NDIMS || throw(DimensionMismatch("coords must be ($NDIMS, N)"))
    ps = tns.point_sets[set_id]
    size(coords, 2) == ps.n ||
        throw(DimensionMismatch("update_point_set! expects same size; use resize_point_set! otherwise"))
    tns.point_sets[set_id] = PointSet(coords)
    tns.dirty[set_id] = true
    return tns
end

# ---------------- active-search registration ------------------------------

"""
    set_active_search!(tns, qid, tid)

Register the directed pair (query=qid, target=tid) as an active search. A
fresh neighbor buffer and edge buffer are attached on first registration.
Calling with an already-registered pair is a **no-op** — the existing
buffers are kept and will be refilled by the next `run!` /
`materialize_all_neighbors!` / `build_edges!` call. There is no need (or
way) to "reset" a pair through this function.
"""
function set_active_search!(tns::TNS{T,NDIMS}, q::Integer, t::Integer) where {T,NDIMS}
    pair = (Int32(q), Int32(t))
    pair in tns.active_pairs && return tns
    push!(tns.active_pairs, pair)
    # Attach fresh lazy buffers (empty until materialized / built).
    coords_like = tns.point_sets[q].coords
    push!(tns.neighbor_buffers, NeighborBuffer(coords_like))
    push!(tns.edge_buffers, make_edge_buffer(T, NDIMS, coords_like))
    return tns
end

# ---------------- run ------------------------------------------------------

function run!(tns::TNS{T,NDIMS}) where {T,NDIMS}
    tns.radius > 0 || throw(ArgumentError("set_search_radius! before run!"))
    if tns.dev === :cpu
        return _run_cpu!(tns)
    elseif tns.dev === :cuda
        return _run_cuda!(tns)
    else
        throw(ArgumentError("add at least one point set before run!"))
    end
end

function _run_cpu!(tns::TNS{T,NDIMS}) where {T,NDIMS}
    # For each dirty set: bin, sort, RLE, build, refit bounds.
    @inbounds for sid in eachindex(tns.point_sets)
        tns.dirty[sid] || continue

        ps = tns.point_sets[sid]
        coords = ps.coords
        n = Int(ps.n)

        # Empty point set: leave the tree empty (n_nodes==0). Traversal
        # short-circuits on empty trees so all queries return zero neighbors.
        if n == 0
            empty!(tns.morton_codes[sid])
            empty!(tns.permutation[sid])
            tns.trees[sid].n_nodes = Int32(0)
            tns.dirty[sid] = false
            continue
        end

        # origin per set (stable across refits)
        origin = _point_origin(coords)
        tns.origin[sid] = origin

        morton = tns.morton_codes[sid]
        ensure_capacity!(morton, n)
        resize!(morton, n)
        bin_cpu!(morton, coords, origin, tns.cell_size)

        perm = tns.permutation[sid]
        ensure_capacity!(perm, n)
        resize!(perm, n)
        sort_by_key_cpu!(perm, morton)

        # RLE cells (scratch; not retained in tns state)
        cell_morton = UInt64[]
        cell_first = Int32[]
        cell_last = Int32[]
        rle_cells_cpu!(cell_morton, cell_first, cell_last, morton, perm)

        tree = tns.trees[sid]
        build_cpu!(tree, morton, perm, tns.target_leaf_size)
        refit_bounds_cpu!(tree, coords, perm)

        tns.dirty[sid] = false
    end

    # Drop Morton-code scratch when refit mode is off: they're only useful for
    # detecting cell crossings on the next update. Keeps steady-state memory
    # minimal.
    if !tns.refit_mode
        for mc in tns.morton_codes
            empty!(mc); sizehint!(mc, 0)
        end
    end

    # Neighbor buffers are lazy: nothing to do unless the user asked to
    # materialize a pair. Invalidate any previously materialized lists so
    # get_neighborlist on stale data can't silently succeed.
    for buf in tns.neighbor_buffers
        buf.materialized = false
    end
    for buf in tns.edge_buffers
        buf.materialized = false
    end
    return tns
end

# ---------------- per-point list access -----------------------------------

"""
    get_neighborlist(tns, qid, tid, i) -> Vector{Int32}

Collect the indices of all points in set `tid` that lie within `radius` of
point `i` in set `qid`. Allocates a fresh Vector each call — use
`for_each_neighbor` for the zero-allocation path.
"""
function get_neighborlist(tns::TNS{T,NDIMS}, qid::Integer, tid::Integer, i::Integer) where {T,NDIMS}
    out = Int32[]
    exclude_self = (Int32(qid) == Int32(tid))
    i32 = Int32(i)
    for_each_neighbor(tns, qid, tid, i) do j
        if !(exclude_self && j == i32)
            push!(out, j)
        end
    end
    return out
end

"""
    build_edges!(tns, qid, tid) -> EdgeBuffer

Populate the `EdgeBuffer` for the active pair `(qid, tid)` with flat-COO
edge data: `senders`, `receivers`, `rel_displacement` (normalized
`coords_q[:,i] - coords_t[:,j]`) and `rel_dist_norm` (normalized distance).
Self-pair (qid == tid) excludes j == i.

Run `set_active_search!(tns, qid, tid)` and `run!(tns)` first.
"""
function build_edges!(tns::TNS{T,NDIMS}, qid::Integer, tid::Integer) where {T,NDIMS}
    pair_idx = findfirst(==((Int32(qid), Int32(tid))), tns.active_pairs)
    pair_idx === nothing && error("build_edges!: pair ($qid, $tid) not registered; call set_active_search! first")
    buf = tns.edge_buffers[pair_idx]
    if tns.dev === :cpu
        build_edges_cpu!(
            buf, tns.trees[tid], tns.permutation[tid],
            tns.point_sets[qid].coords, tns.point_sets[tid].coords,
            tns.query_stacks[1], tns.radius, qid == tid,
        )
    elseif tns.dev === :cuda
        _build_edges_cuda!(tns, pair_idx, qid, tid)
    else
        error("build_edges!: TNS not initialized; add a point set first")
    end
    return buf
end

"""
    build_edges(tns, qid, tid) -> NamedTuple

Convenience wrapper that returns `(senders, receivers, rel_displacement,
rel_dist_norm)` as a NamedTuple. Same backing storage as the underlying
`EdgeBuffer`, so the arrays alias `tns.edge_buffers[…]`.
"""
function build_edges(tns::TNS{T,NDIMS}, qid::Integer, tid::Integer) where {T,NDIMS}
    buf = build_edges!(tns, qid, tid)
    return (
        senders = buf.senders,
        receivers = buf.receivers,
        rel_displacement = buf.rel_displacement,
        rel_dist_norm = buf.rel_dist_norm,
    )
end

"""
    materialize_all_neighbors!(tns)

Build CSR-style neighbor lists for every active pair. Opt-in; default
`for_each_neighbor` iteration uses no materialized storage.

CPU-only in v0.1. On a CUDA `TNS`, use `for_each_neighbor_device` /
`@for_each_neighbor_device_inline` for in-kernel iteration, or `build_edges`
for a flat-COO edge list.
"""
function materialize_all_neighbors!(tns::TNS{T,NDIMS}) where {T,NDIMS}
    tns.dev === :cpu || error("materialize_all_neighbors! on GPU is not wired yet; use for_each_neighbor_device")
    @inbounds for (idx, (q, t)) in enumerate(tns.active_pairs)
        buf = tns.neighbor_buffers[idx]
        tree = tns.trees[t]
        perm_t = tns.permutation[t]
        coords_q = tns.point_sets[q].coords
        coords_t = tns.point_sets[t].coords
        exclude_self = (q == t)
        materialize_cpu!(buf, tree, perm_t, coords_q, coords_t,
                         tns.query_stacks[1], tns.radius, exclude_self)
    end
    return tns
end

# ---------------- z-sort --------------------------------------------------

"""
    prepare_zsort!(tns) -> tns

Precondition check matching the C++ TreeNSearch API: asserts that the
permutation for every registered point set is populated (i.e. `run!(tns)`
has been called since the last build). Does not allocate or mutate state —
call this when you want a clear error before invoking `apply_zsort!` from
a hot loop. It is safe but unnecessary to call it every step;
`apply_zsort!` itself does not depend on it.
"""
function prepare_zsort!(tns::TNS)
    for set_id in eachindex(tns.point_sets)
        length(tns.permutation[set_id]) == tns.point_sets[set_id].n ||
            error("prepare_zsort!: call run! first so the permutation for set $set_id is available")
    end
    return tns
end

"""
    apply_zsort!(tns, set_id, user_array) -> permuted_array

Return a new array whose entries follow the z-order permutation built for
point set `set_id`. Accepts a 1-D vector (length N) or a 2-D matrix
(`(F, N)`); higher-rank inputs are rejected. Works on both CPU and CUDA
backends — the returned array lives on the same device as `user_array`.
"""
function apply_zsort!(tns::TNS, set_id::Integer, user_array::AbstractArray)
    perm = tns.permutation[set_id]
    if tns.dev === :cuda
        return _apply_zsort_cuda(perm, user_array)
    end
    return apply_zsort_cpu!(similar(user_array), perm, user_array)
end

# ---------------- device-side iterator stubs ------------------------------
# These dispatch to the CUDA extension at runtime when the user is on GPU.

function device_view(tns::TNS, qid::Integer, tid::Integer)
    tns.dev === :cuda || error("device_view requires a CUDA TNS")
    return _device_view(tns, qid, tid)
end

# Placeholder so symbol resolves; the real method lives in the CUDA ext and
# uses `@inline` so the user's `@cuda` kernel inlines the traversal.
function for_each_neighbor_device end
