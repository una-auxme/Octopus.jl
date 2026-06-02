module TreeNSearchCUDAExt

using TreeNSearch
using CUDA
using StaticArrays
import TreeNSearch: _run_cuda!, _device_view, for_each_neighbor_device,
                    _build_edges_cuda!, _apply_zsort_cuda,
                    TNS, PointSet, Octree, NeighborBuffer, EdgeBuffer,
                    bin_cpu!, sort_by_key_cpu!, rle_cells_cpu!,
                    build_cpu!, refit_bounds_cpu!, _point_origin,
                    ensure_capacity!

function __init__()
    TreeNSearch._CUDA_EXT_LOADED[] = true
end

# ---------------- device view ----------------------------------------------
# An isbits struct holding CuDeviceArrays, safe to pass into user kernels.
# Parametric on NDIMS so 2D and 3D get distinct dispatched methods.

struct DeviceView{T,NDIMS}
    coords_q::CUDA.CuDeviceMatrix{T,1}
    coords_t::CUDA.CuDeviceMatrix{T,1}
    bounds_min::CUDA.CuDeviceMatrix{T,1}
    bounds_max::CUDA.CuDeviceMatrix{T,1}
    node_first::CUDA.CuDeviceVector{Int32,1}
    node_last::CUDA.CuDeviceVector{Int32,1}
    node_children::CUDA.CuDeviceMatrix{Int32,1}
    perm::CUDA.CuDeviceVector{Int32,1}
    n_nodes::Int32
    radius::T
    radius_sq::T
end

function _device_view(tns::TNS{T,NDIMS}, qid::Integer, tid::Integer) where {T,NDIMS}
    tns.dev === :cuda || error("device_view requires a CUDA TNS")
    qs = tns.point_sets[qid]
    ts = tns.point_sets[tid]
    tree = tns.trees[tid]
    perm = tns.permutation[tid]
    DeviceView{T,NDIMS}(
        CUDA.cudaconvert(qs.coords),
        CUDA.cudaconvert(ts.coords),
        CUDA.cudaconvert(tree.node_bounds_min),
        CUDA.cudaconvert(tree.node_bounds_max),
        CUDA.cudaconvert(tree.node_first),
        CUDA.cudaconvert(tree.node_last),
        CUDA.cudaconvert(tree.node_children),
        CUDA.cudaconvert(perm),
        tree.n_nodes,
        tns.radius,
        tns.radius * tns.radius,
    )
end

# ---------------- device-side iterator ------------------------------------
# Callable from inside a user's @cuda kernel. Each thread has its own local
# stack in register/shared memory. We keep the stack compact (depth ≤ 40).

@inline function _aabb_dist_sq_dev(px::T, py::T, pz::T,
                                    lo1::T, lo2::T, lo3::T,
                                    hi1::T, hi2::T, hi3::T)::T where {T}
    dx = max(lo1 - px, zero(T)); dx = max(dx, px - hi1)
    dy = max(lo2 - py, zero(T)); dy = max(dy, py - hi2)
    dz = max(lo3 - pz, zero(T)); dz = max(dz, pz - hi3)
    return dx*dx + dy*dy + dz*dz
end

@inline function _aabb_dist_sq_dev(px::T, py::T,
                                    lo1::T, lo2::T,
                                    hi1::T, hi2::T)::T where {T}
    dx = max(lo1 - px, zero(T)); dx = max(dx, px - hi1)
    dy = max(lo2 - py, zero(T)); dy = max(dy, py - hi2)
    return dx*dx + dy*dy
end

"""
    for_each_neighbor_device(callback, view, i)

Iterate indices `j` in the target point set that are within `view.radius` of
`view.coords_q[:, i]`. Callable only from inside a `@cuda` kernel. Two
methods exist: 3D (`DeviceView{T,3}`) and 2D (`DeviceView{T,2}`).
"""
@inline function for_each_neighbor_device(f::F, dv::DeviceView{T,3}, i::Integer) where {F,T}
    dv.n_nodes == Int32(0) && return nothing

    px = @inbounds dv.coords_q[1, i]
    py = @inbounds dv.coords_q[2, i]
    pz = @inbounds dv.coords_q[3, i]
    r_sq = dv.radius_sq

    stack = MVector{40,Int32}(undef)
    sp = 1
    @inbounds stack[sp] = Int32(1)

    while sp > 0
        nid = @inbounds stack[sp]
        sp -= 1

        @inbounds begin
            dsq = _aabb_dist_sq_dev(
                px, py, pz,
                dv.bounds_min[1, nid], dv.bounds_min[2, nid], dv.bounds_min[3, nid],
                dv.bounds_max[1, nid], dv.bounds_max[2, nid], dv.bounds_max[3, nid],
            )
        end
        dsq > r_sq && continue

        if @inbounds(dv.node_children[1, nid]) == Int32(-1)
            f0 = @inbounds dv.node_first[nid]
            l0 = @inbounds dv.node_last[nid]
            @inbounds for k in f0:l0
                j = dv.perm[k]
                qx = dv.coords_t[1, j]
                qy = dv.coords_t[2, j]
                qz = dv.coords_t[3, j]
                ddx = qx - px; ddy = qy - py; ddz = qz - pz
                d2 = ddx*ddx + ddy*ddy + ddz*ddz
                if d2 <= r_sq
                    f(j)
                end
            end
        else
            @inbounds for c in 1:8
                cid = dv.node_children[c, nid]
                if cid != Int32(-1)
                    cdsq = _aabb_dist_sq_dev(
                        px, py, pz,
                        dv.bounds_min[1, cid], dv.bounds_min[2, cid], dv.bounds_min[3, cid],
                        dv.bounds_max[1, cid], dv.bounds_max[2, cid], dv.bounds_max[3, cid],
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

@inline function for_each_neighbor_device(f::F, dv::DeviceView{T,2}, i::Integer) where {F,T}
    dv.n_nodes == Int32(0) && return nothing

    px = @inbounds dv.coords_q[1, i]
    py = @inbounds dv.coords_q[2, i]
    r_sq = dv.radius_sq

    stack = MVector{40,Int32}(undef)
    sp = 1
    @inbounds stack[sp] = Int32(1)

    while sp > 0
        nid = @inbounds stack[sp]
        sp -= 1

        @inbounds begin
            dsq = _aabb_dist_sq_dev(
                px, py,
                dv.bounds_min[1, nid], dv.bounds_min[2, nid],
                dv.bounds_max[1, nid], dv.bounds_max[2, nid],
            )
        end
        dsq > r_sq && continue

        if @inbounds(dv.node_children[1, nid]) == Int32(-1)
            f0 = @inbounds dv.node_first[nid]
            l0 = @inbounds dv.node_last[nid]
            @inbounds for k in f0:l0
                j = dv.perm[k]
                qx = dv.coords_t[1, j]
                qy = dv.coords_t[2, j]
                ddx = qx - px; ddy = qy - py
                d2 = ddx*ddx + ddy*ddy
                if d2 <= r_sq
                    f(j)
                end
            end
        else
            @inbounds for c in 1:4
                cid = dv.node_children[c, nid]
                if cid != Int32(-1)
                    cdsq = _aabb_dist_sq_dev(
                        px, py,
                        dv.bounds_min[1, cid], dv.bounds_min[2, cid],
                        dv.bounds_max[1, cid], dv.bounds_max[2, cid],
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

# ---------------- build orchestration --------------------------------------
# v0.1 strategy (documented in the plan): GPU-resident coords, CPU-hosted
# build via a temporary copy, then copy tree back to GPU. The heavy cost
# (per-query traversal) stays on GPU via for_each_neighbor_device.

function _run_cuda!(tns::TNS{T,NDIMS}) where {T,NDIMS}
    @inbounds for sid in eachindex(tns.point_sets)
        tns.dirty[sid] || continue
        ps = tns.point_sets[sid]
        coords_gpu = ps.coords
        n = Int(ps.n)

        # Empty point set: leave the tree empty so device-side traversal
        # short-circuits. Avoids _point_origin reading coords[1, 1] on n=0.
        if n == 0
            gpu_tree = tns.trees[sid]
            gpu_tree.n_nodes = Int32(0)
            tns.permutation[sid] = CuArray(Int32[])
            tns.morton_codes[sid] = CuArray(UInt64[])
            tns.dirty[sid] = false
            continue
        end

        # Transient CPU copy — O(N) transfer, amortized over many queries.
        coords_cpu = Array(coords_gpu)

        origin = _point_origin(coords_cpu)
        tns.origin[sid] = origin

        morton_cpu = UInt64[]
        ensure_capacity!(morton_cpu, n); resize!(morton_cpu, n)
        bin_cpu!(morton_cpu, coords_cpu, origin, tns.cell_size)

        perm_cpu = Vector{Int32}(undef, n)
        sort_by_key_cpu!(perm_cpu, morton_cpu)

        cell_m = UInt64[]; cell_f = Int32[]; cell_l = Int32[]
        rle_cells_cpu!(cell_m, cell_f, cell_l, morton_cpu, perm_cpu)

        # Build on CPU using a CPU scratch Octree, then copy arrays to GPU.
        cpu_tree = Octree{T,NDIMS}(coords_cpu)
        build_cpu!(cpu_tree, morton_cpu, perm_cpu, tns.target_leaf_size)
        refit_bounds_cpu!(cpu_tree, coords_cpu, perm_cpu)

        # Upload tree to GPU.
        gpu_tree = tns.trees[sid]
        gpu_tree.node_bounds_min = CuArray(cpu_tree.node_bounds_min)
        gpu_tree.node_bounds_max = CuArray(cpu_tree.node_bounds_max)
        gpu_tree.node_first      = CuArray(cpu_tree.node_first)
        gpu_tree.node_last       = CuArray(cpu_tree.node_last)
        gpu_tree.node_children   = CuArray(cpu_tree.node_children)
        gpu_tree.n_nodes         = cpu_tree.n_nodes

        tns.permutation[sid] = CuArray(perm_cpu)
        if tns.refit_mode
            tns.morton_codes[sid] = CuArray(morton_cpu)
        else
            tns.morton_codes[sid] = CuArray(UInt64[])
        end

        tns.dirty[sid] = false
    end

    for buf in tns.neighbor_buffers
        buf.materialized = false
    end
    for buf in tns.edge_buffers
        buf.materialized = false
    end
    return tns
end

# ---------------- edge construction (CUDA) --------------------------------
# Two-pass build matching the CPU layout: count → exclusive scan → fill.
# Each thread services one query point i; pass-B writes are into disjoint
# index slices [offsets[i]+1 .. offsets[i]+counts[i]] so no atomics needed.
#
# Kernels are dim-aware via the DeviceView{T,NDIMS} type parameter; the
# displacement write loops over `1:NDIMS` (constant-folded at compile time).

# Two NDIMS-specialized methods because the inline-traversal macros are
# parse-time and need a literal 3D-vs-2D choice. Closure form would box `c`
# (CUDA rejects boxed locals); the macro keeps `c` in a register.

function _count_edges_kernel!(counts, dv::DeviceView{T,3}, exclude_self, n_q) where {T}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n_q && return nothing
    i32 = Int32(i)
    c = Int32(0)
    TreeNSearch.@for_each_neighbor_device_inline dv i j begin
        if !exclude_self || j != i32
            c += Int32(1)
        end
    end
    @inbounds counts[i] = c
    return nothing
end

function _count_edges_kernel!(counts, dv::DeviceView{T,2}, exclude_self, n_q) where {T}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n_q && return nothing
    i32 = Int32(i)
    c = Int32(0)
    TreeNSearch.@for_each_neighbor_device_inline_2d dv i j begin
        if !exclude_self || j != i32
            c += Int32(1)
        end
    end
    @inbounds counts[i] = c
    return nothing
end

function _fill_edges_kernel_3d!(senders, receivers, rdisp, rdist,
                                dv::DeviceView{T,3}, offsets, exclude_self, inv_r, n_q) where {T}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n_q && return nothing
    i32 = Int32(i)
    cursor = @inbounds offsets[i]
    px = @inbounds dv.coords_q[1, i]
    py = @inbounds dv.coords_q[2, i]
    pz = @inbounds dv.coords_q[3, i]
    inv_r_local = inv_r
    TreeNSearch.@for_each_neighbor_device_inline dv i j begin
        if !exclude_self || j != i32
            cursor += Int32(1)
            qx = @inbounds dv.coords_t[1, j]
            qy = @inbounds dv.coords_t[2, j]
            qz = @inbounds dv.coords_t[3, j]
            dx = px - qx; dy = py - qy; dz = pz - qz
            d2 = dx*dx + dy*dy + dz*dz
            @inbounds senders[cursor]   = j
            @inbounds receivers[cursor] = i32
            @inbounds rdisp[1, cursor]  = dx * inv_r_local
            @inbounds rdisp[2, cursor]  = dy * inv_r_local
            @inbounds rdisp[3, cursor]  = dz * inv_r_local
            @inbounds rdist[1, cursor]  = CUDA.sqrt(d2) * inv_r_local
        end
    end
    return nothing
end

function _fill_edges_kernel_2d!(senders, receivers, rdisp, rdist,
                                dv::DeviceView{T,2}, offsets, exclude_self, inv_r, n_q) where {T}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n_q && return nothing
    i32 = Int32(i)
    cursor = @inbounds offsets[i]
    px = @inbounds dv.coords_q[1, i]
    py = @inbounds dv.coords_q[2, i]
    inv_r_local = inv_r
    TreeNSearch.@for_each_neighbor_device_inline_2d dv i j begin
        if !exclude_self || j != i32
            cursor += Int32(1)
            qx = @inbounds dv.coords_t[1, j]
            qy = @inbounds dv.coords_t[2, j]
            dx = px - qx; dy = py - qy
            d2 = dx*dx + dy*dy
            @inbounds senders[cursor]   = j
            @inbounds receivers[cursor] = i32
            @inbounds rdisp[1, cursor]  = dx * inv_r_local
            @inbounds rdisp[2, cursor]  = dy * inv_r_local
            @inbounds rdist[1, cursor]  = CUDA.sqrt(d2) * inv_r_local
        end
    end
    return nothing
end

function _build_edges_cuda!(tns::TNS{T,NDIMS}, pair_idx::Integer, qid::Integer, tid::Integer) where {T,NDIMS}
    buf = tns.edge_buffers[pair_idx]
    coords_q = tns.point_sets[qid].coords
    n_q = Int(tns.point_sets[qid].n)
    exclude_self = (qid == tid)
    r = tns.radius
    inv_r = inv(r)

    if n_q == 0
        if length(buf.senders) != 0
            buf.senders   = similar(coords_q, Int32, 0)
            buf.receivers = similar(coords_q, Int32, 0)
        end
        if size(buf.rel_displacement, 2) != 0
            buf.rel_displacement = similar(coords_q, T, (NDIMS, 0))
        end
        if size(buf.rel_dist_norm, 2) != 0
            buf.rel_dist_norm = similar(coords_q, T, (1, 0))
        end
        buf.n_edges = Int32(0)
        buf.materialized = true
        return buf
    end

    dv = _device_view(tns, qid, tid)

    counts = CUDA.zeros(Int32, n_q)
    threads = 128
    blocks  = cld(n_q, threads)
    @cuda threads=threads blocks=blocks _count_edges_kernel!(counts, dv, exclude_self, Int32(n_q))

    # Exclusive scan: offsets[i] = sum(counts[1:i-1]).
    inclusive = CUDA.cumsum(counts)
    offsets   = inclusive .- counts
    n_edges   = Int(CUDA.@allowscalar inclusive[end])

    if length(buf.senders) != n_edges
        buf.senders   = similar(coords_q, Int32, n_edges)
        buf.receivers = similar(coords_q, Int32, n_edges)
    end
    if size(buf.rel_displacement, 2) != n_edges
        buf.rel_displacement = similar(coords_q, T, (NDIMS, n_edges))
    end
    if size(buf.rel_dist_norm, 2) != n_edges
        buf.rel_dist_norm = similar(coords_q, T, (1, n_edges))
    end
    buf.n_edges = Int32(n_edges)

    if n_edges > 0
        if NDIMS == 3
            @cuda threads=threads blocks=blocks _fill_edges_kernel_3d!(
                buf.senders, buf.receivers, buf.rel_displacement, buf.rel_dist_norm,
                dv, offsets, exclude_self, T(inv_r), Int32(n_q))
        else
            @cuda threads=threads blocks=blocks _fill_edges_kernel_2d!(
                buf.senders, buf.receivers, buf.rel_displacement, buf.rel_dist_norm,
                dv, offsets, exclude_self, T(inv_r), Int32(n_q))
        end
    end

    buf.materialized = true
    return buf
end

# ---------------- z-sort apply (GPU) --------------------------------------
# Mirrors apply_zsort_cpu!: 1-D / 2-D only, last-axis (column) permute. The
# CuArray indexing path runs as a device-resident gather without scalar
# transfers — perm is already a CuArray on the same device as user_array.

function _apply_zsort_cuda(perm::AbstractVector{Int32}, user_array::AbstractArray)
    ndims(user_array) <= 2 || throw(ArgumentError(
        "apply_zsort! supports 1-D or 2-D arrays only; got $(ndims(user_array))-D"))
    length(perm) == size(user_array, ndims(user_array)) ||
        throw(DimensionMismatch("perm length $(length(perm)) does not match last axis $(size(user_array, ndims(user_array)))"))
    if ndims(user_array) == 1
        return user_array[perm]
    else
        return user_array[:, perm]
    end
end

# ---------------- octree constructor for GPU arrays -----------------------
# `Octree{T,NDIMS}` constructor takes an `AbstractArray` to pick backend; we
# need it to work when the user hands us a CuArray.

function TreeNSearch.Octree{T,NDIMS}(backend_like::CuArray) where {T,NDIMS}
    NCH = 1 << NDIMS
    MF = CUDA.zeros(T, NDIMS, 0)
    MI = CUDA.zeros(Int32, NCH, 0)
    VI = CUDA.zeros(Int32, 0)
    Octree{T,NDIMS,typeof(VI),typeof(MF),typeof(MI)}(
        MF, similar(MF), VI, similar(VI), MI, Int32(0))
end

end # module
