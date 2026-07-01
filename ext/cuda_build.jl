# GPU-resident octree build. Replaces the v0.1 host round-trip (copy coords
# D→H, bin/sort/build/refit on CPU single-threaded, upload tree H→D) with an
# all-device pipeline:
#
#   1. origin   = min over coords per axis - 1e-6        (GPU reduction)
#   2. morton   = bin + Z-encode each point              (map kernel)
#   3. perm     = sortperm(morton)                       (GPU radix/quick sort)
#   4. topology = level-synchronous BFS, atomic child alloc, no per-level sync
#   5. refit    = bottom-up AABB fit, deepest level first
#
# The produced tree is an ordinary (NDIMS, n_nodes) flat-SoA Octree that the
# existing `for_each_neighbor_device` traversal consumes unchanged. The build
# touches the host exactly once (final node count + overflow flag); everything
# else — including the per-level node-id bounds — stays resident on the device.
#
# Dim is carried as `Val{NDIMS}` so NCH = 2^NDIMS and the bits-per-axis budget
# constant-fold; one code path serves both quadtree (2D) and octree (3D).

# Bits-per-axis must match src/morton.jl: 21 (3D, 63-bit code) / 31 (2D, 62-bit).
@inline _gpu_bits(::Val{3}) = Int32(21)
@inline _gpu_bits(::Val{2}) = Int32(31)
@inline _gpu_maxlev(::Val{3}) = Int32(21)
@inline _gpu_maxlev(::Val{2}) = Int32(31)

# ---------------- phase 2: Morton binning ---------------------------------

function _morton_bin_kernel!(morton, coords, o1::T, o2::T, o3::T, inv_cs::T,
                             n::Int32, ::Val{3}) where {T}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n && return nothing
    @inbounds begin
        px = coords[1, i]; py = coords[2, i]; pz = coords[3, i]
        ix = clamp(unsafe_trunc(Int32, floor((px - o1) * inv_cs)), Int32(0), Int32(0x1fffff))
        iy = clamp(unsafe_trunc(Int32, floor((py - o2) * inv_cs)), Int32(0), Int32(0x1fffff))
        iz = clamp(unsafe_trunc(Int32, floor((pz - o3) * inv_cs)), Int32(0), Int32(0x1fffff))
        morton[i] = Octopus.morton_encode3(ix, iy, iz)
    end
    return nothing
end

function _morton_bin_kernel!(morton, coords, o1::T, o2::T, o3::T, inv_cs::T,
                             n::Int32, ::Val{2}) where {T}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n && return nothing
    @inbounds begin
        px = coords[1, i]; py = coords[2, i]
        ix = clamp(unsafe_trunc(Int32, floor((px - o1) * inv_cs)), Int32(0), Int32(0x7fffffff))
        iy = clamp(unsafe_trunc(Int32, floor((py - o2) * inv_cs)), Int32(0), Int32(0x7fffffff))
        morton[i] = Octopus.morton_encode2(ix, iy)
    end
    return nothing
end

# ---------------- octant partition helper ---------------------------------
# Within a Morton-sorted node range [f, l], the octant value
# `(morton[perm[i]] >> shift) & mask` is non-decreasing in i, so octants form
# contiguous runs. `_octant_first_geq` binary-searches the first sorted index
# whose octant value is ≥ v — i.e. the boundary between octant (v-1) and v.

@inline function _octant_first_geq(morton, perm, f::Int32, l::Int32,
                                   shift::Int32, mask::UInt64, v::Int32)
    lo = f; hi = l + Int32(1)
    @inbounds while lo < hi
        mid = (lo + hi) >> 1
        ov = Int32((morton[perm[mid]] >> shift) & mask)
        if ov >= v
            hi = mid
        else
            lo = mid + Int32(1)
        end
    end
    return lo
end

@inline function _node_is_leaf(count::Int32, lvl::Int32, leaf::Int32, maxlev::Int32)
    return count <= leaf || lvl >= maxlev
end

@inline _level_shift_dev(lvl::Int32, ::Val{NDIMS}) where {NDIMS} =
    Int32(NDIMS) * _gpu_bits(Val(NDIMS)) - Int32(NDIMS) * (lvl + Int32(1))

# ---------------- topology build (sync-free) ------------------------------
# Level-synchronous BFS over the Morton-sorted permutation with no per-level
# host sync. Node ids come from a single device atomic counter (fetch-add, no
# scan); node arrays are over-allocated to a safe estimate with a device
# overflow flag + host retry (no per-level grow); the loop runs a fixed
# maxlev+1 rounds (empty levels are no-op launches) so the host never needs a
# per-level node count; per-level [start,end] bounds live on-device, advanced
# by a 1-thread kernel between rounds. Exactly one host sync per build (final
# count + overflow). A node level == the loop round, so no node_level array.

function _init_build_kernel!(nf, nl, lev_start, lev_end, counter, n::Int32)
    @inbounds begin
        nf[1] = Int32(1); nl[1] = n
        lev_start[1] = Int32(1); lev_end[1] = Int32(1)
        counter[1] = Int32(1)
    end
    return nothing
end

# One BFS round. Threads map onto the current level's node-id range (read from
# the device). A leaf stamps -1 sentinels; an internal node fetch-adds its
# non-empty-octant count from `counter` to reserve a contiguous child block.
function _build_level_kernel!(nf, nl, nchild, morton, perm, counter,
                              lev_start, lev_end, capacity::Int32, overflow,
                              leaf::Int32, L::Int32, ::Val{NDIMS}) where {NDIMS}
    p = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    # Once an earlier level overran `capacity`, this whole build attempt is
    # discarded and the host retries with a larger estimate. Stop every later
    # level immediately: a bailing parent reserves a child block in the counter
    # but writes none of its slots, so the sub-`capacity` slots of a straddling
    # block hold uninitialized node_first/node_last. Without this gate the next
    # level reads that garbage [f,l] (the `g > capacity` guard below can't catch
    # g <= capacity) and drives an out-of-range perm[] read in the octant search.
    @inbounds (overflow[1] != Int32(0)) && return nothing
    @inbounds cur_start = lev_start[L + Int32(1)]
    @inbounds cur_end   = lev_end[L + Int32(1)]
    g = cur_start + p - Int32(1)
    g > cur_end && return nothing
    # A prior level may have overshot capacity (the atomic counter is bumped
    # before the overflow check and never rolled back), so cur_end can exceed
    # `capacity`. Reject ghost node ids before any nf/nl read or nchild write
    # to keep the (about-to-be-discarded) overflow attempt memory-safe.
    g > capacity && return nothing
    NCH = Int32(1) << NDIMS
    maxlev = _gpu_maxlev(Val(NDIMS))
    @inbounds f = nf[g]
    @inbounds l = nl[g]
    count = l - f + Int32(1)
    if _node_is_leaf(count, L, leaf, maxlev)
        for o in Int32(1):NCH
            @inbounds nchild[o, g] = Int32(-1)
        end
        return nothing
    end
    shift = _level_shift_dev(L, Val(NDIMS))
    mask  = UInt64(NCH - Int32(1))

    # Octant boundaries b[1..NCH+1]; MVector{9} covers 3D (NCH+1=9) and 2D (5).
    b = MVector{9,Int32}(undef)
    @inbounds b[1] = f
    @inbounds b[NCH + Int32(1)] = l + Int32(1)
    for o in Int32(1):(NCH - Int32(1))
        @inbounds b[o + Int32(1)] = _octant_first_geq(morton, perm, f, l, shift, mask, o)
    end
    nc = Int32(0)
    for o in Int32(1):NCH
        (@inbounds(b[o]) < @inbounds(b[o + Int32(1)])) && (nc += Int32(1))
    end

    base = CUDA.atomic_add!(pointer(counter, 1), nc)   # returns pre-add value
    if base + nc > capacity
        @inbounds overflow[1] = Int32(1)               # discard tree, host retries
        for o in Int32(1):NCH
            @inbounds nchild[o, g] = Int32(-1)
        end
        return nothing
    end
    m = Int32(0)
    for o in Int32(1):NCH
        @inbounds lo = b[o]
        @inbounds hi = b[o + Int32(1)]
        if lo < hi
            m += Int32(1)
            cid = base + m
            @inbounds nf[cid] = lo
            @inbounds nl[cid] = hi - Int32(1)
            @inbounds nchild[o, g] = cid
        else
            @inbounds nchild[o, g] = Int32(-1)
        end
    end
    return nothing
end

# Promote the next level's [start,end] from the running counter. 1 thread.
function _advance_level_kernel!(lev_start, lev_end, counter, L::Int32)
    @inbounds begin
        new_n = counter[1]
        lev_start[L + Int32(2)] = lev_end[L + Int32(1)] + Int32(1)
        lev_end[L + Int32(2)]   = new_n
    end
    return nothing
end

# Refit one level (deepest first); node-id range read from the device.
function _refit_level_kernel!(bmin, bmax, nf, nl, nchild, coords, perm,
                                 lev_start, lev_end, L::Int32, ::Val{NDIMS}) where {NDIMS}
    p = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    @inbounds cur_start = lev_start[L + Int32(1)]
    @inbounds cur_end   = lev_end[L + Int32(1)]
    g = cur_start + p - Int32(1)
    g > cur_end && return nothing
    T = eltype(bmin)
    NCH = Int32(1) << NDIMS
    INF = typemax(T)
    NEG = typemin(T)
    @inbounds if nchild[1, g] == Int32(-1)
        f = nf[g]; l = nl[g]
        for d in 1:NDIMS
            lo = INF; hi = NEG
            for k in f:l
                x = coords[d, perm[k]]
                lo = min(lo, x); hi = max(hi, x)
            end
            bmin[d, g] = lo; bmax[d, g] = hi
        end
    else
        for d in 1:NDIMS
            lo = INF; hi = NEG
            for c in Int32(1):NCH
                cid = nchild[c, g]
                cid == Int32(-1) && continue
                lo = min(lo, bmin[d, cid]); hi = max(hi, bmax[d, cid])
            end
            bmin[d, g] = lo; bmax[d, g] = hi
        end
    end
    return nothing
end

function _build_gpu_tree!(gpu_tree::Octree{T,NDIMS}, coords::CuMatrix{T},
                             n::Int, cell_size::T, target_leaf_size::Int32,
                             want_morton::Bool) where {T,NDIMS}
    NCH = 1 << NDIMS
    threads = 256
    maxlev = Int(_gpu_maxlev(Val(NDIMS)))

    # --- origin / morton / sort (same as v1) ---
    mins = vec(Array(minimum(coords; dims = 2)))
    origin = ntuple(d -> mins[d] - T(1e-6), Val(NDIMS))
    morton = CuArray{UInt64}(undef, n)
    inv_cs = inv(cell_size)
    o3 = NDIMS == 3 ? origin[3] : zero(T)
    @cuda threads=threads blocks=cld(n, threads) _morton_bin_kernel!(
        morton, coords, origin[1], origin[2], o3, inv_cs, Int32(n), Val(NDIMS))
    perm = Int32.(sortperm(morton))

    # --- topology: fixed-round, sync-free, with overflow retry ---
    est = max(NCH * cld(n, max(Int(target_leaf_size), 1)) + 64, 64)
    gridb = cld(n, threads)
    local nf, nl, nchild, lev_start, lev_end, n_nodes
    while true
        nf     = CuArray{Int32}(undef, est)
        nl     = CuArray{Int32}(undef, est)
        nchild = CuArray{Int32}(undef, NCH, est)
        counter   = CuArray{Int32}(undef, 1)
        lev_start = CuArray{Int32}(undef, maxlev + 2)
        lev_end   = CuArray{Int32}(undef, maxlev + 2)
        overflow  = CUDA.zeros(Int32, 1)

        @cuda threads=1 _init_build_kernel!(nf, nl, lev_start, lev_end, counter, Int32(n))
        for L in 0:maxlev
            @cuda threads=threads blocks=gridb _build_level_kernel!(
                nf, nl, nchild, morton, perm, counter, lev_start, lev_end,
                Int32(est), overflow, target_leaf_size, Int32(L), Val(NDIMS))
            @cuda threads=1 _advance_level_kernel!(lev_start, lev_end, counter, Int32(L))
        end

        # The only host sync of the whole build.
        n_nodes = Int(CUDA.@allowscalar counter[1])
        (CUDA.@allowscalar overflow[1]) == Int32(0) && break
        est = max(2 * est, n_nodes + 64)               # rare: retry larger
    end

    # --- refit, deepest level first (level→id map read on device) ---
    bmin = CuArray{T}(undef, NDIMS, n_nodes)
    bmax = CuArray{T}(undef, NDIMS, n_nodes)
    for L in maxlev:-1:0
        @cuda threads=threads blocks=gridb _refit_level_kernel!(
            bmin, bmax, nf, nl, nchild, coords, perm, lev_start, lev_end,
            Int32(L), Val(NDIMS))
    end

    gpu_tree.node_bounds_min = bmin
    gpu_tree.node_bounds_max = bmax
    gpu_tree.node_first    = nf[1:n_nodes]
    gpu_tree.node_last     = nl[1:n_nodes]
    gpu_tree.node_children = nchild[:, 1:n_nodes]
    gpu_tree.n_nodes       = Int32(n_nodes)

    return (perm, want_morton ? morton : nothing)
end
