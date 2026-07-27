#
# Copyright (c) 2026 Josef Jouaux, Chair of Mechatronics, University of Augsburg
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# Path 2 spike: optimize the write_kernel.
#
# Phase-0 profile showed write_kernel is 75% of 2D N=20k wall time. The
# current implementation in vs_graphnetsim_gpu.jl uses `for_each_neighbor_device`
# with a `do j ... end` closure. Because the closure captures `cursors[i]` and
# advances it per-neighbor, every emitted edge does a global-memory load+store
# of the cursor. That's `n_edges` extra round-trips through device memory.
#
# This spike compares three variants on the *same* tree:
#   (V0) baseline — closure + cursors[i] global mutation (what we ship today)
#   (V1) inlined  — copy-paste the traversal, keep oo in a register, write
#                   cursors[i] only once at the end. Also precomputes inv_r.
#   (V2) inlined  + fast division (multiply by inv_r instead of /radius
#                   inside the hot loop; also rsqrt for distance norm).
#
# Run: julia --project=benchmarks --threads=auto benchmarks/spike_write_kernel.jl

using Octopus
using CUDA
using StaticArrays
using Random
using Printf
using Statistics

CUDA.functional() || error("CUDA required")
const _CUDA_EXT = Base.get_extension(Octopus, :OctopusCUDAExt)
@assert _CUDA_EXT !== nothing "OctopusCUDAExt not loaded"

# ------------------------------------------------------------------------
# V0 baseline — same code as vs_graphnetsim_gpu.jl::write_kernel_3d!
# ------------------------------------------------------------------------

function write_v0_3d!(senders, receivers, rd, rdn, cursors, dv, coords_orig, N, radius)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > N && return nothing
    i32 = Int32(i)

    @inbounds start = cursors[i]
    @inbounds senders[start]   = i32
    @inbounds receivers[start] = i32
    @inbounds rd[1, start] = 0f0; @inbounds rd[2, start] = 0f0; @inbounds rd[3, start] = 0f0
    @inbounds rdn[start] = 0f0
    @inbounds cursors[i] = start + Int32(1)

    px = @inbounds coords_orig[1, i]
    py = @inbounds coords_orig[2, i]
    pz = @inbounds coords_orig[3, i]

    for_each_neighbor_device(dv, i) do j
        if j != i32
            @inbounds oo = cursors[i]
            qx = @inbounds coords_orig[1, j]
            qy = @inbounds coords_orig[2, j]
            qz = @inbounds coords_orig[3, j]
            dx = px - qx; dy = py - qy; dz = pz - qz
            d2 = dx*dx + dy*dy + dz*dz
            @inbounds senders[oo] = j
            @inbounds receivers[oo] = i32
            @inbounds rd[1, oo] = dx / radius
            @inbounds rd[2, oo] = dy / radius
            @inbounds rd[3, oo] = dz / radius
            @inbounds rdn[oo] = sqrt(d2) / radius
            @inbounds cursors[i] = oo + Int32(1)
        end
    end
    return nothing
end

# ------------------------------------------------------------------------
# V1 inlined — traversal copy-paste, cursor in register, single write-back
# ------------------------------------------------------------------------

function write_v1_3d!(senders, receivers, rd, rdn, cursors, dv, coords_orig, N, radius)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > N && return nothing
    i32 = Int32(i)

    @inbounds oo = cursors[i]

    # Self edge
    @inbounds senders[oo]   = i32
    @inbounds receivers[oo] = i32
    @inbounds rd[1, oo] = 0f0; @inbounds rd[2, oo] = 0f0; @inbounds rd[3, oo] = 0f0
    @inbounds rdn[oo] = 0f0
    oo += Int32(1)

    px = @inbounds coords_orig[1, i]
    py = @inbounds coords_orig[2, i]
    pz = @inbounds coords_orig[3, i]
    r_sq = dv.radius_sq
    inv_r = 1f0 / radius

    # Inlined traversal — register stack of size 40, depth-first descent
    stack = MVector{40, Int32}(undef)
    sp = 1
    @inbounds stack[sp] = Int32(1)

    while sp > 0
        nid = @inbounds stack[sp]
        sp -= 1

        @inbounds begin
            lo1 = dv.bounds_min[1, nid]; lo2 = dv.bounds_min[2, nid]; lo3 = dv.bounds_min[3, nid]
            hi1 = dv.bounds_max[1, nid]; hi2 = dv.bounds_max[2, nid]; hi3 = dv.bounds_max[3, nid]
        end
        # AABB-sphere overlap
        ddx = max(lo1 - px, 0f0); ddx = max(ddx, px - hi1)
        ddy = max(lo2 - py, 0f0); ddy = max(ddy, py - hi2)
        ddz = max(lo3 - pz, 0f0); ddz = max(ddz, pz - hi3)
        if ddx*ddx + ddy*ddy + ddz*ddz > r_sq
            continue
        end

        if @inbounds(dv.node_children[1, nid]) == Int32(-1)
            # Leaf — flat distance loop with per-edge write
            f0 = @inbounds dv.node_first[nid]
            l0 = @inbounds dv.node_last[nid]
            @inbounds for k in f0:l0
                j = dv.perm[k]
                if j != i32
                    qx = coords_orig[1, j]; qy = coords_orig[2, j]; qz = coords_orig[3, j]
                    dx = qx - px; dy = qy - py; dz = qz - pz
                    d2 = dx*dx + dy*dy + dz*dz
                    if d2 <= r_sq
                        senders[oo] = j
                        receivers[oo] = i32
                        rd[1, oo] = dx * inv_r
                        rd[2, oo] = dy * inv_r
                        rd[3, oo] = dz * inv_r
                        rdn[oo] = sqrt(d2) * inv_r
                        oo += Int32(1)
                    end
                end
            end
        else
            @inbounds for c in 1:8
                cid = dv.node_children[c, nid]
                if cid != Int32(-1)
                    sp += 1
                    stack[sp] = cid
                end
            end
        end
    end

    @inbounds cursors[i] = oo
    return nothing
end

# ------------------------------------------------------------------------
# V2 inlined + child early-cull (skip pushing children that don't intersect)
# ------------------------------------------------------------------------

function write_v2_3d!(senders, receivers, rd, rdn, cursors, dv, coords_orig, N, radius)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > N && return nothing
    i32 = Int32(i)

    @inbounds oo = cursors[i]

    @inbounds senders[oo]   = i32
    @inbounds receivers[oo] = i32
    @inbounds rd[1, oo] = 0f0; @inbounds rd[2, oo] = 0f0; @inbounds rd[3, oo] = 0f0
    @inbounds rdn[oo] = 0f0
    oo += Int32(1)

    px = @inbounds coords_orig[1, i]
    py = @inbounds coords_orig[2, i]
    pz = @inbounds coords_orig[3, i]
    r_sq = dv.radius_sq
    inv_r = 1f0 / radius

    stack = MVector{40, Int32}(undef)
    sp = 1
    @inbounds stack[sp] = Int32(1)

    while sp > 0
        nid = @inbounds stack[sp]
        sp -= 1

        if @inbounds(dv.node_children[1, nid]) == Int32(-1)
            # Already known to overlap (parent/root pre-checked).
            f0 = @inbounds dv.node_first[nid]
            l0 = @inbounds dv.node_last[nid]
            @inbounds for k in f0:l0
                j = dv.perm[k]
                if j != i32
                    qx = coords_orig[1, j]; qy = coords_orig[2, j]; qz = coords_orig[3, j]
                    dx = qx - px; dy = qy - py; dz = qz - pz
                    d2 = dx*dx + dy*dy + dz*dz
                    if d2 <= r_sq
                        senders[oo] = j
                        receivers[oo] = i32
                        rd[1, oo] = dx * inv_r
                        rd[2, oo] = dy * inv_r
                        rd[3, oo] = dz * inv_r
                        rdn[oo] = sqrt(d2) * inv_r
                        oo += Int32(1)
                    end
                end
            end
        else
            # Internal — test each child's AABB before pushing
            @inbounds for c in 1:8
                cid = dv.node_children[c, nid]
                if cid != Int32(-1)
                    lo1 = dv.bounds_min[1, cid]; lo2 = dv.bounds_min[2, cid]; lo3 = dv.bounds_min[3, cid]
                    hi1 = dv.bounds_max[1, cid]; hi2 = dv.bounds_max[2, cid]; hi3 = dv.bounds_max[3, cid]
                    ddx = max(lo1 - px, 0f0); ddx = max(ddx, px - hi1)
                    ddy = max(lo2 - py, 0f0); ddy = max(ddy, py - hi2)
                    ddz = max(lo3 - pz, 0f0); ddz = max(ddz, pz - hi3)
                    if ddx*ddx + ddy*ddy + ddz*ddz <= r_sq
                        sp += 1
                        stack[sp] = cid
                    end
                end
            end
        end
    end

    @inbounds cursors[i] = oo
    return nothing
end

# ------------------------------------------------------------------------
# Driver — set up TNS, get DV, count, alloc, run each variant.
# ------------------------------------------------------------------------

@inline function synced_ns()
    CUDA.synchronize()
    return time_ns()
end

function bench(f, n=5)
    f(); CUDA.synchronize()
    times = Float64[]
    for _ in 1:n
        t0 = synced_ns()
        f()
        t1 = synced_ns()
        push!(times, (t1 - t0) / 1e9)
    end
    return median(times)
end

function count_kernel!(counts, dv, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > N && return nothing
    @inbounds counts[i] = Int32(1)
    for_each_neighbor_device(dv, i) do j
        if j != Int32(i)
            @inbounds counts[i] += Int32(1)
        end
    end
    return nothing
end

function setup_scene(pos2d::Matrix{Float32}, radius::Float32)
    N = size(pos2d, 2)
    # Pad to 3D for the v0.1 path (matches what vs_graphnetsim_gpu does).
    pos3 = vcat(pos2d, zeros(Float32, 1, N))
    gpu_pos2 = CuArray(pos2d)
    gpu_pos3 = CuArray(pos3)
    tns = TNS(Float32; ndims=3)
    set_search_radius!(tns, radius)
    id = add_point_set!(tns, gpu_pos3)
    set_active_search!(tns, id, id)
    run!(tns)
    dv = device_view(tns, id, id)

    counts = CUDA.zeros(Int32, N)
    threads = 128; blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks count_kernel!(counts, dv, Int32(N))
    CUDA.synchronize()
    offsets = CUDA.cumsum(counts) .- counts .+ Int32(1)
    n_edges = Int(CUDA.@allowscalar offsets[end] + counts[end] - 1)
    return (; tns, dv, gpu_pos3, N, n_edges, offsets, threads, blocks)
end

function run_variant(kernel::F, scene, radius) where {F}
    (; dv, gpu_pos3, N, n_edges, offsets, threads, blocks) = scene
    senders = CuArray{Int32}(undef, n_edges)
    receivers = CuArray{Int32}(undef, n_edges)
    rd = CuArray{Float32}(undef, 3, n_edges)
    rdn = CuArray{Float32}(undef, 1, n_edges)
    cursors = copy(offsets)
    return bench() do
        cursors .= offsets
        @cuda threads=threads blocks=blocks kernel(senders, receivers, rd, rdn, cursors, dv, gpu_pos3, Int32(N), radius)
    end
end

# Edge-set match check: run V0 once, V1 once, compare sorted edge tuples.
function verify(scene, radius)
    (; dv, gpu_pos3, N, n_edges, offsets, threads, blocks) = scene
    function collect_edges(kernel)
        s = CuArray{Int32}(undef, n_edges)
        r = CuArray{Int32}(undef, n_edges)
        rd = CuArray{Float32}(undef, 3, n_edges)
        rdn = CuArray{Float32}(undef, 1, n_edges)
        cursors = copy(offsets)
        @cuda threads=threads blocks=blocks kernel(s, r, rd, rdn, cursors, dv, gpu_pos3, Int32(N), radius)
        CUDA.synchronize()
        return Set(zip(Array(r), Array(s)))
    end
    e0 = collect_edges(write_v0_3d!)
    e1 = collect_edges(write_v1_3d!)
    e2 = collect_edges(write_v2_3d!)
    return (length(symdiff(e0, e1)), length(symdiff(e0, e2)))
end

function main()
    println("write_kernel optimization spike (NVIDIA ", name(CUDA.device()), ", Julia ", VERSION, ")")
    println()

    # The headline test: 2D dam-break N=20k where write_kernel was 58.66 ms.
    Random.seed!(42)
    for (label, N, radius) in (
        ("2D N=  1k r=0.072",   1_000, 0.072f0),
        ("2D N=  5k r=0.072",   5_000, 0.072f0),
        ("2D N= 20k r=0.072",  20_000, 0.072f0),
    )
        Random.seed!(42)
        pos = Matrix{Float32}(undef, 2, N)
        pos[1, :] = 0.3f0  .* rand(Float32, N)
        pos[2, :] = 0.15f0 .* rand(Float32, N)

        scene = setup_scene(pos, radius)
        (s01, s02) = verify(scene, radius)

        t_v0 = run_variant(write_v0_3d!, scene, radius) * 1000
        t_v1 = run_variant(write_v1_3d!, scene, radius) * 1000
        t_v2 = run_variant(write_v2_3d!, scene, radius) * 1000

        println("=" ^ 90)
        println(label, "   n_edges = ", scene.n_edges)
        println("-" ^ 90)
        @printf("  V0 closure (baseline)    : %7.2f ms\n", t_v0)
        @printf("  V1 inlined + inv_r       : %7.2f ms   (%.2fx vs V0)   edge-set diff = %d\n",
                t_v1, t_v0 / t_v1, s01)
        @printf("  V2 V1 + child early-cull : %7.2f ms   (%.2fx vs V0)   edge-set diff = %d\n",
                t_v2, t_v0 / t_v2, s02)
        println()
        CUDA.reclaim()
    end
end

main()
