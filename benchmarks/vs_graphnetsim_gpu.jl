#!/usr/bin/env julia
# GPU benchmark matching GraphNetSim.jl's `point_neighbor_ns(::CuArray, ...)`.
# Reference: GraphNetSim.jl/src/graph.jl:319-360
#
# GraphNetSim's GPU path:
#   - Build `GridNeighborhoodSearch` on CPU (via Array(pos)).
#   - `adapt(CUDABackend(), nhs)` to move cell list to GPU.
#   - `foreach_point_neighbor` runs kernels on GPU, callback mutates GPU arrays.
#
# Octopus GPU path:
#   - CuArray coords → `add_point_set!` infers :cuda backend.
#   - `run!(tns)` builds on CPU from a transient copy and uploads tree (v0.1
#     strategy documented in the plan; native GPU build is v0.2).
#   - User kernel calls `for_each_neighbor_device` inside `@cuda`.
#
# Run:
#   julia --project=benchmarks --threads=auto benchmarks/vs_graphnetsim_gpu.jl

using Octopus
using PointNeighbors
using CUDA
using Adapt
using StaticArrays
using BenchmarkTools
using Random
using Printf
using Statistics

CUDA.functional() || error("CUDA not functional; cannot run GPU benchmark")
println("Using GPU: ", name(CUDA.device()))

# ------------------------------------------------------------------------
# PointNeighbors GPU reference — verbatim shape from GraphNetSim.jl:319.
# ------------------------------------------------------------------------

function pn_point_neighbor_ns_gpu(pos::CuArray, radius::Float32)
    system = pos
    min_corner = minimum(pos; dims = 2)
    max_corner = maximum(pos; dims = 2)

    D = size(pos, 1)
    N = size(pos, 2)

    # Size FullGridCellList to avoid BoundsError on dense random clouds.
    extent = Array(vec(max_corner - min_corner))
    cell_vol = prod(extent) / prod(ceil.(extent ./ radius))
    avg_per_cell = N * cell_vol / max(prod(extent), eps(Float32))
    mppc = max(Int(ceil(4 * avg_per_cell)) + 32, 100)

    nhs = GridNeighborhoodSearch{D}(;
        search_radius = radius,
        n_points = N,
        cell_list = FullGridCellList(; min_corner, max_corner,
                                     search_radius = radius,
                                     max_points_per_cell = mppc),
    )

    # GraphNetSim initializes with Array(pos) (build on CPU, then adapt).
    initialize!(nhs, Array(system), Array(pos))
    nhs_gpu = adapt(CUDABackend(), nhs)

    n_neighbors_gpu = CUDA.zeros(Int, N)
    foreach_point_neighbor(system, pos, nhs_gpu) do i, _, _, _
        n_neighbors_gpu[i] += 1
    end
    n_edges = Int(sum(n_neighbors_gpu))

    senders = CuArray{Int32}(undef, n_edges)
    receivers = CuArray{Int32}(undef, n_edges)
    rel_displacement = CuArray{Float32}(undef, D, n_edges)
    rel_dist_norm = CuArray{Float32}(undef, 1, n_edges)

    offset = CUDA.cumsum(n_neighbors_gpu) .- n_neighbors_gpu .+ 1
    foreach_point_neighbor(system, pos, nhs_gpu) do i, j, pos_diff, distance
        receivers[offset[i]] = i
        senders[offset[i]] = j
        for d in 1:D
            rel_displacement[d, offset[i]] = pos_diff[d] / radius
        end
        rel_dist_norm[offset[i]] = distance / radius
        offset[i] += 1
    end
    CUDA.synchronize()
    return senders, receivers, rel_displacement, rel_dist_norm, nhs_gpu
end

# ------------------------------------------------------------------------
# Octopus GPU path — same output shape, same self-edge semantics.
# ------------------------------------------------------------------------

# Pass A: count (including self). The closure mutation of a scalar local
# doesn't survive GPU compilation (boxing), so accumulate in `counts[i]` —
# each thread writes only to its own slot so there's no race.
function count_kernel!(counts, dv, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > N && return nothing
    @inbounds counts[i] = Int32(1)  # self
    for_each_neighbor_device(dv, i) do j
        if j != Int32(i)
            @inbounds counts[i] += Int32(1)
        end
    end
    return nothing
end

# Pass B: write edges. We inline the tree traversal here (rather than going
# through `for_each_neighbor_device`'s closure-based iterator) so the cursor
# can stay in a register and `inv_r` is precomputed. The closure path forces
# Julia to box `cursor` and reload it from global memory per emitted edge,
# which on the A30 costs ~40% of the wall time at 2D N=20k. Spike data:
# benchmarks/spike_write_kernel.jl shows V2 (this layout) is 1.43× faster
# at N=20k 2D, 1.75× at N=5k, 2.05× at N=1k vs the closure path. Edge sets
# are byte-for-byte identical.

function write_kernel_3d!(senders, receivers, rd, rdn, cursors, dv, coords_orig, N, radius)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > N && return nothing
    i32 = Int32(i)

    @inbounds oo = cursors[i]

    # Self edge first
    @inbounds senders[oo]   = i32
    @inbounds receivers[oo] = i32
    @inbounds rd[1, oo] = 0f0; @inbounds rd[2, oo] = 0f0; @inbounds rd[3, oo] = 0f0
    @inbounds rdn[oo] = 0f0
    oo += Int32(1)

    px = @inbounds coords_orig[1, i]
    py = @inbounds coords_orig[2, i]
    pz = @inbounds coords_orig[3, i]
    inv_r = 1f0 / radius

    @for_each_neighbor_device_inline dv i j begin
        if j != i32
            qx = @inbounds coords_orig[1, j]
            qy = @inbounds coords_orig[2, j]
            qz = @inbounds coords_orig[3, j]
            dx = qx - px; dy = qy - py; dz = qz - pz
            d2 = dx*dx + dy*dy + dz*dz
            @inbounds senders[oo]   = j
            @inbounds receivers[oo] = i32
            @inbounds rd[1, oo] = dx * inv_r
            @inbounds rd[2, oo] = dy * inv_r
            @inbounds rd[3, oo] = dz * inv_r
            @inbounds rdn[oo] = sqrt(d2) * inv_r
            oo += Int32(1)
        end
    end

    @inbounds cursors[i] = oo
    return nothing
end

function write_kernel_2d!(senders, receivers, rd, rdn, cursors, dv, coords_orig, N, radius)
    # 2D coords are padded to 3D for the v0.1 path; we pass `coords_orig`
    # (the (2,N) original) so feature dx/dy match the user's request.
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > N && return nothing
    i32 = Int32(i)

    @inbounds oo = cursors[i]

    @inbounds senders[oo]   = i32
    @inbounds receivers[oo] = i32
    @inbounds rd[1, oo] = 0f0; @inbounds rd[2, oo] = 0f0
    @inbounds rdn[oo] = 0f0
    oo += Int32(1)

    px = @inbounds coords_orig[1, i]
    py = @inbounds coords_orig[2, i]
    inv_r = 1f0 / radius

    # The macro tests distance against `dv.coords_t` (the padded-to-3D view
    # of the input). Because z is always 0 in the padded copy, 3D distance
    # equals 2D distance, so the filter is exact.
    @for_each_neighbor_device_inline dv i j begin
        if j != i32
            qx = @inbounds coords_orig[1, j]
            qy = @inbounds coords_orig[2, j]
            dx = qx - px; dy = qy - py
            d2 = dx*dx + dy*dy
            @inbounds senders[oo]   = j
            @inbounds receivers[oo] = i32
            @inbounds rd[1, oo] = dx * inv_r
            @inbounds rd[2, oo] = dy * inv_r
            @inbounds rdn[oo] = sqrt(d2) * inv_r
            oo += Int32(1)
        end
    end

    @inbounds cursors[i] = oo
    return nothing
end

function tns_point_neighbor_ns_gpu(pos_orig::CuArray{Float32}, radius::Float32)
    D = size(pos_orig, 1)
    N = size(pos_orig, 2)

    # Octopus v0.1 is 3D-only. For D=2 we pad to 3D (z=0) for the tree;
    # rel_displacement and rel_dist_norm are still computed in D dimensions.
    pos3 = if D == 3
        pos_orig
    else
        p = CUDA.zeros(Float32, 3, N)
        p[1:D, :] .= pos_orig
        p
    end

    tns = TNS(Float32; ndims = 3)
    set_search_radius!(tns, radius)
    id = add_point_set!(tns, pos3)
    set_active_search!(tns, id, id)
    run!(tns)

    dv = device_view(tns, id, id)

    # Pass A — counts
    counts = CUDA.zeros(Int32, N)
    threads = 128
    blocks = cld(N, threads)
    @cuda threads = threads blocks = blocks count_kernel!(counts, dv, Int32(N))
    CUDA.synchronize()

    # Offsets: exclusive prefix-sum → first slot per point is 1-based.
    offsets = CUDA.cumsum(counts) .- counts .+ Int32(1)
    n_edges = Int(CUDA.@allowscalar offsets[end] + counts[end] - 1)

    senders = CuArray{Int32}(undef, n_edges)
    receivers = CuArray{Int32}(undef, n_edges)
    rd  = CuArray{Float32}(undef, D, n_edges)
    rdn = CuArray{Float32}(undef, 1, n_edges)

    # write kernels mutate cursors — copy offsets so we keep them intact.
    cursors = copy(offsets)
    if D == 3
        @cuda threads = threads blocks = blocks write_kernel_3d!(
            senders, receivers, rd, rdn, cursors, dv, pos_orig, Int32(N), radius)
    else
        @cuda threads = threads blocks = blocks write_kernel_2d!(
            senders, receivers, rd, rdn, cursors, dv, pos_orig, Int32(N), radius)
    end
    CUDA.synchronize()
    return senders, receivers, rd, rdn, tns
end

# ------------------------------------------------------------------------
# Correctness
# ------------------------------------------------------------------------

function edge_set_gpu(senders::CuVector{Int32}, receivers::CuVector{Int32})
    s = Array(senders); r = Array(receivers)
    out = Set{Tuple{Int32, Int32}}()
    @inbounds for k in eachindex(s)
        push!(out, (r[k], s[k]))
    end
    return out
end

# ------------------------------------------------------------------------
# Benchmark helpers
# ------------------------------------------------------------------------

function fmt_time(ns)
    if ns < 1e6
        return @sprintf("%7.2f μs", ns/1e3)
    elseif ns < 1e9
        return @sprintf("%7.2f ms", ns/1e6)
    else
        return @sprintf("%7.2f  s", ns/1e9)
    end
end

fmt_bytes(b) = b < 2^20 ? @sprintf("%7.2f KB", b/1024) :
               b < 2^30 ? @sprintf("%7.2f MB", b/2^20) :
                          @sprintf("%7.2f GB", b/2^30)

# Block until GPU finishes so the benchmark counts GPU work, not queue.
wrap_sync(f) = () -> (f(); CUDA.synchronize(); nothing)

function gpu_alloc_bytes(nhs)
    # Count only device-resident arrays — summarysize on adapted CPU+GPU
    # structs double-counts host-side wrappers. For PN we walk the adapted
    # struct; for Octopus we walk the trees + perm + morton + stacks.
    return Base.summarysize(nhs)
end

# ------------------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------------------

struct Scenario
    name::String
    pos::Matrix{Float32}    # host original; benchmark uploads once per run
    radius::Float32
end

function build_scenarios()
    scenarios = Scenario[]

    Random.seed!(42)
    for N in (1_000, 5_000, 20_000)
        pos = Matrix{Float32}(undef, 2, N)
        pos[1, :] = 0.3f0  .* rand(Float32, N)
        pos[2, :] = 0.15f0 .* rand(Float32, N)
        push!(scenarios, Scenario("2D dam-break-like  N=$(lpad(N,6))", pos, 0.072f0))
    end

    Random.seed!(43)
    for (N, r) in ((5_000, 0.05f0), (50_000, 0.02f0), (200_000, 0.01f0))
        pos = rand(Float32, 3, N)
        push!(scenarios, Scenario("3D uniform         N=$(lpad(N,6))", pos, r))
    end

    return scenarios
end

# ------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------

function run_benchmark()
    println("\nGraphNetSim GPU neighborhood-search benchmark")
    println("Octopus.jl vs PointNeighbors.jl (matches `point_neighbor_ns(::CuArray)`)")
    println("Julia: ", VERSION, "  threads: ", Threads.nthreads())
    println("GPU:   ", name(CUDA.device()))
    println()

    for sc in build_scenarios()
        N = size(sc.pos, 2)
        D = size(sc.pos, 1)
        gpu_pos = CuArray(sc.pos)

        println("=" ^ 100)
        println(sc.name, "   (D=", D, ", r=", sc.radius, ")")
        println("-" ^ 100)

        # Correctness — run both once and diff edge sets.
        sp, rp, _, _, nhs_pn = pn_point_neighbor_ns_gpu(gpu_pos, sc.radius)
        st, rt, _, _, tns   = tns_point_neighbor_ns_gpu(gpu_pos, sc.radius)
        set_pn = edge_set_gpu(sp, rp)
        set_tns = edge_set_gpu(st, rt)
        match = (set_pn == set_tns)
        ok_str = match ? "yes" : "NO ($(length(symdiff(set_pn, set_tns))) symdiff)"
        @printf("  edges (PN / TNS): %s / %s   identical: %s\n",
                length(sp), length(st), ok_str)

        # GPU memory overhead — device-resident bytes only. `Base.summarysize`
        # doesn't recurse into CuArrays, so we walk the structs explicitly.

        tree = tns.trees[1]
        tns_dev_bytes = sizeof(tree.node_bounds_min) +
                        sizeof(tree.node_bounds_max) +
                        sizeof(tree.node_first) +
                        sizeof(tree.node_last) +
                        sizeof(tree.node_children) +
                        sizeof(tns.permutation[1])

        # PN's adapted nhs_gpu wraps a FullGridCellList whose cells field is a
        # `DynamicVectorOfVectors{Int32, CuMatrix{Int32}}` — dense per-cell
        # slab (max_points_per_cell × n_cells). Also a CuVector{Int32} lengths
        # + a CuVector{Int32} for cell membership (used by update!).
        cells = nhs_pn.cell_list.cells
        pn_dev_bytes = sizeof(cells.backend) + sizeof(cells.lengths)
        # update_buffer + cell_index fields on the NHS itself (also CuArrays)
        if hasfield(typeof(nhs_pn), :cell_index)
            pn_dev_bytes += sizeof(nhs_pn.cell_index)
        end
        if hasfield(typeof(nhs_pn), :update_buffer)
            pn_dev_bytes += sizeof(nhs_pn.update_buffer)
        end

        # Free one scenario's intermediates to avoid fragmentation.
        sp = rp = st = rt = nothing
        CUDA.reclaim()

        # Benchmark both. wrap_sync ensures the time includes CUDA work, not
        # just queue submission.
        b_pn  = @benchmark (pn_point_neighbor_ns_gpu($gpu_pos, $sc.radius); CUDA.synchronize()) samples=5 evals=1 seconds=30
        b_tns = @benchmark (tns_point_neighbor_ns_gpu($gpu_pos, $sc.radius); CUDA.synchronize()) samples=5 evals=1 seconds=30

        println()
        @printf("%-16s %15s   device overhead: %s\n",
                "PointNeighbors", fmt_time(median(b_pn.times)), fmt_bytes(pn_dev_bytes))
        @printf("%-16s %15s   device overhead: %s\n",
                "Octopus",    fmt_time(median(b_tns.times)), fmt_bytes(tns_dev_bytes))

        spd = median(b_pn.times) / median(b_tns.times)
        memx = pn_dev_bytes / max(tns_dev_bytes, 1)
        @printf("%-16s speedup: %.2fx   memory factor: %.2fx leaner\n",
                "TNS vs PN", spd, memx)

        CUDA.reclaim()
        println()
    end
end

run_benchmark()
