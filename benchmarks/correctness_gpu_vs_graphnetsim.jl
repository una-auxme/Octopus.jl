#
# Copyright (c) 2026 Josef Jouaux
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# GPU correctness + perf check: Octopus vs GraphNetSim's
# `point_neighbor_ns(::CuArray, ...)` (the only TNS-relevant moving part inside
# `build_graph`).
#
# Reference: `point_neighbor_ns(::CuArray, ::Float32)` from
# GraphNetSim.jl/src/graph.jl:319-360, with the same FullGridCellList capacity
# bump used in the CPU check (default 100 overflows on uniform clouds).
#
# Calls Octopus's *public* CUDA `build_edges` directly. The CUDA edge
# kernels were patched to use the inline-traversal macros (no closure-captured
# locals); `build_edges` is now the production entry point a port of
# `build_graph` would call.
#
# Run:
#   julia --project=benchmarks --threads=auto benchmarks/correctness_gpu_vs_graphnetsim.jl

using Octopus
using PointNeighbors
using CUDA
using Adapt
using BenchmarkTools
using Random
using Printf
using Statistics

CUDA.functional() || error("CUDA not functional; cannot run GPU benchmark")
println("GPU: ", name(CUDA.device()))

# -----------------------------------------------------------------------------
# Reference: GraphNetSim's GPU `point_neighbor_ns`
# -----------------------------------------------------------------------------

function pn_point_neighbor_ns_gpu(pos::CuArray{Float32}, radius::Float32)
    system = pos
    min_corner = minimum(pos; dims = 2)
    max_corner = maximum(pos; dims = 2)
    D = size(pos, 1)
    N = size(pos, 2)

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
    return (; senders, receivers, rel_displacement, rel_dist_norm, nhs = nhs_gpu)
end

# -----------------------------------------------------------------------------
# Octopus: native CUDA `build_edges`. This is the production path that a
# port of `build_graph` would call. Excludes the self-edge for symmetric
# search; PN's reference includes it, so we strip self from PN before joining.
# -----------------------------------------------------------------------------

function tns_build_edges_gpu(pos::CuArray{Float32}, radius::Float32)
    D = size(pos, 1)
    tns = TNS(Float32; ndims = D)
    set_search_radius!(tns, radius)
    id = add_point_set!(tns, pos)
    set_active_search!(tns, id, id)
    run!(tns)
    out = build_edges(tns, id, id)
    CUDA.synchronize()
    return (; senders = out.senders, receivers = out.receivers,
             rel_displacement = out.rel_displacement,
             rel_dist_norm = out.rel_dist_norm, tns)
end

# -----------------------------------------------------------------------------
# Comparison helpers
# -----------------------------------------------------------------------------

function strip_self_pair(senders, receivers, rdisp, rdist)
    keep = senders .!= receivers
    return senders[keep], receivers[keep], rdisp[:, keep], rdist[:, keep]
end

function sort_by_recv_send(senders, receivers, rdisp, rdist)
    perm = sortperm(collect(zip(Int.(receivers), Int.(senders))))
    return senders[perm], receivers[perm], rdisp[:, perm], rdist[:, perm]
end

function compare(label, pos_h, radius)
    println("─" ^ 92)
    @printf("%s   D=%d  N=%d  r=%.4f\n", label, size(pos_h, 1), size(pos_h, 2), radius)

    pos_d = CuArray(pos_h)
    pn = pn_point_neighbor_ns_gpu(pos_d, radius)
    tn = tns_build_edges_gpu(pos_d, radius)

    s_pn = Array(pn.senders);  r_pn = Array(pn.receivers)
    d_pn = Array(pn.rel_displacement); n_pn = Array(pn.rel_dist_norm)
    s_t  = Array(tn.senders);  r_t  = Array(tn.receivers)
    d_t  = Array(tn.rel_displacement); n_t = Array(tn.rel_dist_norm)

    n_pts = size(pos_h, 2)
    pn_self = sum(s_pn .== r_pn)
    @printf("  edges:        PN=%d  TNS=%d  Δ=%d  expected=%d (PN self=%d)\n",
            length(s_pn), length(s_t), length(s_pn) - length(s_t), n_pts, pn_self)

    # Strip PN's self pair so both sides have only j != i edges.
    s_pn2, r_pn2, d_pn2, n_pn2 = strip_self_pair(s_pn, r_pn, d_pn, n_pn)

    if length(s_pn2) != length(s_t)
        @printf("  ✗ edge counts differ after self-strip: PN=%d  TNS=%d\n",
                length(s_pn2), length(s_t))
        return (; pn, tn, ok = false)
    end

    s_pn3, r_pn3, d_pn3, n_pn3 = sort_by_recv_send(s_pn2, r_pn2, d_pn2, n_pn2)
    s_t3,  r_t3,  d_t3,  n_t3  = sort_by_recv_send(s_t,  r_t,  d_t,  n_t)

    eq_pairs = all(s_pn3 .== s_t3) && all(r_pn3 .== r_t3)
    if !eq_pairs
        set_pn = Set(zip(Int.(r_pn3), Int.(s_pn3)))
        set_t  = Set(zip(Int.(r_t3),  Int.(s_t3)))
        sd = symdiff(set_pn, set_t)
        @printf("  ✗ edge pairs differ after sort: %d in symdiff\n", length(sd))
        return (; pn, tn, ok = false)
    end

    abs_disp = maximum(abs.(d_pn3 .- d_t3))
    abs_dist = maximum(abs.(n_pn3 .- n_t3))
    tol = 1f-5
    ok = (abs_disp <= tol) && (abs_dist <= tol)
    @printf("  ✓ same edges, same order after sort\n")
    @printf("  rel_displacement |Δ|max = %.3e\n", abs_disp)
    @printf("  rel_dist_norm    |Δ|max = %.3e\n", abs_dist)
    @printf("  → %s (tol=%.0e)\n", ok ? "PASS" : "FAIL", tol)

    return (; pn, tn, ok)
end

# -----------------------------------------------------------------------------
# Memory accounting (device-resident bytes only).
# -----------------------------------------------------------------------------

function tns_device_bytes(tns)
    tree = tns.trees[1]
    return sizeof(tree.node_bounds_min) +
           sizeof(tree.node_bounds_max) +
           sizeof(tree.node_first) +
           sizeof(tree.node_last) +
           sizeof(tree.node_children) +
           sizeof(tns.permutation[1])
end

function pn_device_bytes(nhs)
    cells = nhs.cell_list.cells
    bytes = sizeof(cells.backend) + sizeof(cells.lengths)
    if hasfield(typeof(nhs), :cell_index)
        bytes += sizeof(nhs.cell_index)
    end
    if hasfield(typeof(nhs), :update_buffer)
        bytes += sizeof(nhs.update_buffer)
    end
    return bytes
end

fmt_time(ns) = ns < 1e6 ? @sprintf("%7.2f μs", ns / 1e3) :
               ns < 1e9 ? @sprintf("%7.2f ms", ns / 1e6) :
                          @sprintf("%7.2f  s", ns / 1e9)
fmt_bytes(b) = b < 2^20 ? @sprintf("%7.2f KB", b / 1024) :
               b < 2^30 ? @sprintf("%7.2f MB", b / 2^20) :
                          @sprintf("%7.2f GB", b / 2^30)

function benchmark_one(pos_h, radius)
    pos_d = CuArray(pos_h)
    CUDA.synchronize()

    # 3 samples (instead of 5) keeps live device memory under control on dense
    # 2D scenarios; we still get a reasonable median.
    b_pn = @benchmark begin
        r = pn_point_neighbor_ns_gpu($pos_d, $radius)
        CUDA.synchronize()
        r = nothing
        GC.gc(true)
        CUDA.reclaim()
    end samples = 3 evals = 1 seconds = 60
    GC.gc(true); CUDA.reclaim()

    b_tn = @benchmark begin
        r = tns_build_edges_gpu($pos_d, $radius)
        CUDA.synchronize()
        r = nothing
        GC.gc(true)
        CUDA.reclaim()
    end samples = 3 evals = 1 seconds = 60
    GC.gc(true); CUDA.reclaim()

    return b_pn, b_tn
end

# -----------------------------------------------------------------------------
# Scenarios
# -----------------------------------------------------------------------------

function build_scenarios()
    scenarios = Tuple{String, Matrix{Float32}, Float32}[]

    # 2D dam-break-like at the published radius (0.072) with N≥10k blows past
    # 24 GB of device memory because the cloud is dense and each particle has
    # ~2.5k neighbors → ~25M edges × (sender+receiver+displacement+distance)
    # ≈ half a GB per pass, and BenchmarkTools holds 5 samples. Stick to N=1k
    # for the correctness/perf check and use the N=5k 3D case as the larger
    # 2D-equivalent — same per-particle neighbor count, different geometry.
    Random.seed!(42)
    for N in (1_000, 5_000)
        pos = Matrix{Float32}(undef, 2, N)
        pos[1, :] = 0.3f0 .* rand(Float32, N)
        pos[2, :] = 0.15f0 .* rand(Float32, N)
        push!(scenarios, ("2D dam-break-like  N=$(lpad(N, 6))", pos, 0.072f0))
    end

    Random.seed!(43)
    for (N, r) in ((5_000, 0.05f0), (50_000, 0.02f0), (200_000, 0.01f0))
        pos = rand(Float32, 3, N)
        push!(scenarios, ("3D uniform         N=$(lpad(N, 6))", pos, r))
    end

    return scenarios
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function run_all()
    println("\nGPU correctness + perf: Octopus vs GraphNetSim.point_neighbor_ns(::CuArray)")
    println("Julia ", VERSION, "  threads=", Threads.nthreads())
    println()

    for (label, pos_h, radius) in build_scenarios()
        result = compare(label, pos_h, radius)

        tns_b = tns_device_bytes(result.tn.tns)
        pn_b  = pn_device_bytes(result.pn.nhs)

        result = nothing
        GC.gc(true); GC.gc(true)
        CUDA.reclaim()

        b_pn, b_tn = benchmark_one(pos_h, radius)

        @printf("  PointNeighbors  median: %s   device overhead: %s\n",
                fmt_time(median(b_pn.times)), fmt_bytes(pn_b))
        @printf("  Octopus     median: %s   device overhead: %s\n",
                fmt_time(median(b_tn.times)), fmt_bytes(tns_b))
        spd = median(b_pn.times) / median(b_tn.times)
        memx = pn_b / max(tns_b, 1)
        @printf("  TNS vs PN       speedup: %.2fx   memory factor: %.2fx leaner\n",
                spd, memx)
        CUDA.reclaim()
        println()
    end
    println("─" ^ 92)
end

run_all()
