#
# Copyright (c) 2026 Josef Jouaux, Chair of Mechatronics, University of Augsburg
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# Benchmark matching GraphNetSim.jl's `point_neighbor_ns` usage pattern:
#   build nhs -> count neighbors per particle -> allocate edge arrays
#   -> populate senders/receivers/rel_displacement/rel_dist_norm.
# Called once per ODE step in training/inference, so what matters is the total
# per-call wall time, not build-then-many-queries.
#
# See reference: GraphNetSim.jl/src/graph.jl:378-414
#
# Run:
#   julia --project=benchmarks --threads=auto benchmarks/vs_graphnetsim.jl

using Octopus
using PointNeighbors
using BenchmarkTools
using Random
using Printf
using Statistics

# ------------------------------------------------------------------------
# Reference implementation — verbatim from GraphNetSim/src/graph.jl:378.
# ------------------------------------------------------------------------

function pn_point_neighbor_ns(pos::Matrix{Float32}, radius::Float32)
    system = pos
    min_corner = minimum(pos; dims=2)
    max_corner = maximum(pos; dims=2)
    # FullGridCellList default max_points_per_cell=100 overflows on uniform
    # random clouds at these densities. Estimate a sane upper bound so
    # initialize! doesn't throw BoundsError.
    D = size(pos, 1)
    N = size(pos, 2)
    cell_vol = prod(vec(max_corner - min_corner)) / prod(vec(ceil.((max_corner - min_corner) / radius)))
    avg_per_cell = N * cell_vol / max(prod(vec(max_corner - min_corner)), eps(Float32))
    # 4x average + 32 floor; safely covers non-uniform clusters at r-scale.
    mppc = max(Int(ceil(4 * avg_per_cell)) + 32, 100)
    nhs = GridNeighborhoodSearch{D}(;
        search_radius = radius,
        n_points = N,
        cell_list = FullGridCellList(; min_corner, max_corner,
                                     search_radius = radius,
                                     max_points_per_cell = mppc),
    )
    initialize!(nhs, system, pos)

    n_neighbors = zeros(Int, size(pos, 2))
    foreach_point_neighbor(system, pos, nhs) do i, _, _, _
        n_neighbors[i] += 1
    end

    n_edges = sum(n_neighbors)
    senders = Vector{Int32}(undef, n_edges)
    receivers = Vector{Int32}(undef, n_edges)
    rel_displacement = Array{Float32}(undef, size(pos, 1), n_edges)
    rel_dist_norm = Array{Float32}(undef, 1, n_edges)

    offset = cumsum(n_neighbors) .- n_neighbors .+ 1
    foreach_point_neighbor(system, pos, nhs) do i, j, pos_diff, distance
        receivers[offset[i]] = i
        senders[offset[i]] = j
        @inbounds for d in 1:size(pos, 1)
            rel_displacement[d, offset[i]] = pos_diff[d] / radius
        end
        rel_dist_norm[offset[i]] = distance / radius
        offset[i] += 1
    end

    return senders, receivers, rel_displacement, rel_dist_norm, nhs
end

# ------------------------------------------------------------------------
# Octopus port of the same operation.
# Octopus v0.1 is 3D-only — 2D coords are padded to 3D (z = 0) before
# construction; edges are recomputed in the original D-dimensional space so
# the output's rel_displacement matches what GraphNetSim expects.
# ------------------------------------------------------------------------

function tns_point_neighbor_ns(pos::Matrix{Float32}, radius::Float32)
    D = size(pos, 1)
    n = size(pos, 2)

    # Pad to 3D if needed. Separate pos3 backing so the tree sees 3D.
    pos3 = if D == 3
        pos
    else
        p = zeros(Float32, 3, n)
        @inbounds for k in 1:n
            for d in 1:D
                p[d, k] = pos[d, k]
            end
        end
        p
    end

    tns = TNS(Float32; ndims = 3)
    set_search_radius!(tns, radius)
    id = add_point_set!(tns, pos3)
    set_active_search!(tns, id, id)
    run!(tns)

    # GraphNetSim's `point_neighbor_ns` callback counts the self-pair (i,i)
    # because PN's foreach_point_neighbor emits it (distance=0 ≤ r). We match
    # that semantics here so the edge set is identical.
    n_neighbors = zeros(Int, n)
    Threads.@threads for i in 1:n
        c = 1  # include self
        for_each_neighbor(tns, id, id, i) do j
            j != Int32(i) && (c += 1)
        end
        n_neighbors[i] = c
    end

    n_edges = sum(n_neighbors)
    senders = Vector{Int32}(undef, n_edges)
    receivers = Vector{Int32}(undef, n_edges)
    rel_displacement = Array{Float32}(undef, D, n_edges)
    rel_dist_norm = Array{Float32}(undef, 1, n_edges)

    offset = cumsum(n_neighbors) .- n_neighbors .+ 1
    # Pass B — populate. Each thread writes to a disjoint slice via offset[i].
    Threads.@threads for i in 1:n
        i32 = Int32(i)
        local_off = offset[i]
        # Emit self-edge first to match GraphNetSim.
        receivers[local_off] = i32
        senders[local_off] = i32
        @inbounds for d in 1:D
            rel_displacement[d, local_off] = 0f0
        end
        rel_dist_norm[local_off] = 0f0
        local_off += 1

        for_each_neighbor(tns, id, id, i) do j
            if j != i32
                receivers[local_off] = i32
                senders[local_off] = j
                d_sq = 0f0
                @inbounds for d in 1:D
                    δ = pos[d, i] - pos[d, j]
                    rel_displacement[d, local_off] = δ / radius
                    d_sq += δ * δ
                end
                rel_dist_norm[local_off] = sqrt(d_sq) / radius
                local_off += 1
            end
        end
    end

    return senders, receivers, rel_displacement, rel_dist_norm, tns
end

# ------------------------------------------------------------------------
# Correctness check: PN's output order depends on iteration; ours may differ.
# We normalize both to sets of (receiver, sender) pairs and compare.
# ------------------------------------------------------------------------

function edge_set(senders::Vector{Int32}, receivers::Vector{Int32})
    s = Set{Tuple{Int32, Int32}}()
    @inbounds for k in eachindex(senders)
        push!(s, (receivers[k], senders[k]))
    end
    return s
end

# ------------------------------------------------------------------------
# Scenarios modelled on GraphNetSim dam-break + 3D SPH ranges.
# Meta.json for dam_break 2D: radius ≈ 0.072, bounds [0,0.3]×[0,0.15].
# 3D SPH typical: cell_size ≈ radius ≈ 0.01–0.03, unit cube.
# ------------------------------------------------------------------------

struct Scenario
    name::String
    pos::Matrix{Float32}
    radius::Float32
end

function build_scenarios()
    scenarios = Scenario[]

    # --- 2D dam-break scale (padded to 3D for TNS) ---
    Random.seed!(42)
    for N in (1_000, 5_000, 20_000)
        # Uniform over [0,0.3] × [0,0.15] (2D bounding box from dam_break meta)
        pos = Matrix{Float32}(undef, 2, N)
        pos[1, :] = 0.3f0 .* rand(Float32, N)
        pos[2, :] = 0.15f0 .* rand(Float32, N)
        push!(scenarios, Scenario("2D dam-break-like  N=$(lpad(N,6))", pos, 0.072f0))
    end

    # --- 3D typical SPH scale ---
    Random.seed!(43)
    for (N, r) in ((5_000, 0.05f0), (50_000, 0.02f0), (200_000, 0.01f0))
        pos = rand(Float32, 3, N)
        push!(scenarios, Scenario("3D uniform         N=$(lpad(N,6))", pos, r))
    end

    return scenarios
end

# ------------------------------------------------------------------------
# Run once for correctness + per-call timing.
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

fmt_bytes(b) = b < 2^20 ? @sprintf("%7.2f KB", b/1024) : @sprintf("%7.2f MB", b/2^20)

function run_benchmark()
    println("\nGraphNetSim neighborhood-search benchmark")
    println("Octopus.jl vs PointNeighbors.jl, matching `point_neighbor_ns`")
    println("Julia: ", VERSION, "  threads: ", Threads.nthreads())
    println()

    for sc in build_scenarios()
        N = size(sc.pos, 2)
        D = size(sc.pos, 1)

        println("=" ^ 100)
        println(sc.name, "   (D=", D, ", r=", sc.radius, ")")
        println("-" ^ 100)

        # Warmup + correctness
        senders_pn, receivers_pn, rd_pn, rdn_pn, nhs_pn = pn_point_neighbor_ns(sc.pos, sc.radius)
        senders_tns, receivers_tns, rd_tns, rdn_tns, tns = tns_point_neighbor_ns(sc.pos, sc.radius)

        set_pn  = edge_set(senders_pn, receivers_pn)
        set_tns = edge_set(senders_tns, receivers_tns)
        n_edges_pn = length(senders_pn)
        n_edges_tns = length(senders_tns)
        match = (set_pn == set_tns)
        ok_str = match ? "yes" : "NO ($(length(symdiff(set_pn, set_tns))) symdiff)"

        @printf("  edges (PN / TNS): %s / %s   identical: %s\n",
                n_edges_pn, n_edges_tns, ok_str)
        println()

        # Benchmark
        b_pn  = @benchmark pn_point_neighbor_ns($sc.pos, $sc.radius)  samples=5 evals=1 seconds=30
        b_tns = @benchmark tns_point_neighbor_ns($sc.pos, $sc.radius) samples=5 evals=1 seconds=30

        # Memory — final nhs structure overhead.
        nhs_bytes  = Base.summarysize(nhs_pn)  - Base.summarysize(sc.pos)
        tns_bytes  = Base.summarysize(tns)     - Base.summarysize(sc.pos)

        @printf("%-16s %15s %15s   memory overhead: %s\n",
                "PointNeighbors", fmt_time(median(b_pn.times)),
                fmt_bytes(median(b_pn.memory)), fmt_bytes(nhs_bytes))
        @printf("%-16s %15s %15s   memory overhead: %s\n",
                "Octopus", fmt_time(median(b_tns.times)),
                fmt_bytes(median(b_tns.memory)), fmt_bytes(tns_bytes))

        # Derived ratios
        spd = median(b_pn.times) / median(b_tns.times)
        mem_ratio = nhs_bytes / max(tns_bytes, 1)
        @printf("%-16s speedup: %.2fx   memory factor: %.2fx leaner\n",
                "TNS vs PN", spd, mem_ratio)
        println()
    end
end

run_benchmark()
