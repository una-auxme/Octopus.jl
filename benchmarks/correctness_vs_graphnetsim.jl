#
# Copyright (c) 2026 Josef Jouaux, Chair of Mechatronics, University of Augsburg
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# Correctness check: Octopus.jl vs GraphNetSim.jl's `build_graph` neighbor
# search core (`point_neighbor_ns`, which is the only TNS-relevant moving part
# inside build_graph — the rest is feature normalization).
#
# `point_neighbor_ns` is reproduced verbatim from GraphNetSim.jl/src/graph.jl
# (CPU branch). The reference returns
#   (senders, receivers, rel_displacement, rel_dist_norm)
# with PointNeighbors' convention:
#   pos_diff   = pos[:,i] - pos[:,j]    (i = receiver, j = sender)
#   rel_disp   = pos_diff / radius
#   rel_dist   = ‖pos_diff‖ / radius
#
# We compare against Octopus.build_edges, which produces the same fields
# with the same sign convention but excludes the self pair (j == i). To get
# bitwise-comparable outputs, we strip the self pair from the PN reference
# before joining on (sender, receiver) and comparing rel_disp / rel_dist.
#
# Run:
#   julia --project=benchmarks --threads=auto benchmarks/correctness_vs_graphnetsim.jl

using Octopus
using PointNeighbors
using Random
using Printf

# -----------------------------------------------------------------------------
# Reference: GraphNetSim.jl/src/graph.jl:378-414 (CPU). The only modification
# is the `max_points_per_cell` upper-bound: GraphNetSim's call uses the default
# (=100) which overflows on uniform random clouds. The cell-list overflow is a
# property of FullGridCellList capacity, not of the neighbor-search semantics,
# so bumping it preserves the reference's semantics while letting it run.
# -----------------------------------------------------------------------------

function pn_point_neighbor_ns(pos::Matrix{Float32}, radius::Float32)
    system = pos
    min_corner = minimum(pos; dims=2)
    max_corner = maximum(pos; dims=2)
    D = size(pos, 1)
    N = size(pos, 2)
    cell_vol = prod(vec(max_corner - min_corner)) /
               prod(vec(ceil.((max_corner - min_corner) / radius)))
    avg_per_cell = N * cell_vol / max(prod(vec(max_corner - min_corner)), eps(Float32))
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

    return senders, receivers, rel_displacement, rel_dist_norm
end

# -----------------------------------------------------------------------------
# Octopus run, built on the package's `build_edges` (the *production* API
# that `build_graph` would replace `point_neighbor_ns` with).
# -----------------------------------------------------------------------------

function tns_build_edges(pos::Matrix{Float32}, radius::Float32)
    D = size(pos, 1)
    n = size(pos, 2)

    pos3 = if D == 3
        pos
    else
        # TNS v0.1: the TNS object can be 2D, but its API expects (NDIMS, N).
        # We use ndims=D so build_edges runs in the native dimension and the
        # returned rel_displacement is D-dimensional (matches PN).
        pos
    end

    tns = TNS(Float32; ndims = D)
    set_search_radius!(tns, radius)
    id = add_point_set!(tns, pos3)
    set_active_search!(tns, id, id)
    run!(tns)
    out = build_edges(tns, id, id)
    return out.senders, out.receivers, out.rel_displacement, out.rel_dist_norm
end

# -----------------------------------------------------------------------------
# Comparison: drop self pair from PN, sort both edge lists by (receiver, sender)
# canonical key, then compare componentwise.
# -----------------------------------------------------------------------------

function strip_self_pair(senders, receivers, rdisp, rdist)
    keep = senders .!= receivers
    return senders[keep], receivers[keep], rdisp[:, keep], rdist[:, keep]
end

function sort_edges(senders, receivers, rdisp, rdist)
    # Stable sort key: (receiver, sender). Within a fixed receiver, neighbor
    # order is implementation-dependent; sorting by sender canonicalizes it.
    n = length(senders)
    perm = sortperm(collect(zip(Int.(receivers), Int.(senders))))
    return senders[perm], receivers[perm], rdisp[:, perm], rdist[:, perm]
end

function compare(label, pos, radius)
    println("─" ^ 88)
    @printf("%s   D=%d  N=%d  r=%.4f\n", label, size(pos, 1), size(pos, 2), radius)

    s_pn, r_pn, d_pn, n_pn = pn_point_neighbor_ns(pos, radius)
    s_tns, r_tns, d_tns, n_tns = tns_build_edges(pos, radius)

    # Sanity: PN should have N more edges than TNS (one self pair per point).
    n_pts = size(pos, 2)
    expected_diff = n_pts
    actual_diff = length(s_pn) - length(s_tns)
    pn_self_count = sum(s_pn .== r_pn)
    @printf("  edges:        PN=%d  TNS=%d  Δ=%d  expected=%d (self pairs in PN: %d)\n",
            length(s_pn), length(s_tns), actual_diff, expected_diff, pn_self_count)

    # Strip self from PN and sort both sides.
    s_pn2, r_pn2, d_pn2, n_pn2 = strip_self_pair(s_pn, r_pn, d_pn, n_pn)
    s_pn3, r_pn3, d_pn3, n_pn3 = sort_edges(s_pn2, r_pn2, d_pn2, n_pn2)
    s_t3,  r_t3,  d_t3,  n_t3  = sort_edges(s_tns, r_tns, d_tns, n_tns)

    eq_count = (length(s_pn3) == length(s_t3))
    eq_pairs = eq_count && all(s_pn3 .== s_t3) && all(r_pn3 .== r_t3)
    if !eq_count
        @printf("  edge counts differ after self-strip: PN=%d  TNS=%d\n",
                length(s_pn3), length(s_t3))
        return
    end

    if !eq_pairs
        # Fall back to set comparison.
        set_pn = Set(zip(Int.(r_pn3), Int.(s_pn3)))
        set_t  = Set(zip(Int.(r_t3),  Int.(s_t3)))
        sd = symdiff(set_pn, set_t)
        @printf("  ✗ edge pairs differ even after sort: %d in symdiff\n", length(sd))
        if !isempty(sd)
            for (k, e) in enumerate(collect(sd))
                k > 5 && break
                @printf("    sample: (recv=%d, send=%d)\n", e[1], e[2])
            end
        end
        return
    end

    # Per-edge value comparison. PN's pos_diff and TNS's rel_displacement are
    # both float32; both compute distance via dot+sqrt so we expect bitwise or
    # 1-ulp agreement on uniform random clouds. Use a tiny relative tolerance.
    abs_disp = maximum(abs.(d_pn3 .- d_t3))
    abs_dist = maximum(abs.(n_pn3 .- n_t3))
    rel_disp_max = abs_disp                           # values are O(1)/r-scaled
    rel_dist_max = abs_dist
    @printf("  ✓ same edges, same order after sort\n")
    @printf("  rel_displacement |Δ|max = %.3e\n", abs_disp)
    @printf("  rel_dist_norm    |Δ|max = %.3e\n", abs_dist)

    tol = 1f-5
    ok_disp = abs_disp <= tol
    ok_dist = abs_dist <= tol
    overall = ok_disp && ok_dist
    @printf("  → %s (tol=%.0e)\n", overall ? "PASS" : "FAIL", tol)
end

# -----------------------------------------------------------------------------
# Scenarios.
# -----------------------------------------------------------------------------

function main()
    println("Correctness: Octopus.build_edges vs GraphNetSim.point_neighbor_ns")
    println("Julia ", VERSION, "  threads=", Threads.nthreads())

    # --- 2D dam-break-like ---
    Random.seed!(42)
    for N in (1_000, 10_000)
        pos = Matrix{Float32}(undef, 2, N)
        pos[1, :] = 0.3f0 .* rand(Float32, N)
        pos[2, :] = 0.15f0 .* rand(Float32, N)
        compare(@sprintf("2D uniform     N=%d", N), pos, 0.072f0)
    end

    # --- 3D uniform random ---
    Random.seed!(43)
    for (N, r) in ((5_000, 0.05f0), (50_000, 0.02f0))
        pos = rand(Float32, 3, N)
        compare(@sprintf("3D uniform     N=%d", N), pos, r)
    end

    # --- 3D structured grid (regular spacing — adversarial for tie-breaks) ---
    let
        n = 25
        coords = Vector{Float32}[]
        for i in 0:n-1, j in 0:n-1, k in 0:n-1
            push!(coords, Float32[i, j, k] .* 0.04f0)
        end
        pos = reduce(hcat, coords)
        compare("3D regular grid", pos, 0.05f0)
    end

    println("─" ^ 88)
end

main()
