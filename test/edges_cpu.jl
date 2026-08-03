#
# Copyright (c) 2026 Josef Jouaux
# Copyright (c) 2022-present, CompactNSearch contributors
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Test
using Octopus
using Random

# Brute-force reference for the full (senders, receivers, rel_displacement,
# rel_dist_norm) tuple. Convention: rel_displacement = (q - t) / r, where the
# query point is `i` (receiver) and the target/neighbor is `j` (sender).
function brute_force_edges(coords_q::Matrix{Float32}, coords_t::Matrix{Float32},
                           r::Float32; exclude_self::Bool)
    senders   = Int32[]
    receivers = Int32[]
    disp_cols = Vector{Vector{Float32}}()
    dists     = Float32[]
    r_sq = r * r
    inv_r = 1f0 / r
    n_q = size(coords_q, 2); n_t = size(coords_t, 2)
    @inbounds for i in 1:n_q
        for j in 1:n_t
            exclude_self && j == i && continue
            dx = coords_q[1, i] - coords_t[1, j]
            dy = coords_q[2, i] - coords_t[2, j]
            dz = coords_q[3, i] - coords_t[3, j]
            d2 = dx*dx + dy*dy + dz*dz
            if d2 <= r_sq
                push!(senders, Int32(j))
                push!(receivers, Int32(i))
                push!(disp_cols, Float32[dx*inv_r, dy*inv_r, dz*inv_r])
                push!(dists, sqrt(d2) * inv_r)
            end
        end
    end
    return senders, receivers, disp_cols, dists
end

# Sort an edge list (senders, receivers, displacement matrix, distance matrix)
# by (receiver, sender) so we can compare across implementations whose
# traversal order differs.
function sort_edges(senders::AbstractVector, receivers::AbstractVector,
                    rel_disp::AbstractMatrix, rel_dist::AbstractMatrix)
    perm = sortperm(collect(zip(Array(receivers), Array(senders))))
    s = Array(senders)[perm]
    r = Array(receivers)[perm]
    d = Array(rel_disp)[:, perm]
    n = Array(rel_dist)[:, perm]
    return s, r, d, n
end

function compare_edges_to_brute(coords::Matrix{Float32}, r::Float32; seed=0)
    Random.seed!(seed)
    tns = TNS(Float32)
    set_search_radius!(tns, r)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)

    e = build_edges(tns, id, id)
    s, r_, d, n = sort_edges(e.senders, e.receivers, e.rel_displacement, e.rel_dist_norm)

    bs, br, bd, bdist = brute_force_edges(coords, coords, r; exclude_self=true)
    bperm = sortperm(collect(zip(br, bs)))
    bs_sorted = bs[bperm]
    br_sorted = br[bperm]

    @test s == bs_sorted
    @test r_ == br_sorted

    # Element-wise displacement / distance comparison after sorting.
    for k in 1:length(s)
        @test d[:, k] ≈ bd[bperm[k]] atol=1f-5
        @test n[1, k] ≈ bdist[bperm[k]] atol=1f-5
        @test n[1, k] ≤ 1f0 + 1f-5
    end
end

@testset "edges N=200, radius 1x cell" begin
    Random.seed!(11)
    coords = rand(Float32, 3, 200)
    compare_edges_to_brute(coords, 0.15f0)
end

@testset "edges N=1000, radius 0.5x cell" begin
    Random.seed!(21)
    coords = rand(Float32, 3, 1000)
    compare_edges_to_brute(coords, 0.05f0)
end

@testset "edges N=2000 clustered" begin
    Random.seed!(31)
    coords = 0.1f0 .* randn(Float32, 3, 2000)
    compare_edges_to_brute(coords, 0.03f0)
end

@testset "edges asymmetric (two sets, no self exclusion)" begin
    Random.seed!(41)
    A = rand(Float32, 3, 150)
    B = rand(Float32, 3, 250)
    r = 0.12f0

    tns = TNS(Float32)
    set_search_radius!(tns, r)
    a = add_point_set!(tns, A)
    b = add_point_set!(tns, B)
    set_active_search!(tns, a, b)  # query a → target b
    run!(tns)

    e = build_edges(tns, a, b)
    s, r_, d, n = sort_edges(e.senders, e.receivers, e.rel_displacement, e.rel_dist_norm)

    bs, br, bd, bdist = brute_force_edges(A, B, r; exclude_self=false)
    bperm = sortperm(collect(zip(br, bs)))

    @test s == bs[bperm]
    @test r_ == br[bperm]
    for k in 1:length(s)
        @test d[:, k] ≈ bd[bperm[k]] atol=1f-5
        @test n[1, k] ≈ bdist[bperm[k]] atol=1f-5
    end
end

@testset "edges agree with materialize_all_neighbors!" begin
    # Same TNS exercises both code paths: the (receiver, sender) set from
    # build_edges! must match the CSR neighbor relation.
    Random.seed!(51)
    coords = rand(Float32, 3, 400)
    r = 0.07f0
    tns = TNS(Float32)
    set_search_radius!(tns, r)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)

    materialize_all_neighbors!(tns)
    e = build_edges(tns, id, id)

    pair_idx = findfirst(==((Int32(id), Int32(id))), tns.active_pairs)
    nb = tns.neighbor_buffers[pair_idx]

    csr_pairs = Set{Tuple{Int32,Int32}}()
    n = size(coords, 2)
    for i in 1:n
        lo = nb.offsets[i] + 1
        hi = nb.offsets[i + 1]
        for k in lo:hi
            push!(csr_pairs, (Int32(i), nb.flat[k]))
        end
    end

    edge_pairs = Set{Tuple{Int32,Int32}}()
    for k in 1:length(e.senders)
        push!(edge_pairs, (e.receivers[k], e.senders[k]))
    end

    @test edge_pairs == csr_pairs
end

@testset "edges output shapes/types are flat-COO Vector{Int32} / Matrix{T}" begin
    Random.seed!(61)
    coords = rand(Float32, 3, 300)
    r = 0.08f0
    tns = TNS(Float32)
    set_search_radius!(tns, r)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    e = build_edges(tns, id, id)

    n_edges = length(e.senders)
    @test e.senders   isa Vector{Int32}
    @test e.receivers isa Vector{Int32}
    @test e.rel_displacement isa Matrix{Float32}
    @test e.rel_dist_norm    isa Matrix{Float32}
    @test size(e.rel_displacement) == (3, n_edges)
    @test size(e.rel_dist_norm)    == (1, n_edges)
end

@testset "edges invalidated and rebuilt across runs" begin
    Random.seed!(71)
    coords1 = rand(Float32, 3, 150)
    r = 0.1f0
    tns = TNS(Float32)
    set_search_radius!(tns, r)
    id = add_point_set!(tns, coords1)
    set_active_search!(tns, id, id)
    run!(tns)
    e1 = build_edges(tns, id, id)
    n1 = length(e1.senders)
    @test n1 > 0

    # Move particles, re-run, re-build.
    coords2 = rand(Float32, 3, 150)
    update_point_set!(tns, id, coords2)
    run!(tns)
    e2 = build_edges(tns, id, id)
    @test length(e2.senders) > 0
    # Result reflects the new positions.
    bs, br, _, _ = brute_force_edges(coords2, coords2, r; exclude_self=true)
    @test length(e2.senders) == length(bs)
end
