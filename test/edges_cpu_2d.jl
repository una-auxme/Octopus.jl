#
# Copyright (c) 2026 Josef Jouaux, Chair of Mechatronics, University of Augsburg
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Test
using Octopus
using Random

# Brute-force edges for 2D. Convention matches 3D edges:
#   rel_displacement = (q - t) / r, receiver = i, sender = j.
function brute_force_edges_2d(coords_q::Matrix{Float32}, coords_t::Matrix{Float32},
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
            d2 = dx*dx + dy*dy
            if d2 <= r_sq
                push!(senders, Int32(j))
                push!(receivers, Int32(i))
                push!(disp_cols, Float32[dx*inv_r, dy*inv_r])
                push!(dists, sqrt(d2) * inv_r)
            end
        end
    end
    return senders, receivers, disp_cols, dists
end

function sort_edges_2d(senders::AbstractVector, receivers::AbstractVector,
                       rel_disp::AbstractMatrix, rel_dist::AbstractMatrix)
    perm = sortperm(collect(zip(Array(receivers), Array(senders))))
    s = Array(senders)[perm]
    r = Array(receivers)[perm]
    d = Array(rel_disp)[:, perm]
    n = Array(rel_dist)[:, perm]
    return s, r, d, n
end

function compare_edges_to_brute_2d(coords::Matrix{Float32}, r::Float32; seed=0)
    Random.seed!(seed)
    tns = TNS(Float32; ndims=2)
    set_search_radius!(tns, r)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)

    e = build_edges(tns, id, id)
    s, r_, d, n = sort_edges_2d(e.senders, e.receivers, e.rel_displacement, e.rel_dist_norm)

    bs, br, bd, bdist = brute_force_edges_2d(coords, coords, r; exclude_self=true)
    bperm = sortperm(collect(zip(br, bs)))
    bs_sorted = bs[bperm]
    br_sorted = br[bperm]

    @test s == bs_sorted
    @test r_ == br_sorted

    for k in 1:length(s)
        @test d[:, k] ≈ bd[bperm[k]] atol=1f-5
        @test n[1, k] ≈ bdist[bperm[k]] atol=1f-5
        @test n[1, k] ≤ 1f0 + 1f-5
    end
end

@testset "2D edges N=200, radius 1x cell" begin
    Random.seed!(211)
    coords = rand(Float32, 2, 200)
    compare_edges_to_brute_2d(coords, 0.15f0)
end

@testset "2D edges N=1000, radius 0.5x cell" begin
    Random.seed!(221)
    coords = rand(Float32, 2, 1000)
    compare_edges_to_brute_2d(coords, 0.05f0)
end

@testset "2D edges N=2000 clustered" begin
    Random.seed!(231)
    coords = 0.1f0 .* randn(Float32, 2, 2000)
    compare_edges_to_brute_2d(coords, 0.03f0)
end

@testset "2D edges asymmetric (two sets, no self exclusion)" begin
    Random.seed!(241)
    A = rand(Float32, 2, 150)
    B = rand(Float32, 2, 250)
    r = 0.12f0

    tns = TNS(Float32; ndims=2)
    set_search_radius!(tns, r)
    a = add_point_set!(tns, A)
    b = add_point_set!(tns, B)
    set_active_search!(tns, a, b)
    run!(tns)

    e = build_edges(tns, a, b)
    s, r_, d, n = sort_edges_2d(e.senders, e.receivers, e.rel_displacement, e.rel_dist_norm)

    bs, br, bd, bdist = brute_force_edges_2d(A, B, r; exclude_self=false)
    bperm = sortperm(collect(zip(br, bs)))

    @test s == bs[bperm]
    @test r_ == br[bperm]
    for k in 1:length(s)
        @test d[:, k] ≈ bd[bperm[k]] atol=1f-5
        @test n[1, k] ≈ bdist[bperm[k]] atol=1f-5
    end
end

@testset "2D edges output shapes/types" begin
    Random.seed!(261)
    coords = rand(Float32, 2, 300)
    r = 0.08f0
    tns = TNS(Float32; ndims=2)
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
    @test size(e.rel_displacement) == (2, n_edges)
    @test size(e.rel_dist_norm)    == (1, n_edges)
end
