#
# Copyright (c) 2026 Josef Jouaux, Chair of Mechatronics, University of Augsburg
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Test
using Octopus
using Random

# brute_force_neighbors is defined in test/build.jl, included by runtests.jl
# before this file. We don't re-include to avoid method-overwrite warnings.

function compare_to_brute(coords::Matrix{Float32}, r::Float32; seed=0)
    Random.seed!(seed)
    tns = TNS(Float32)
    set_search_radius!(tns, r)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    n = size(coords, 2)
    for i in 1:n
        got = sort(get_neighborlist(tns, id, id, i))
        want = sort(brute_force_neighbors(coords, i, r))
        @test got == want
    end
end

@testset "N=50, radius 1x cell" begin
    Random.seed!(10)
    coords = rand(Float32, 3, 50)
    compare_to_brute(coords, 0.15f0)
end

@testset "N=500, radius 0.5x cell" begin
    Random.seed!(20)
    coords = rand(Float32, 3, 500)
    compare_to_brute(coords, 0.05f0)
end

@testset "N=2000, radius 2x cell" begin
    Random.seed!(30)
    coords = rand(Float32, 3, 2000)
    compare_to_brute(coords, 0.1f0)
end

@testset "N=5000 clustered" begin
    Random.seed!(40)
    coords = 0.1f0 .* randn(Float32, 3, 5000)
    compare_to_brute(coords, 0.03f0)
end

@testset "asymmetric search between two sets" begin
    Random.seed!(50)
    A = rand(Float32, 3, 200)
    B = rand(Float32, 3, 300)
    r = 0.1f0

    tns = TNS(Float32)
    set_search_radius!(tns, r)
    a = add_point_set!(tns, A)
    b = add_point_set!(tns, B)
    set_active_search!(tns, a, b)   # a queries into b
    run!(tns)

    r_sq = r * r
    for i in 1:size(A, 2)
        want = Int32[]
        @inbounds for j in 1:size(B, 2)
            dx = B[1, j] - A[1, i]
            dy = B[2, j] - A[2, i]
            dz = B[3, j] - A[3, i]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 <= r_sq
                push!(want, Int32(j))
            end
        end
        got = sort(get_neighborlist(tns, a, b, i))
        @test got == sort(want)
    end
end

@testset "materialize_all_neighbors! agrees with lazy" begin
    Random.seed!(60)
    coords = rand(Float32, 3, 500)
    r = 0.08f0
    tns = TNS(Float32)
    set_search_radius!(tns, r)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    materialize_all_neighbors!(tns)

    # pair index for (id, id)
    pair_idx = findfirst(==((Int32(id), Int32(id))), tns.active_pairs)
    @test pair_idx !== nothing
    buf = tns.neighbor_buffers[pair_idx]
    @test buf.materialized

    for i in 1:size(coords, 2)
        lo = buf.offsets[i] + 1
        hi = buf.offsets[i + 1]
        got = sort(buf.flat[lo:hi])
        want = sort(get_neighborlist(tns, id, id, i))
        @test got == want
    end
end
