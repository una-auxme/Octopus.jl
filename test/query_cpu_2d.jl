using Test
using Octopus
using Random

# brute_force_neighbors_2d is defined in test/build_2d.jl, included by
# runtests.jl before this file.

function compare_to_brute_2d(coords::Matrix{Float32}, r::Float32; seed=0)
    Random.seed!(seed)
    tns = TNS(Float32; ndims=2)
    set_search_radius!(tns, r)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    n = size(coords, 2)
    for i in 1:n
        got = sort(get_neighborlist(tns, id, id, i))
        want = sort(brute_force_neighbors_2d(coords, i, r))
        @test got == want
    end
end

@testset "2D N=50, radius 1x cell" begin
    Random.seed!(110)
    coords = rand(Float32, 2, 50)
    compare_to_brute_2d(coords, 0.15f0)
end

@testset "2D N=500, radius 0.5x cell" begin
    Random.seed!(120)
    coords = rand(Float32, 2, 500)
    compare_to_brute_2d(coords, 0.05f0)
end

@testset "2D N=2000, radius 2x cell" begin
    Random.seed!(130)
    coords = rand(Float32, 2, 2000)
    compare_to_brute_2d(coords, 0.1f0)
end

@testset "2D N=5000 clustered" begin
    Random.seed!(140)
    coords = 0.1f0 .* randn(Float32, 2, 5000)
    compare_to_brute_2d(coords, 0.03f0)
end

@testset "2D asymmetric search between two sets" begin
    Random.seed!(150)
    A = rand(Float32, 2, 200)
    B = rand(Float32, 2, 300)
    r = 0.1f0

    tns = TNS(Float32; ndims=2)
    set_search_radius!(tns, r)
    a = add_point_set!(tns, A)
    b = add_point_set!(tns, B)
    set_active_search!(tns, a, b)
    run!(tns)

    r_sq = r * r
    for i in 1:size(A, 2)
        want = Int32[]
        @inbounds for j in 1:size(B, 2)
            dx = B[1, j] - A[1, i]
            dy = B[2, j] - A[2, i]
            d2 = dx * dx + dy * dy
            if d2 <= r_sq
                push!(want, Int32(j))
            end
        end
        got = sort(get_neighborlist(tns, a, b, i))
        @test got == sort(want)
    end
end

@testset "2D materialize_all_neighbors! agrees with lazy" begin
    Random.seed!(160)
    coords = rand(Float32, 2, 500)
    r = 0.08f0
    tns = TNS(Float32; ndims=2)
    set_search_radius!(tns, r)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    materialize_all_neighbors!(tns)

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
