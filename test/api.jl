#
# Copyright (c) 2026 Josef Jouaux
# Copyright (c) 2022-present, CompactNSearch contributors
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Test
using Octopus
using Random

@testset "TNS constructor defaults" begin
    tns = TNS()
    @test tns.dev === :uninitialized
    @test tns.target_leaf_size == Int32(32)
    @test tns.refit_mode == false
end

@testset "radius must be positive" begin
    tns = TNS()
    @test_throws ArgumentError set_search_radius!(tns, -1f0)
end

@testset "dimension mismatch" begin
    tns = TNS()
    set_search_radius!(tns, 0.1f0)
    bad = rand(Float32, 2, 10)
    @test_throws DimensionMismatch add_point_set!(tns, bad)
end

@testset "run! requires radius and at least one set" begin
    tns = TNS()
    @test_throws ArgumentError run!(tns)
end

@testset "update_point_set! forbids size change" begin
    Random.seed!(1)
    coords = rand(Float32, 3, 100)
    tns = TNS(); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    bad = rand(Float32, 3, 99)
    @test_throws DimensionMismatch update_point_set!(tns, id, bad)
end

@testset "for_each_neighbor is callback-only and matches list" begin
    Random.seed!(2)
    coords = rand(Float32, 3, 200)
    r = 0.1f0
    tns = TNS(); set_search_radius!(tns, r)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)

    for i in 1:size(coords, 2)
        seen = Int32[]
        for_each_neighbor(tns, id, id, i) do j
            if j != Int32(i)
                push!(seen, j)
            end
        end
        @test sort(seen) == sort(get_neighborlist(tns, id, id, i))
    end
end
