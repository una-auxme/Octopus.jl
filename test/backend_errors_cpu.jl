#
# Copyright (c) 2026 Josef Jouaux, Chair of Mechatronics, University of Augsburg
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Test
using Octopus
using Random

# Backend-mismatch error contracts that can be checked without CUDA loaded.
# The mirror image (host calls on a :cuda TNS) lives in the CUDA test files
# under JULIA_OCTOPUS_TEST_CUDA=1.

@testset "device_view on a CPU TNS errors" begin
    Random.seed!(5001)
    coords = rand(Float32, 3, 50)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    @test_throws ErrorException device_view(tns, id, id)
end

@testset "run! before any point set errors with a friendly message" begin
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    @test_throws ArgumentError run!(tns)
end

@testset "run! before set_search_radius! errors" begin
    Random.seed!(5002)
    coords = rand(Float32, 3, 50)
    tns = TNS(Float32)
    # add_point_set! without a radius is allowed (just registers the set).
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    @test_throws ArgumentError run!(tns)
end

@testset "build_edges! before run! still produces correct edges" begin
    # `build_edges!` doesn't require `run!` per se — but the tree is empty
    # until the first run. After run!, edges work; before, the tree has
    # n_nodes=0 and traversal short-circuits to no edges. Lock that.
    Random.seed!(5003)
    coords = rand(Float32, 3, 100)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)

    # n_nodes == 0 → traversal short-circuits to zero edges (no crash).
    e0 = build_edges(tns, id, id)
    @test length(e0.senders) == 0

    run!(tns)
    e1 = build_edges(tns, id, id)
    @test length(e1.senders) > 0
end

@testset "prepare_zsort! errors when run! hasn't populated perm" begin
    Random.seed!(5004)
    coords = rand(Float32, 3, 50)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    @test_throws ErrorException prepare_zsort!(tns)
end

@testset "apply_zsort! rejects mismatched length" begin
    Random.seed!(5005)
    coords = rand(Float32, 3, 50)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)

    # Wrong length triggers the @assert inside apply_zsort_cpu! (AssertionError).
    wrong = rand(Float32, 49)
    @test_throws AssertionError apply_zsort!(tns, id, wrong)
end

@testset "set_search_radius! rejects zero and negative" begin
    tns = TNS(Float32)
    @test_throws ArgumentError set_search_radius!(tns, 0.0f0)
    @test_throws ArgumentError set_search_radius!(tns, -0.1f0)
end

@testset "add_point_set! rejects wrong row count (3D and 2D)" begin
    tns3 = TNS(Float32; ndims=3); set_search_radius!(tns3, 0.1f0)
    @test_throws DimensionMismatch add_point_set!(tns3, rand(Float32, 2, 10))
    @test_throws DimensionMismatch add_point_set!(tns3, rand(Float32, 4, 10))

    tns2 = TNS(Float32; ndims=2); set_search_radius!(tns2, 0.1f0)
    @test_throws DimensionMismatch add_point_set!(tns2, rand(Float32, 3, 10))
    @test_throws DimensionMismatch add_point_set!(tns2, rand(Float32, 1, 10))
end
