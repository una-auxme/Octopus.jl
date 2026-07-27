#
# Copyright (c) 2026 Josef Jouaux, Chair of Mechatronics, University of Augsburg
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Test
using Octopus
using Random

# Hand-permutation reference: gather along the last axis using `perm`.
function _ref_zsort(perm, x::AbstractVector)
    [x[perm[i]] for i in eachindex(perm)]
end
function _ref_zsort(perm, x::AbstractMatrix)
    out = similar(x)
    @inbounds for i in eachindex(perm)
        out[:, i] .= x[:, perm[i]]
    end
    out
end

@testset "apply_zsort! CPU 1-D" begin
    Random.seed!(11)
    coords = rand(Float32, 3, 200)
    tns = TNS(Float32)
    set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    prepare_zsort!(tns)

    masses = rand(Float32, 200)
    sorted = apply_zsort!(tns, id, masses)

    perm = collect(tns.permutation[id])
    @test sorted == _ref_zsort(perm, masses)
    @test typeof(sorted) === typeof(masses)
end

@testset "apply_zsort! CPU 2-D" begin
    Random.seed!(13)
    coords = rand(Float32, 3, 200)
    tns = TNS(Float32)
    set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)

    velocities = rand(Float32, 4, 200)
    sorted = apply_zsort!(tns, id, velocities)

    perm = collect(tns.permutation[id])
    @test sorted == _ref_zsort(perm, velocities)
end

@testset "apply_zsort! CPU rejects 3-D input" begin
    Random.seed!(17)
    coords = rand(Float32, 3, 50)
    tns = TNS(Float32)
    set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)

    feats = rand(Float32, 3, 3, 50)
    @test_throws ArgumentError apply_zsort!(tns, id, feats)
end

@testset "prepare_zsort! errors before run!" begin
    coords = rand(Float32, 3, 32)
    tns = TNS(Float32)
    set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    @test_throws ErrorException prepare_zsort!(tns)
end
