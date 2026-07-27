#
# Copyright (c) 2026 Josef Jouaux, Chair of Mechatronics, University of Augsburg
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Test
using Octopus
using Random
using CUDA

if !CUDA.functional()
    @info "CUDA unavailable; skipping GPU zsort tests"
else
    @testset "apply_zsort! CUDA 1-D" begin
        Random.seed!(311)
        coords = rand(Float32, 3, 256)
        gpu_coords = CuArray(coords)
        tns = TNS(Float32)
        set_search_radius!(tns, 0.1f0)
        id = add_point_set!(tns, gpu_coords)
        set_active_search!(tns, id, id)
        run!(tns)

        masses_h = rand(Float32, 256)
        masses = CuArray(masses_h)
        sorted = apply_zsort!(tns, id, masses)

        perm_h = Array(tns.permutation[id])
        @test sorted isa CuArray{Float32,1}
        @test Array(sorted) == [masses_h[perm_h[i]] for i in eachindex(perm_h)]
    end

    @testset "apply_zsort! CUDA 2-D" begin
        Random.seed!(313)
        coords = rand(Float32, 3, 256)
        gpu_coords = CuArray(coords)
        tns = TNS(Float32)
        set_search_radius!(tns, 0.1f0)
        id = add_point_set!(tns, gpu_coords)
        set_active_search!(tns, id, id)
        run!(tns)

        feats_h = rand(Float32, 5, 256)
        feats = CuArray(feats_h)
        sorted = apply_zsort!(tns, id, feats)

        perm_h = Array(tns.permutation[id])
        ref = similar(feats_h)
        @inbounds for i in eachindex(perm_h)
            ref[:, i] = feats_h[:, perm_h[i]]
        end
        @test sorted isa CuArray{Float32,2}
        @test Array(sorted) == ref
    end

    @testset "apply_zsort! CUDA rejects 3-D input" begin
        Random.seed!(317)
        coords = rand(Float32, 3, 64)
        gpu_coords = CuArray(coords)
        tns = TNS(Float32)
        set_search_radius!(tns, 0.1f0)
        id = add_point_set!(tns, gpu_coords)
        set_active_search!(tns, id, id)
        run!(tns)

        feats = CuArray(rand(Float32, 2, 2, 64))
        @test_throws ArgumentError apply_zsort!(tns, id, feats)
    end
end
