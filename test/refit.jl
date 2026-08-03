#
# Copyright (c) 2026 Josef Jouaux
# Copyright (c) 2022-present, CompactNSearch contributors
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Test
using Octopus
using Random

# Refit mode: when on, the morton_codes scratch is retained between run!
# calls so the next rebuild can detect cell crossings cheaply. When off, the
# scratch is dropped to keep steady-state memory minimal.

@testset "default mode drops morton_codes after run!" begin
    Random.seed!(2001)
    coords = rand(Float32, 3, 300)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    @test tns.refit_mode == false

    run!(tns)
    @test length(tns.morton_codes[id]) == 0
    # Permutation is always retained — it's the index into target coords.
    @test length(tns.permutation[id]) == 300
end

@testset "refit mode retains morton_codes" begin
    Random.seed!(2002)
    coords = rand(Float32, 3, 300)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    set_refit_mode!(tns, true)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)

    run!(tns)
    @test tns.refit_mode == true
    @test length(tns.morton_codes[id]) == 300
end

@testset "toggling refit_mode after run! marks dirty" begin
    Random.seed!(2003)
    coords = rand(Float32, 3, 200)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    @test !tns.dirty[id]

    set_refit_mode!(tns, true)
    @test tns.dirty[id]
    run!(tns)
    # After run!, morton_codes are present.
    @test length(tns.morton_codes[id]) == 200

    set_refit_mode!(tns, false)
    @test tns.dirty[id]
    run!(tns)
    @test length(tns.morton_codes[id]) == 0
end

@testset "refit-mode rebuild produces same neighbor relation as default" begin
    # Functional equivalence: refit_mode is an internal optimisation, the
    # observable neighbor relation must match either way.
    Random.seed!(2004)
    coords = rand(Float32, 3, 500)
    r = 0.08f0

    tns_a = TNS(Float32); set_search_radius!(tns_a, r)
    id_a = add_point_set!(tns_a, coords)
    set_active_search!(tns_a, id_a, id_a)
    run!(tns_a)

    tns_b = TNS(Float32); set_search_radius!(tns_b, r)
    set_refit_mode!(tns_b, true)
    id_b = add_point_set!(tns_b, coords)
    set_active_search!(tns_b, id_b, id_b)
    run!(tns_b)

    for i in 1:size(coords, 2)
        a = sort(get_neighborlist(tns_a, id_a, id_a, i))
        b = sort(get_neighborlist(tns_b, id_b, id_b, i))
        @test a == b
    end
end

@testset "refit mode survives an update_point_set! + run!" begin
    Random.seed!(2005)
    coords1 = rand(Float32, 3, 200)
    coords2 = rand(Float32, 3, 200)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    set_refit_mode!(tns, true)
    id = add_point_set!(tns, coords1)
    set_active_search!(tns, id, id)
    run!(tns)
    @test length(tns.morton_codes[id]) == 200

    update_point_set!(tns, id, coords2)
    run!(tns)
    @test length(tns.morton_codes[id]) == 200  # still retained
    @test length(tns.permutation[id]) == 200
end

# 2D version of the same invariants — cheap, catches dimension-specific drift.

@testset "2D default mode drops morton_codes after run!" begin
    Random.seed!(2006)
    coords = rand(Float32, 2, 300)
    tns = TNS(Float32; ndims=2); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    @test length(tns.morton_codes[id]) == 0
end

@testset "2D refit mode retains morton_codes" begin
    Random.seed!(2007)
    coords = rand(Float32, 2, 300)
    tns = TNS(Float32; ndims=2); set_search_radius!(tns, 0.1f0)
    set_refit_mode!(tns, true)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    @test length(tns.morton_codes[id]) == 300
end
