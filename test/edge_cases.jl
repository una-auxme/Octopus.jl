#
# Copyright (c) 2026 Josef Jouaux, Chair of Mechatronics, University of Augsburg
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Test
using Octopus
using Random

# Edge cases and boundary behaviors. These are the cases that bite first in
# the wild and rarely show up in random benchmarks.

# ---- Empty point sets (N=0) ---------------------------------------------

@testset "empty point set on CPU: run! short-circuits" begin
    # `add_point_set!` requires the matrix to have the right row count, but N=0
    # is allowed. The plan is for the tree to be empty (n_nodes==0) and every
    # query to return zero neighbors with no crash.
    empty3 = Matrix{Float32}(undef, 3, 0)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, empty3)
    set_active_search!(tns, id, id)
    run!(tns)
    @test tns.trees[id].n_nodes == Int32(0)
    @test length(tns.permutation[id]) == 0
end

@testset "empty point set on CPU: build_edges returns 0-edge buffer" begin
    empty3 = Matrix{Float32}(undef, 3, 0)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, empty3)
    set_active_search!(tns, id, id)
    run!(tns)
    e = build_edges(tns, id, id)
    @test length(e.senders)   == 0
    @test length(e.receivers) == 0
    @test size(e.rel_displacement) == (3, 0)
    @test size(e.rel_dist_norm)    == (1, 0)
end

@testset "empty point set on CPU: materialize_all_neighbors!" begin
    empty3 = Matrix{Float32}(undef, 3, 0)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, empty3)
    set_active_search!(tns, id, id)
    run!(tns)
    materialize_all_neighbors!(tns)
    pair_idx = findfirst(==((Int32(id), Int32(id))), tns.active_pairs)
    @test tns.neighbor_buffers[pair_idx].materialized
    @test length(tns.neighbor_buffers[pair_idx].flat) == 0
    @test tns.neighbor_buffers[pair_idx].offsets[1] == Int32(0)
end

@testset "empty 2D point set on CPU" begin
    empty2 = Matrix{Float32}(undef, 2, 0)
    tns = TNS(Float32; ndims=2); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, empty2)
    set_active_search!(tns, id, id)
    run!(tns)
    @test tns.trees[id].n_nodes == Int32(0)
    e = build_edges(tns, id, id)
    @test size(e.rel_displacement) == (2, 0)
end

# ---- Distance boundary --------------------------------------------------

@testset "point at exact distance r is included (3D)" begin
    # The traversal uses d² ≤ r²; a pair exactly at d == r must be a neighbor.
    r = 1.0f0
    coords = Float32[0 r; 0 0; 0 0]                       # two points on x-axis at d=1
    tns = TNS(Float32); set_search_radius!(tns, r)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    @test Int32(2) in get_neighborlist(tns, id, id, 1)
    @test Int32(1) in get_neighborlist(tns, id, id, 2)
end

@testset "point just past r is excluded (3D)" begin
    r = 1.0f0
    coords = Float32[0 (1.001f0 * r); 0 0; 0 0]
    tns = TNS(Float32); set_search_radius!(tns, r)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    @test isempty(get_neighborlist(tns, id, id, 1))
    @test isempty(get_neighborlist(tns, id, id, 2))
end

@testset "point at exact distance r is included (2D)" begin
    r = 1.0f0
    coords = Float32[0 r; 0 0]
    tns = TNS(Float32; ndims=2); set_search_radius!(tns, r)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    @test Int32(2) in get_neighborlist(tns, id, id, 1)
end

# ---- Degenerate inputs --------------------------------------------------

@testset "all points at one location (3D)" begin
    # All-coincident points stress the bit-exhaustion path in the build
    # (single non-empty octant at every level).
    n = 100
    coords = zeros(Float32, 3, n)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    # Every other point is a neighbor of every point (excluding self).
    for i in 1:n
        @test sort(get_neighborlist(tns, id, id, i)) == sort(Int32[j for j in 1:n if j != i])
    end
end

@testset "all points at one location (2D)" begin
    n = 50
    coords = zeros(Float32, 2, n)
    tns = TNS(Float32; ndims=2); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    for i in 1:n
        @test length(get_neighborlist(tns, id, id, i)) == n - 1
    end
end

# ---- Coordinate-space robustness ----------------------------------------

@testset "negative coordinates" begin
    # _point_origin shifts the AABB so binning is non-negative; check it
    # actually works for clouds with negative coords.
    Random.seed!(3001)
    coords = 2f0 .* rand(Float32, 3, 200) .- 1f0   # in [-1, 1]
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)

    # Spot-check against brute force on a few points.
    r = 0.1f0; r2 = r * r
    for i in [1, 50, 150, 200]
        want = Int32[]
        for j in 1:size(coords, 2)
            j == i && continue
            dx = coords[1, j] - coords[1, i]
            dy = coords[2, j] - coords[2, i]
            dz = coords[3, j] - coords[3, i]
            dx*dx + dy*dy + dz*dz <= r2 && push!(want, Int32(j))
        end
        @test sort(get_neighborlist(tns, id, id, i)) == sort(want)
    end
end

@testset "tiny radius returns no neighbors" begin
    Random.seed!(3002)
    coords = rand(Float32, 3, 200)
    tns = TNS(Float32); set_search_radius!(tns, 1f-8)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    for i in 1:size(coords, 2)
        @test isempty(get_neighborlist(tns, id, id, i))
    end
end

@testset "huge radius includes everyone (3D)" begin
    Random.seed!(3003)
    n = 100
    coords = rand(Float32, 3, n)
    tns = TNS(Float32); set_search_radius!(tns, 100f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    for i in 1:n
        @test length(get_neighborlist(tns, id, id, i)) == n - 1
    end
end

@testset "build invariants hold under all-coincident input (3D)" begin
    # Regression: bit-exhaustion path emits single-child chains. The
    # permutation must still cover every particle exactly once.
    n = 64
    coords = zeros(Float32, 3, n)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    perm = tns.permutation[id]
    @test sort(perm) == collect(Int32(1):Int32(n))
end
