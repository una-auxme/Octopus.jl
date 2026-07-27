#
# Copyright (c) 2026 Josef Jouaux, Chair of Mechatronics, University of Augsburg
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Test
using Octopus
using Random

# 2D brute-force neighbor reference, mirroring the 3D one in test/build.jl.
function brute_force_neighbors_2d(coords::Matrix{Float32}, i::Int, r::Float32; exclude_self=true)
    out = Int32[]
    n = size(coords, 2)
    r_sq = r * r
    @inbounds for j in 1:n
        exclude_self && j == i && continue
        dx = coords[1, j] - coords[1, i]
        dy = coords[2, j] - coords[2, i]
        d2 = dx * dx + dy * dy
        if d2 <= r_sq
            push!(out, Int32(j))
        end
    end
    return out
end

function assert_tree_invariants_2d(tns::TNS, set_id::Int)
    tree = tns.trees[set_id]
    perm = tns.permutation[set_id]
    n = Int(tns.point_sets[set_id].n)
    n_nodes = Int(tree.n_nodes)
    @test n_nodes >= 1

    @test tree.node_first[1] == Int32(1)
    @test tree.node_last[1] == Int32(n)
    @test sort(perm) == collect(Int32(1):Int32(n))

    # Quadtree: 4 children rows.
    @test size(tree.node_children, 1) == 4
    @test size(tree.node_bounds_min, 1) == 2
    @test size(tree.node_bounds_max, 1) == 2

    for nid in 1:n_nodes
        if tree.node_children[1, nid] == Int32(-1)
            c = tree.node_last[nid] - tree.node_first[nid] + Int32(1)
            @test c >= Int32(1)
        else
            union_count = Int32(0)
            for k in 1:4
                cid = tree.node_children[k, nid]
                cid == Int32(-1) && continue
                @test cid > nid
                @test tree.node_first[cid] >= tree.node_first[nid]
                @test tree.node_last[cid] <= tree.node_last[nid]
                union_count += tree.node_last[cid] - tree.node_first[cid] + Int32(1)
            end
            @test union_count == tree.node_last[nid] - tree.node_first[nid] + Int32(1)
        end
    end

    bmin = tree.node_bounds_min
    bmax = tree.node_bounds_max
    coords = tns.point_sets[set_id].coords
    for nid in 1:n_nodes
        if tree.node_children[1, nid] == Int32(-1)
            for k in tree.node_first[nid]:tree.node_last[nid]
                p = perm[k]
                @test coords[1, p] >= bmin[1, nid] - 1f-6
                @test coords[1, p] <= bmax[1, nid] + 1f-6
                @test coords[2, p] >= bmin[2, nid] - 1f-6
                @test coords[2, p] <= bmax[2, nid] + 1f-6
            end
        end
    end
end

@testset "2D uniform square, N=500" begin
    Random.seed!(101)
    coords = rand(Float32, 2, 500)
    tns = TNS(Float32; ndims=2)
    set_search_radius!(tns, 0.08f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    assert_tree_invariants_2d(tns, id)
end

@testset "2D clustered cloud, N=2000" begin
    Random.seed!(102)
    blob1 = 0.1f0 .* randn(Float32, 2, 1000) .+ 0.25f0
    blob2 = 0.1f0 .* randn(Float32, 2, 1000) .- 0.25f0
    coords = hcat(blob1, blob2)
    tns = TNS(Float32; ndims=2)
    set_search_radius!(tns, 0.05f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    assert_tree_invariants_2d(tns, id)
end

@testset "2D single point" begin
    coords = reshape(Float32[0, 0], 2, 1)
    tns = TNS(Float32; ndims=2)
    set_search_radius!(tns, 1f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    @test tns.trees[id].n_nodes == Int32(1)
    @test Octopus.get_neighborlist(tns, id, id, 1) == Int32[]
end
