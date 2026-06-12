using Test
using Octopus
using Random

# Float64 smoke tests. The README calls Float64 "works but unbenchmarked".
# We don't measure perf here; we just lock that the build/query/edge paths
# are functionally correct so users who pick Float64 don't hit a surprise.

function _bf_neighbors(coords::Matrix{Float64}, i::Int, r::Float64; exclude_self=true)
    out = Int32[]
    r_sq = r * r
    n = size(coords, 2)
    @inbounds for j in 1:n
        exclude_self && j == i && continue
        d2 = zero(Float64)
        for d in 1:size(coords, 1)
            δ = coords[d, j] - coords[d, i]
            d2 += δ * δ
        end
        if d2 <= r_sq
            push!(out, Int32(j))
        end
    end
    return out
end

@testset "Float64 3D build + query against brute force" begin
    Random.seed!(4001)
    coords = rand(Float64, 3, 400)
    r = 0.1
    tns = TNS(Float64)
    @test tns isa TNS{Float64,3}
    set_search_radius!(tns, r)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)

    for i in 1:size(coords, 2)
        got = sort(get_neighborlist(tns, id, id, i))
        want = sort(_bf_neighbors(coords, i, r))
        @test got == want
    end
end

@testset "Float64 3D edges output types and shapes" begin
    Random.seed!(4002)
    coords = rand(Float64, 3, 300)
    r = 0.08
    tns = TNS(Float64)
    set_search_radius!(tns, r)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    e = build_edges(tns, id, id)

    n_edges = length(e.senders)
    @test e.senders          isa Vector{Int32}
    @test e.receivers        isa Vector{Int32}
    @test e.rel_displacement isa Matrix{Float64}
    @test e.rel_dist_norm    isa Matrix{Float64}
    @test size(e.rel_displacement) == (3, n_edges)
    @test size(e.rel_dist_norm)    == (1, n_edges)
    # rel_dist_norm is normalized by r, so all values ≤ 1 (plus float slack).
    for k in 1:n_edges
        @test e.rel_dist_norm[1, k] <= 1.0 + 1e-12
    end
end

@testset "Float64 2D build + query against brute force" begin
    Random.seed!(4003)
    coords = rand(Float64, 2, 300)
    r = 0.1
    tns = TNS(Float64; ndims=2)
    @test tns isa TNS{Float64,2}
    set_search_radius!(tns, r)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)

    for i in 1:size(coords, 2)
        got = sort(get_neighborlist(tns, id, id, i))
        want = sort(_bf_neighbors(coords, i, r))
        @test got == want
    end
end

@testset "Float64 materialize_all_neighbors! agrees with lazy" begin
    Random.seed!(4004)
    coords = rand(Float64, 3, 300)
    r = 0.1
    tns = TNS(Float64); set_search_radius!(tns, r)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    materialize_all_neighbors!(tns)

    pair_idx = findfirst(==((Int32(id), Int32(id))), tns.active_pairs)
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

@testset "Float64 apply_zsort! 1-D + 2-D" begin
    Random.seed!(4005)
    coords = rand(Float64, 3, 200)
    tns = TNS(Float64); set_search_radius!(tns, 0.1)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)

    masses = rand(Float64, 200)
    sorted = apply_zsort!(tns, id, masses)
    perm = collect(tns.permutation[id])
    @test sorted == [masses[perm[i]] for i in eachindex(perm)]
    @test typeof(sorted) === typeof(masses)

    feats = rand(Float64, 4, 200)
    sorted2 = apply_zsort!(tns, id, feats)
    ref = similar(feats)
    @inbounds for i in eachindex(perm)
        ref[:, i] = feats[:, perm[i]]
    end
    @test sorted2 == ref
end

@testset "Float64 + Float32 cross-rejection" begin
    # TNS{Float32} takes coords::AbstractMatrix{Float32}; passing Float64
    # coords is a method-dispatch mismatch (MethodError).
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    coords64 = rand(Float64, 3, 10)
    @test_throws MethodError add_point_set!(tns, coords64)

    tns2 = TNS(Float64); set_search_radius!(tns2, 0.1)
    coords32 = rand(Float32, 3, 10)
    @test_throws MethodError add_point_set!(tns2, coords32)
end
