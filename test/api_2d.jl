using Test
using TreeNSearch
using Random

@testset "2D TNS constructor" begin
    tns = TNS(Float32; ndims=2)
    @test tns isa TNS{Float32,2}
    @test tns.dev === :uninitialized
end

@testset "2D dimension mismatch on add_point_set!" begin
    tns = TNS(Float32; ndims=2)
    set_search_radius!(tns, 0.1f0)
    bad3 = rand(Float32, 3, 10)
    @test_throws DimensionMismatch add_point_set!(tns, bad3)
    bad4 = rand(Float32, 4, 10)
    @test_throws DimensionMismatch add_point_set!(tns, bad4)
end

@testset "3D dimension mismatch rejects 2D coords" begin
    tns = TNS(Float32; ndims=3)
    set_search_radius!(tns, 0.1f0)
    bad2 = rand(Float32, 2, 10)
    @test_throws DimensionMismatch add_point_set!(tns, bad2)
end

@testset "2D update_point_set! forbids size change" begin
    Random.seed!(301)
    coords = rand(Float32, 2, 100)
    tns = TNS(Float32; ndims=2); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    bad = rand(Float32, 2, 99)
    @test_throws DimensionMismatch update_point_set!(tns, id, bad)
end

@testset "2D update_point_set! rejects wrong row count" begin
    Random.seed!(302)
    coords = rand(Float32, 2, 100)
    tns = TNS(Float32; ndims=2); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    bad = rand(Float32, 3, 100)
    @test_throws DimensionMismatch update_point_set!(tns, id, bad)
end

@testset "2D for_each_neighbor matches list" begin
    Random.seed!(303)
    coords = rand(Float32, 2, 200)
    r = 0.1f0
    tns = TNS(Float32; ndims=2); set_search_radius!(tns, r)
    id = add_point_set!(tns, coords)
    set_symmetric_search!(tns, id, id)
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

@testset "2D edges invalidated and rebuilt across runs" begin
    Random.seed!(304)
    coords1 = rand(Float32, 2, 150)
    r = 0.1f0
    tns = TNS(Float32; ndims=2); set_search_radius!(tns, r)
    id = add_point_set!(tns, coords1)
    set_symmetric_search!(tns, id, id)
    run!(tns)
    e1 = build_edges(tns, id, id)
    n1 = length(e1.senders)
    @test n1 > 0

    coords2 = rand(Float32, 2, 150)
    update_point_set!(tns, id, coords2)
    run!(tns)
    e2 = build_edges(tns, id, id)
    @test length(e2.senders) > 0
end
