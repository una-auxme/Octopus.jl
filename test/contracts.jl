using Test
using TreeNSearch
using Random

# Public-API contracts: idempotency, dirty propagation, buffer lifecycle,
# and the small set of documented behaviors that aren't exercised elsewhere.

@testset "exports are defined" begin
    # Lock the public symbol list. Adding/removing an export is a deliberate
    # decision; this test makes the diff visible.
    for sym in (:TNS,
                :set_search_radius!, :set_refit_mode!,
                :add_point_set!, :resize_point_set!, :update_point_set!,
                :set_active_search!, :set_symmetric_search!,
                :run!,
                :for_each_neighbor, :for_each_neighbor_device,
                :get_neighborlist, :materialize_all_neighbors!,
                :build_edges, :build_edges!, :EdgeBuffer,
                :prepare_zsort!, :apply_zsort!,
                :device_view,
                Symbol("@for_each_neighbor_device_inline"),
                Symbol("@for_each_neighbor_device_inline_2d"))
        @test isdefined(TreeNSearch, sym)
    end
end

@testset "set_active_search! is idempotent" begin
    Random.seed!(1001)
    A = rand(Float32, 3, 50)
    B = rand(Float32, 3, 60)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    a = add_point_set!(tns, A)
    b = add_point_set!(tns, B)

    set_active_search!(tns, a, b)
    n_pairs_1 = length(tns.active_pairs)
    n_bufs_1  = length(tns.neighbor_buffers)
    n_ebufs_1 = length(tns.edge_buffers)

    # Re-register the same pair: docstring promises this is a no-op.
    set_active_search!(tns, a, b)
    @test length(tns.active_pairs)     == n_pairs_1
    @test length(tns.neighbor_buffers) == n_bufs_1
    @test length(tns.edge_buffers)     == n_ebufs_1
end

@testset "set_symmetric_search!(tns, a, a) registers exactly one pair" begin
    Random.seed!(1002)
    coords = rand(Float32, 3, 50)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)

    set_symmetric_search!(tns, id, id)
    @test length(tns.active_pairs) == 1
    @test tns.active_pairs[1] == (Int32(id), Int32(id))
end

@testset "set_symmetric_search!(tns, a, b) registers both directions" begin
    Random.seed!(1003)
    A = rand(Float32, 3, 50)
    B = rand(Float32, 3, 60)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    a = add_point_set!(tns, A)
    b = add_point_set!(tns, B)

    set_symmetric_search!(tns, a, b)
    @test (Int32(a), Int32(b)) in tns.active_pairs
    @test (Int32(b), Int32(a)) in tns.active_pairs
    @test length(tns.active_pairs) == 2
end

@testset "run! is idempotent on unchanged state" begin
    Random.seed!(1004)
    coords = rand(Float32, 3, 200)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    set_symmetric_search!(tns, id, id)

    run!(tns)
    perm1 = copy(tns.permutation[id])
    n_nodes_1 = tns.trees[id].n_nodes
    @test all(.!tns.dirty)  # nothing dirty after run!

    run!(tns)  # no point set changes since last run
    @test tns.permutation[id] == perm1
    @test tns.trees[id].n_nodes == n_nodes_1
    @test all(.!tns.dirty)
end

@testset "set_search_radius! after run! triggers rebuild" begin
    Random.seed!(1005)
    coords = rand(Float32, 3, 200)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    set_symmetric_search!(tns, id, id)
    run!(tns)
    @test all(.!tns.dirty)

    # Changing radius must mark every set dirty so the next run! rebuilds.
    set_search_radius!(tns, 0.05f0)
    @test all(tns.dirty)
    @test tns.radius == 0.05f0

    run!(tns)
    @test all(.!tns.dirty)
    # Neighbor count under tighter radius is ≤ count under looser radius for every point.
    n_loose = length(get_neighborlist(tns, id, id, 1))
    set_search_radius!(tns, 0.2f0)
    run!(tns)
    n_loose2 = length(get_neighborlist(tns, id, id, 1))
    @test n_loose2 >= n_loose
end

@testset "set_refit_mode! marks all sets dirty" begin
    Random.seed!(1006)
    coords = rand(Float32, 3, 100)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    set_symmetric_search!(tns, id, id)
    run!(tns)
    @test !tns.dirty[id]

    set_refit_mode!(tns, true)
    @test tns.dirty[id]
    @test tns.refit_mode == true

    set_refit_mode!(tns, false)
    @test tns.dirty[id]
    @test tns.refit_mode == false
end

@testset "resize_point_set! permits size change; update_point_set! does not" begin
    Random.seed!(1007)
    coords1 = rand(Float32, 3, 100)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords1)
    set_symmetric_search!(tns, id, id)
    run!(tns)

    # resize_point_set! supports a different N (or even 0).
    coords2 = rand(Float32, 3, 75)
    resize_point_set!(tns, id, coords2)
    @test tns.point_sets[id].n == Int32(75)
    @test tns.dirty[id]
    run!(tns)
    @test tns.point_sets[id].n == Int32(75)

    # update_point_set! requires same N.
    bad = rand(Float32, 3, 74)
    @test_throws DimensionMismatch update_point_set!(tns, id, bad)

    same_n = rand(Float32, 3, 75)
    update_point_set!(tns, id, same_n)
    @test tns.point_sets[id].n == Int32(75)
    @test tns.dirty[id]
end

@testset "update_point_set! preserves active_pairs and buffers" begin
    Random.seed!(1008)
    A = rand(Float32, 3, 100)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, A)
    set_symmetric_search!(tns, id, id)
    run!(tns)
    e1 = build_edges(tns, id, id)
    # Snapshot before the rebuild — the returned NamedTuple aliases the
    # EdgeBuffer's backing storage, so the second build_edges call will
    # overwrite e1's fields in place.
    s1 = copy(e1.senders)
    n1 = length(s1)

    pairs_before = copy(tns.active_pairs)
    n_bufs_before = length(tns.neighbor_buffers)

    # Move points: same N, fresh coords.
    A2 = rand(Float32, 3, 100)
    update_point_set!(tns, id, A2)
    run!(tns)
    e2 = build_edges(tns, id, id)

    @test tns.active_pairs == pairs_before
    @test length(tns.neighbor_buffers) == n_bufs_before
    # The edges reflect the new positions (results changed).
    @test length(e2.senders) != n1 || collect(e2.senders) != s1
end

@testset "for_each_neighbor does NOT filter self when q==t" begin
    # The lazy iterator is callback-only; the user opts into self-filtering
    # when q==t. get_neighborlist does this filter for them. Lock this contract
    # so callers can rely on it.
    Random.seed!(1009)
    coords = rand(Float32, 3, 30)
    tns = TNS(Float32); set_search_radius!(tns, 2.0f0)  # large: every point is its own nbr
    id = add_point_set!(tns, coords)
    set_symmetric_search!(tns, id, id)
    run!(tns)

    saw_self = false
    for_each_neighbor(tns, id, id, 7) do j
        if j == Int32(7)
            saw_self = true
        end
    end
    @test saw_self

    # get_neighborlist filters self out:
    @test !(Int32(7) in get_neighborlist(tns, id, id, 7))
end

@testset "build_edges returns NamedTuple aliased to underlying EdgeBuffer" begin
    Random.seed!(1010)
    coords = rand(Float32, 3, 150)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    set_symmetric_search!(tns, id, id)
    run!(tns)
    e = build_edges(tns, id, id)

    pair_idx = findfirst(==((Int32(id), Int32(id))), tns.active_pairs)
    buf = tns.edge_buffers[pair_idx]
    # The NamedTuple fields must alias the buffer (same backing storage,
    # not a fresh allocation per call).
    @test e.senders          === buf.senders
    @test e.receivers        === buf.receivers
    @test e.rel_displacement === buf.rel_displacement
    @test e.rel_dist_norm    === buf.rel_dist_norm
end

@testset "build_edges! errors when pair not registered" begin
    Random.seed!(1011)
    coords = rand(Float32, 3, 50)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    # Note: no set_active_search! / set_symmetric_search! before run!
    run!(tns)
    @test_throws ErrorException build_edges!(tns, id, id)
end

@testset "run! invalidates previously materialized buffers" begin
    Random.seed!(1012)
    coords = rand(Float32, 3, 150)
    tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
    id = add_point_set!(tns, coords)
    set_symmetric_search!(tns, id, id)
    run!(tns)
    materialize_all_neighbors!(tns)
    pair_idx = findfirst(==((Int32(id), Int32(id))), tns.active_pairs)
    @test tns.neighbor_buffers[pair_idx].materialized

    e = build_edges(tns, id, id)
    @test tns.edge_buffers[pair_idx].materialized

    # Re-running must clear materialized flags so callers can't read stale data
    # without an explicit refill.
    run!(tns)
    @test !tns.neighbor_buffers[pair_idx].materialized
    @test !tns.edge_buffers[pair_idx].materialized
end

@testset "two TNS instances are independent" begin
    Random.seed!(1013)
    A = rand(Float32, 3, 80)
    B = rand(Float32, 3, 120)

    tns1 = TNS(Float32); set_search_radius!(tns1, 0.1f0)
    a = add_point_set!(tns1, A)
    set_symmetric_search!(tns1, a, a)
    run!(tns1)

    tns2 = TNS(Float32); set_search_radius!(tns2, 0.2f0)
    b = add_point_set!(tns2, B)
    set_symmetric_search!(tns2, b, b)
    run!(tns2)

    @test length(tns1.point_sets) == 1
    @test length(tns2.point_sets) == 1
    @test tns1.radius == 0.1f0
    @test tns2.radius == 0.2f0
    @test tns1.point_sets[a].n == Int32(80)
    @test tns2.point_sets[b].n == Int32(120)
end

@testset "TNS{Float32,2} carries the right type parameters" begin
    tns3 = TNS(Float32; ndims=3)
    tns2 = TNS(Float32; ndims=2)
    @test tns3 isa TNS{Float32,3}
    @test tns2 isa TNS{Float32,2}
end

@testset "asymmetric search direction matters" begin
    # Querying A→B is not the same registration as B→A; both must be added
    # explicitly (or via set_symmetric_search!).
    Random.seed!(1014)
    A = rand(Float32, 3, 30)
    B = rand(Float32, 3, 40)
    tns = TNS(Float32); set_search_radius!(tns, 0.15f0)
    a = add_point_set!(tns, A); b = add_point_set!(tns, B)
    set_active_search!(tns, a, b)         # only one direction
    run!(tns)

    # Pair (a, b) exists; (b, a) does not.
    @test (Int32(a), Int32(b)) in tns.active_pairs
    @test !((Int32(b), Int32(a)) in tns.active_pairs)
    @test_throws ErrorException build_edges!(tns, b, a)
end
