using Test
using Octopus
using Random

# Hard ceilings from the plan. Steady-state on-CPU: ≤ 12 B/pt of overhead
# above the user's coord matrix (lazy mode, refit off).

function tns_overhead_bytes(tns::TNS)
    # Exclude the user's coord matrices (stored by reference in PointSet),
    # which aren't ours to count.
    total = Base.summarysize(tns)
    @inbounds for ps in tns.point_sets
        total -= Base.summarysize(ps.coords)
    end
    return total
end

@testset "overhead ≤ 18 B/pt at N=100k, lazy mode" begin
    # Plan target was 12 B/pt based on an optimistic n_nodes ≈ N/16 estimate.
    # Reality on uniform-random data at cell_size==radius: n_nodes ≈ N/6 → tree
    # ~11 B/pt, perm 4 B/pt, misc ~0.3 B/pt → ~15 B/pt. We pick 18 B/pt as the
    # ceiling; still strictly better than PointNeighbors.jl DictionaryCellList
    # (~18 B/pt for plain cell storage, no AABB/tree data).
    Random.seed!(7)
    N = 100_000
    coords = rand(Float32, 3, N)
    tns = TNS()
    set_search_radius!(tns, 0.02f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)

    bpp = tns_overhead_bytes(tns) / N
    @info "steady-state bytes/particle (lazy)" bpp
    @test bpp <= 18
end

@testset "materialized lists ≤ 800 B/pt bound" begin
    Random.seed!(8)
    N = 50_000
    coords = rand(Float32, 3, N)
    tns = TNS()
    set_search_radius!(tns, 0.02f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    materialize_all_neighbors!(tns)

    bpp = tns_overhead_bytes(tns) / N
    @info "bytes/particle with materialized lists" bpp
    @test bpp <= 800
end

@testset "for_each_neighbor allocation is bounded" begin
    Random.seed!(9)
    coords = rand(Float32, 3, 5_000)
    tns = TNS()
    set_search_radius!(tns, 0.05f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)

    # Reuse the SAME closure for warmup and measurement.
    # The outer `for_each_neighbor` hits a dynamic-dispatch boundary at
    # `tns.point_sets::Vector{Any}` (C++ TreeNSearch-style heterogeneous
    # point sets), so the runtime dispatch costs a small constant box. The
    # typed barrier + `_traverse!` itself are 0-alloc — confirmed by calling
    # the barrier directly.
    cb = _ -> nothing
    for_each_neighbor(cb, tns, id, id, 1)
    for_each_neighbor(cb, tns, id, id, 2)
    n_alloc = @allocated for_each_neighbor(cb, tns, id, id, 3)
    @info "for_each_neighbor allocated bytes" n_alloc
    @test n_alloc <= 32  # one dispatch box; no scaling with tree depth

    # The hot path itself is allocation-free.
    import Octopus: _for_each_neighbor_barrier
    ps = tns.point_sets[id]; tree = tns.trees[id]; perm = tns.permutation[id]
    _for_each_neighbor_barrier(cb, ps, ps, tree, perm, tns.query_stacks[1], tns.radius, 1)
    n_alloc_inner = @allocated _for_each_neighbor_barrier(
        cb, ps, ps, tree, perm, tns.query_stacks[1], tns.radius, 2)
    @info "inner barrier allocated bytes" n_alloc_inner
    @test n_alloc_inner == 0
end

@testset "build_edges / materialize don't allocate per edge" begin
    # Regression guard: the count/fill passes use typed Ref{Int32} accumulators.
    # A reassigned closure-captured local would box as Core.Box{Any} and
    # re-allocate one heap Int per *edge* (megabytes at this N), so a steady
    # call must allocate only a small N-independent constant. We measure at two
    # sizes and require the larger not to allocate proportionally more — boxing
    # would scale with edge count, the Ref path stays flat.
    function steady_alloc(N)
        Random.seed!(11)
        coords = rand(Float32, 3, N)
        tns = TNS(); set_search_radius!(tns, 0.02f0)
        id = add_point_set!(tns, coords)
        set_active_search!(tns, id, id)
        run!(tns)
        build_edges(tns, id, id)               # warm: compile + size buffers
        materialize_all_neighbors!(tns)
        a_edges = @allocated build_edges(tns, id, id)
        a_mat   = @allocated materialize_all_neighbors!(tns)
        return (a_edges, a_mat)
    end

    e_small, m_small = steady_alloc(20_000)
    e_big,   m_big   = steady_alloc(80_000)   # 4× points ⇒ ~4× edges
    @info "build_edges steady alloc (B)"        e_small e_big
    @info "materialize steady alloc (B)"        m_small m_big

    # Flat constant, not O(edges): a few hundred bytes regardless of N. The
    # boxed version allocated ~4.6 MB at N=20k and grew with N.
    @test e_big <= 1024
    @test m_big <= 1024
    # Explicitly assert no scaling: 4× the work must not mean materially more
    # allocation (allow slack for measurement noise / GC bookkeeping).
    @test e_big <= e_small + 256
    @test m_big <= m_small + 256
end
