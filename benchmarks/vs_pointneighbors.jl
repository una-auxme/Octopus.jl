#!/usr/bin/env julia
# Head-to-head comparison: Octopus.jl vs PointNeighbors.jl.
#
# For each configuration (N, radius, distribution) we measure:
#   - Correctness: total neighbor count agrees with the brute-force reference.
#   - Build time: wall time to prepare the acceleration structure from scratch.
#   - Query time: wall time to iterate all neighbors of all points (zero-alloc
#     callback in both libraries).
#   - Memory: Base.summarysize of the final struct, with user coord matrices
#     subtracted so we only report overhead the library owns.
#
# Run:  julia --project=benchmarks --threads=auto benchmarks/vs_pointneighbors.jl

using Octopus
using PointNeighbors
using BenchmarkTools
using Random
using Printf
using Statistics

# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------

# Exclude the user's coord matrix so we only measure library overhead.
function overhead_bytes(search, coords)
    return Base.summarysize(search) - Base.summarysize(coords)
end

# Brute-force total neighbor count for self-search.
function brute_total(coords::AbstractMatrix{T}, r::T) where {T}
    n = size(coords, 2)
    r_sq = r * r
    total = 0
    @inbounds for i in 1:n
        for j in 1:n
            i == j && continue
            dx = coords[1, j] - coords[1, i]
            dy = coords[2, j] - coords[2, i]
            dz = coords[3, j] - coords[3, i]
            d2 = dx*dx + dy*dy + dz*dz
            if d2 <= r_sq
                total += 1
            end
        end
    end
    return total
end

# Total neighbor count via Octopus (self-search; excludes self).
function tns_total(coords::AbstractMatrix{T}, r::T) where {T}
    tns = TNS(T)
    set_search_radius!(tns, r)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    total = Threads.Atomic{Int}(0)
    n = size(coords, 2)
    for i in 1:n
        i32 = Int32(i)
        for_each_neighbor(tns, id, id, i) do j
            j != i32 && Threads.atomic_add!(total, 1)
        end
    end
    return (total[], tns)
end

# Total neighbor count via PointNeighbors GridNeighborhoodSearch.
function pn_grid_total(coords::AbstractMatrix{T}, r; cell_list = nothing) where {T}
    n = size(coords, 2)
    nhs = if cell_list === nothing
        GridNeighborhoodSearch{3}(search_radius = r, n_points = n)
    else
        GridNeighborhoodSearch{3}(search_radius = r, n_points = n, cell_list = cell_list)
    end
    initialize!(nhs, coords, coords)
    total = Ref(0)
    foreach_point_neighbor(coords, coords, nhs) do i, j, pos_diff, d
        i != j && (total[] += 1)
    end
    return (total[], nhs)
end

function pn_precomputed_total(coords::AbstractMatrix{T}, r) where {T}
    n = size(coords, 2)
    nhs = PrecomputedNeighborhoodSearch{3}(search_radius = r, n_points = n)
    initialize!(nhs, coords, coords)
    total = Ref(0)
    foreach_point_neighbor(coords, coords, nhs) do i, j, pos_diff, d
        i != j && (total[] += 1)
    end
    return (total[], nhs)
end

# ------------------------------------------------------------------------
# Timing helpers — a single run for each to stay within a sensible budget.
# ------------------------------------------------------------------------

function time_tns_build_query(coords::AbstractMatrix{T}, r::T) where {T}
    # Build
    build_bench = @benchmark begin
        tns = TNS($T)
        set_search_radius!(tns, $r)
        id = add_point_set!(tns, $coords)
        set_active_search!(tns, id, id)
        run!(tns)
    end samples=3 evals=1 seconds=30

    # Prepare once for query timing.
    tns = TNS(T); set_search_radius!(tns, r)
    id = add_point_set!(tns, coords); set_active_search!(tns, id, id); run!(tns)
    n = size(coords, 2)
    total_ref = Threads.Atomic{Int}(0)
    query_bench = @benchmark begin
        $total_ref[] = 0
        # Parallel per-point query to match PointNeighbors.jl's default
        # (`foreach_point_neighbor` is @threaded internally).
        Threads.@threads for i in 1:$n
            i32 = Int32(i)
            for_each_neighbor($tns, $id, $id, i) do j
                j != i32 && Threads.atomic_add!($total_ref, 1)
            end
        end
    end samples=3 evals=1 seconds=30

    bytes = overhead_bytes(tns, coords)
    return (build_bench, query_bench, bytes, total_ref[])
end

function time_pn_grid_dict(coords::AbstractMatrix{T}, r::T) where {T}
    n = size(coords, 2)
    build_bench = @benchmark begin
        nhs = GridNeighborhoodSearch{3}(search_radius = $r, n_points = $n)
        initialize!(nhs, $coords, $coords)
    end samples=3 evals=1 seconds=30

    nhs = GridNeighborhoodSearch{3}(search_radius = r, n_points = n)
    initialize!(nhs, coords, coords)
    total_ref = Threads.Atomic{Int}(0)
    query_bench = @benchmark begin
        $total_ref[] = 0
        foreach_point_neighbor($coords, $coords, $nhs) do i, j, pos_diff, d
            i != j && Threads.atomic_add!($total_ref, 1)
        end
    end samples=3 evals=1 seconds=30
    return (build_bench, query_bench, overhead_bytes(nhs, coords), total_ref[])
end

function time_pn_grid_hash(coords::AbstractMatrix{T}, r::T) where {T}
    n = size(coords, 2)
    build_bench = @benchmark begin
        cl = SpatialHashingCellList{3}(list_size = 3 * $n)
        nhs = GridNeighborhoodSearch{3}(search_radius = $r, n_points = $n, cell_list = cl)
        initialize!(nhs, $coords, $coords)
    end samples=3 evals=1 seconds=30

    cl = SpatialHashingCellList{3}(list_size = 3 * n)
    nhs = GridNeighborhoodSearch{3}(search_radius = r, n_points = n, cell_list = cl)
    initialize!(nhs, coords, coords)
    total_ref = Threads.Atomic{Int}(0)
    query_bench = @benchmark begin
        $total_ref[] = 0
        foreach_point_neighbor($coords, $coords, $nhs) do i, j, pos_diff, d
            i != j && Threads.atomic_add!($total_ref, 1)
        end
    end samples=3 evals=1 seconds=30
    return (build_bench, query_bench, overhead_bytes(nhs, coords), total_ref[])
end

function time_pn_precomputed_build_query(coords::AbstractMatrix{T}, r::T) where {T}
    n = size(coords, 2)
    build_bench = @benchmark begin
        nhs = PrecomputedNeighborhoodSearch{3}(search_radius = $r, n_points = $n)
        initialize!(nhs, $coords, $coords)
    end samples=3 evals=1 seconds=30

    nhs = PrecomputedNeighborhoodSearch{3}(search_radius = r, n_points = n)
    initialize!(nhs, coords, coords)

    total_ref = Threads.Atomic{Int}(0)
    query_bench = @benchmark begin
        $total_ref[] = 0
        foreach_point_neighbor($coords, $coords, $nhs) do i, j, pos_diff, d
            i != j && Threads.atomic_add!($total_ref, 1)
        end
    end samples=3 evals=1 seconds=30

    bytes = overhead_bytes(nhs, coords)
    return (build_bench, query_bench, bytes, total_ref[])
end

# ------------------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------------------

struct Scenario
    name::String
    coords::Matrix{Float32}
    radius::Float32
end

function build_scenarios()
    Random.seed!(42)
    scs = Scenario[]
    push!(scs, Scenario("uniform  N=  10_000, r=0.05", rand(Float32, 3,  10_000), 0.05f0))
    push!(scs, Scenario("uniform  N= 100_000, r=0.02", rand(Float32, 3, 100_000), 0.02f0))
    push!(scs, Scenario("uniform  N= 500_000, r=0.01", rand(Float32, 3, 500_000), 0.01f0))

    # Clustered: two blobs
    Random.seed!(42)
    a = 0.1f0 .* randn(Float32, 3, 25_000) .+ 0.25f0
    b = 0.1f0 .* randn(Float32, 3, 25_000) .- 0.25f0
    push!(scs, Scenario("clustered N=  50_000, r=0.03", hcat(a, b), 0.03f0))
    return scs
end

# ------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------

function fmt_time(ns)
    if ns < 1e6
        return @sprintf("%7.2f μs", ns/1e3)
    elseif ns < 1e9
        return @sprintf("%7.2f ms", ns/1e6)
    else
        return @sprintf("%7.2f  s", ns/1e9)
    end
end
fmt_bpp(b, n) = @sprintf("%8.2f B/pt (%6.2f MB total)", b/n, b/2^20)

function run_all()
    println("\nOctopus.jl vs PointNeighbors.jl")
    println("Julia: ", VERSION, "  threads: ", Threads.nthreads())
    println()

    for sc in build_scenarios()
        n = size(sc.coords, 2)
        r = sc.radius
        println("=" ^ 100)
        println(sc.name, "   (N=", n, ", r=", r, ")")
        println("-" ^ 100)

        # Ground truth — only for small N to keep things tractable.
        if n <= 50_000
            bf = brute_total(sc.coords, r)
            @printf("%-32s %s\n", "brute-force total neighbors", string(bf))
        else
            bf = nothing
            println("(skipping brute force — N too large)")
        end

        println()
        @printf("%-32s %12s %12s %12s %s\n",
                "backend", "build", "query", "correct?", "memory overhead")
        println("-" ^ 100)

        # Octopus
        b, q, mem, tot = time_tns_build_query(sc.coords, r)
        ok = bf === nothing ? "(N/A)" : (tot == bf ? "yes" : "NO ($tot vs $bf)")
        @printf("%-32s %12s %12s %12s %s\n",
                "Octopus.jl", fmt_time(median(b.times)), fmt_time(median(q.times)),
                ok, fmt_bpp(mem, n))

        # PN Grid + DictionaryCellList (default)
        b, q, mem, tot = time_pn_grid_dict(sc.coords, r)
        ok = bf === nothing ? "(N/A)" : (tot == bf ? "yes" : "NO ($tot vs $bf)")
        @printf("%-32s %12s %12s %12s %s\n",
                "PN Grid+DictionaryCellList",
                fmt_time(median(b.times)), fmt_time(median(q.times)),
                ok, fmt_bpp(mem, n))

        # PN Grid + SpatialHashingCellList
        try
            b, q, mem, tot = time_pn_grid_hash(sc.coords, r)
            ok = bf === nothing ? "(N/A)" : (tot == bf ? "yes" : "NO ($tot vs $bf)")
            @printf("%-32s %12s %12s %12s %s\n",
                    "PN Grid+SpatialHashingCellList",
                    fmt_time(median(b.times)), fmt_time(median(q.times)),
                    ok, fmt_bpp(mem, n))
        catch err
            @printf("%-32s %s\n", "PN Grid+SpatialHashingCellList", "error: $(typeof(err))")
        end

        # PN Precomputed (only at sensible N; materialises a full neighbor matrix)
        if n <= 100_000
            b, q, mem, tot = time_pn_precomputed_build_query(sc.coords, r)
            ok = bf === nothing ? "(N/A)" : (tot == bf ? "yes" : "NO ($tot vs $bf)")
            @printf("%-32s %12s %12s %12s %s\n",
                    "PN Precomputed",
                    fmt_time(median(b.times)), fmt_time(median(q.times)),
                    ok, fmt_bpp(mem, n))
        else
            @printf("%-32s %s\n", "PN Precomputed", "(skipped; N too large for materialised lists)")
        end

        println()
    end
end

run_all()
