#!/usr/bin/env julia
# Phase 0 of the v0.2 plan: profile each phase of both TreeNSearch and
# PointNeighbors on the GraphNetSim `point_neighbor_ns` GPU pipeline, so we
# can answer "is tree shape really the 2D bottleneck?".
#
# Phases instrumented per call (synchronized after each):
#   TreeNSearch GPU:       pad2to3 / construct / run! / count_kernel /
#                          cumsum / alloc_output / write_kernel
#   PointNeighbors GPU:    setup / construct / initialize / adapt /
#                          count_pass / cumsum / alloc_output / write_pass
#
# Run:  julia --project=benchmarks --threads=auto benchmarks/profile_phases.jl

using TreeNSearch
using PointNeighbors
using CUDA
using Adapt
using Random
using Printf
using Statistics

CUDA.functional() || error("CUDA not functional; cannot run GPU profile")

# Synchronized timer — block on GPU before measuring, after each phase.
@inline function synced_ns()
    CUDA.synchronize()
    return time_ns()
end

# Quick wrapper: run f() ntimes after a warmup, return median elapsed seconds.
function bench_segment(f, ntimes::Int = 5)
    f()  # warmup
    CUDA.synchronize()
    times = Float64[]
    for _ in 1:ntimes
        t0 = synced_ns()
        f()
        t1 = synced_ns()
        push!(times, (t1 - t0) / 1e9)
    end
    return median(times)
end

# ------------------------------------------------------------------------
# Phase-by-phase TreeNSearch
# ------------------------------------------------------------------------

function profile_tns(pos_orig::CuArray{Float32}, radius::Float32)
    D = size(pos_orig, 1)
    N = size(pos_orig, 2)

    # Pre-allocate everything once; re-running phases needs stable state.
    # We re-create per-iter where the phase intrinsically allocates fresh.

    pad2to3 = bench_segment() do
        if D == 3
            return pos_orig
        else
            p = CUDA.zeros(Float32, 3, N)
            p[1:D, :] .= pos_orig
            return p
        end
    end

    pos3 = if D == 3
        pos_orig
    else
        p = CUDA.zeros(Float32, 3, N)
        p[1:D, :] .= pos_orig
        p
    end

    construct = bench_segment() do
        tns = TNS(Float32; ndims = 3)
        set_search_radius!(tns, radius)
        add_point_set!(tns, pos3)
        return tns
    end

    # For run! we need a fresh `tns` each iter so dirty flags fire.
    runphase = bench_segment() do
        tns = TNS(Float32; ndims = 3)
        set_search_radius!(tns, radius)
        id = add_point_set!(tns, pos3)
        set_symmetric_search!(tns, id, id)
        run!(tns)
    end

    # Build a stable tns for the kernel-only phases.
    tns = TNS(Float32; ndims = 3)
    set_search_radius!(tns, radius)
    id = add_point_set!(tns, pos3)
    set_symmetric_search!(tns, id, id)
    run!(tns)
    dv = device_view(tns, id, id)

    threads = 128
    blocks = cld(N, threads)

    counts_buf = CUDA.zeros(Int32, N)

    # The count kernel definition is closure-friendly — pulled inline:
    function count_kernel!(counts, dv, N)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        i > N && return nothing
        @inbounds counts[i] = Int32(1)
        for_each_neighbor_device(dv, i) do j
            if j != Int32(i)
                @inbounds counts[i] += Int32(1)
            end
        end
        return nothing
    end

    count_phase = bench_segment() do
        @cuda threads = threads blocks = blocks count_kernel!(counts_buf, dv, Int32(N))
    end

    @cuda threads = threads blocks = blocks count_kernel!(counts_buf, dv, Int32(N))
    CUDA.synchronize()

    cumsum_phase = bench_segment() do
        offsets = CUDA.cumsum(counts_buf) .- counts_buf .+ Int32(1)
        return offsets
    end

    offsets = CUDA.cumsum(counts_buf) .- counts_buf .+ Int32(1)
    n_edges = Int(CUDA.@allowscalar offsets[end] + counts_buf[end] - 1)

    alloc_phase = bench_segment() do
        senders = CuArray{Int32}(undef, n_edges)
        receivers = CuArray{Int32}(undef, n_edges)
        rd  = CuArray{Float32}(undef, D, n_edges)
        rdn = CuArray{Float32}(undef, 1, n_edges)
        cursors = copy(offsets)
        return senders, receivers, rd, rdn, cursors
    end

    senders = CuArray{Int32}(undef, n_edges)
    receivers = CuArray{Int32}(undef, n_edges)
    rd  = CuArray{Float32}(undef, D, n_edges)
    rdn = CuArray{Float32}(undef, 1, n_edges)

    function write_kernel_3d!(senders, receivers, rd, rdn, cursors, dv, coords_orig, N, radius)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        i > N && return nothing
        i32 = Int32(i)
        @inbounds start = cursors[i]
        @inbounds senders[start]   = i32
        @inbounds receivers[start] = i32
        @inbounds rd[1, start] = 0f0; @inbounds rd[2, start] = 0f0; @inbounds rd[3, start] = 0f0
        @inbounds rdn[start] = 0f0
        @inbounds cursors[i] = start + Int32(1)
        px = @inbounds coords_orig[1, i]; py = @inbounds coords_orig[2, i]; pz = @inbounds coords_orig[3, i]
        for_each_neighbor_device(dv, i) do j
            if j != i32
                @inbounds oo = cursors[i]
                qx = @inbounds coords_orig[1, j]; qy = @inbounds coords_orig[2, j]; qz = @inbounds coords_orig[3, j]
                dx = px - qx; dy = py - qy; dz = pz - qz
                d2 = dx*dx + dy*dy + dz*dz
                @inbounds senders[oo] = j; @inbounds receivers[oo] = i32
                @inbounds rd[1, oo] = dx / radius; @inbounds rd[2, oo] = dy / radius; @inbounds rd[3, oo] = dz / radius
                @inbounds rdn[oo] = sqrt(d2) / radius
                @inbounds cursors[i] = oo + Int32(1)
            end
        end
        return nothing
    end

    function write_kernel_2d!(senders, receivers, rd, rdn, cursors, dv, coords_orig, N, radius)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        i > N && return nothing
        i32 = Int32(i)
        @inbounds start = cursors[i]
        @inbounds senders[start]   = i32; @inbounds receivers[start] = i32
        @inbounds rd[1, start] = 0f0; @inbounds rd[2, start] = 0f0
        @inbounds rdn[start] = 0f0
        @inbounds cursors[i] = start + Int32(1)
        px = @inbounds coords_orig[1, i]; py = @inbounds coords_orig[2, i]
        for_each_neighbor_device(dv, i) do j
            if j != i32
                @inbounds oo = cursors[i]
                qx = @inbounds coords_orig[1, j]; qy = @inbounds coords_orig[2, j]
                dx = px - qx; dy = py - qy
                d2 = dx*dx + dy*dy
                @inbounds senders[oo] = j; @inbounds receivers[oo] = i32
                @inbounds rd[1, oo] = dx / radius; @inbounds rd[2, oo] = dy / radius
                @inbounds rdn[oo] = sqrt(d2) / radius
                @inbounds cursors[i] = oo + Int32(1)
            end
        end
        return nothing
    end

    write_phase = bench_segment() do
        cursors = copy(offsets)
        if D == 3
            @cuda threads = threads blocks = blocks write_kernel_3d!(
                senders, receivers, rd, rdn, cursors, dv, pos_orig, Int32(N), radius)
        else
            @cuda threads = threads blocks = blocks write_kernel_2d!(
                senders, receivers, rd, rdn, cursors, dv, pos_orig, Int32(N), radius)
        end
    end

    return (
        pad2to3      = pad2to3,
        construct    = construct,
        run_phase    = runphase,
        count_phase  = count_phase,
        cumsum_phase = cumsum_phase,
        alloc_phase  = alloc_phase,
        write_phase  = write_phase,
        n_edges      = n_edges,
    )
end

# ------------------------------------------------------------------------
# Phase-by-phase PointNeighbors
# ------------------------------------------------------------------------

function _pn_count_pass!(n_neighbors_gpu, pos, nhs_gpu)
    n_neighbors_gpu .= 0
    foreach_point_neighbor(pos, pos, nhs_gpu) do i, _, _, _
        n_neighbors_gpu[i] += 1
    end
    return n_neighbors_gpu
end

function _pn_write_pass!(senders, receivers, rd, rdn, nhs_gpu, pos, n_neighbors_gpu, radius, D)
    offset = CUDA.cumsum(n_neighbors_gpu) .- n_neighbors_gpu .+ 1
    foreach_point_neighbor(pos, pos, nhs_gpu) do i, j, pos_diff, distance
        receivers[offset[i]] = i
        senders[offset[i]] = j
        for d in 1:D
            rd[d, offset[i]] = pos_diff[d] / radius
        end
        rdn[offset[i]] = distance / radius
        offset[i] += 1
    end
    return nothing
end

function _make_nhs_pn(pos::CuArray{Float32}, radius::Float32)
    min_corner = minimum(pos; dims = 2)
    max_corner = maximum(pos; dims = 2)
    D = size(pos, 1); N = size(pos, 2)
    extent = Array(vec(max_corner - min_corner))
    cell_vol = prod(extent) / prod(ceil.(extent ./ radius))
    avg_per_cell = N * cell_vol / max(prod(extent), eps(Float32))
    mppc = max(Int(ceil(4 * avg_per_cell)) + 32, 100)
    return GridNeighborhoodSearch{D}(;
        search_radius = radius, n_points = N,
        cell_list = FullGridCellList(; min_corner, max_corner,
                                     search_radius = radius,
                                     max_points_per_cell = mppc),
    )
end

function profile_pn(pos::CuArray{Float32}, radius::Float32)
    D = size(pos, 1); N = size(pos, 2)

    setup = bench_segment() do
        min_corner = minimum(pos; dims = 2)
        max_corner = maximum(pos; dims = 2)
        return (min_corner, max_corner)
    end

    construct = bench_segment() do
        nhs = _make_nhs_pn(pos, radius)
        return nhs
    end

    init_phase = bench_segment() do
        nhs = _make_nhs_pn(pos, radius)
        initialize!(nhs, Array(pos), Array(pos))
        return nhs
    end

    nhs = _make_nhs_pn(pos, radius)
    initialize!(nhs, Array(pos), Array(pos))

    adapt_phase = bench_segment() do
        adapt(CUDABackend(), nhs)
    end

    nhs_gpu = adapt(CUDABackend(), nhs)

    n_neighbors_gpu = CUDA.zeros(Int, N)
    count_phase = bench_segment() do
        _pn_count_pass!(n_neighbors_gpu, pos, nhs_gpu)
    end
    _pn_count_pass!(n_neighbors_gpu, pos, nhs_gpu)
    n_edges = Int(sum(n_neighbors_gpu))

    cumsum_phase = bench_segment() do
        offset = CUDA.cumsum(n_neighbors_gpu) .- n_neighbors_gpu .+ 1
    end

    alloc_phase = bench_segment() do
        senders = CuArray{Int32}(undef, n_edges)
        receivers = CuArray{Int32}(undef, n_edges)
        rd  = CuArray{Float32}(undef, D, n_edges)
        rdn = CuArray{Float32}(undef, 1, n_edges)
        return senders, receivers, rd, rdn
    end

    senders = CuArray{Int32}(undef, n_edges)
    receivers = CuArray{Int32}(undef, n_edges)
    rel_displacement = CuArray{Float32}(undef, D, n_edges)
    rel_dist_norm    = CuArray{Float32}(undef, 1, n_edges)

    write_phase = bench_segment() do
        _pn_write_pass!(senders, receivers, rel_displacement, rel_dist_norm,
                        nhs_gpu, pos, n_neighbors_gpu, radius, D)
    end

    return (
        setup        = setup,
        construct    = construct,
        init_phase   = init_phase,
        adapt_phase  = adapt_phase,
        count_phase  = count_phase,
        cumsum_phase = cumsum_phase,
        alloc_phase  = alloc_phase,
        write_phase  = write_phase,
        n_edges      = n_edges,
    )
end

# ------------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------------

ms(t) = @sprintf("%7.2f", t * 1e3)

function report_tns(p)
    total = p.run_phase + p.count_phase + p.cumsum_phase + p.alloc_phase + p.write_phase + p.pad2to3
    println("  TreeNSearch phases (ms):")
    @printf("    pad 2D->3D       : %s\n", ms(p.pad2to3))
    @printf("    run! (build+up)  : %s   <- bin/sort/build/refit/upload\n", ms(p.run_phase))
    @printf("    count_kernel     : %s\n", ms(p.count_phase))
    @printf("    cumsum offsets   : %s\n", ms(p.cumsum_phase))
    @printf("    alloc output     : %s\n", ms(p.alloc_phase))
    @printf("    write_kernel     : %s\n", ms(p.write_phase))
    @printf("    --- sum (excl construct) %s ms; n_edges = %d\n", ms(total), p.n_edges)
    return total
end

function report_pn(p)
    total = p.init_phase + p.adapt_phase + p.count_phase + p.cumsum_phase + p.alloc_phase + p.write_phase
    println("  PointNeighbors phases (ms):")
    @printf("    setup (min/max)  : %s\n", ms(p.setup))
    @printf("    construct nhs    : %s\n", ms(p.construct))
    @printf("    initialize! (CPU): %s\n", ms(p.init_phase))
    @printf("    adapt to GPU     : %s\n", ms(p.adapt_phase))
    @printf("    count pass       : %s\n", ms(p.count_phase))
    @printf("    cumsum offsets   : %s\n", ms(p.cumsum_phase))
    @printf("    alloc output     : %s\n", ms(p.alloc_phase))
    @printf("    write pass       : %s\n", ms(p.write_phase))
    @printf("    --- sum (excl setup/construct) %s ms; n_edges = %d\n", ms(total), p.n_edges)
    return total
end

# ------------------------------------------------------------------------
# Scenarios — same as the GraphNetSim benchmarks
# ------------------------------------------------------------------------

struct Scenario
    name::String
    pos::Matrix{Float32}
    radius::Float32
end

function build_scenarios()
    scenarios = Scenario[]
    Random.seed!(42)
    for N in (1_000, 5_000, 20_000)
        pos = Matrix{Float32}(undef, 2, N)
        pos[1, :] = 0.3f0  .* rand(Float32, N)
        pos[2, :] = 0.15f0 .* rand(Float32, N)
        push!(scenarios, Scenario("2D dam-break-like  N=$(lpad(N,6))", pos, 0.072f0))
    end
    Random.seed!(43)
    for (N, r) in ((50_000, 0.02f0), (200_000, 0.01f0))
        pos = rand(Float32, 3, N)
        push!(scenarios, Scenario("3D uniform         N=$(lpad(N,6))", pos, r))
    end
    return scenarios
end

function main()
    println("Phase profile — TreeNSearch.jl vs PointNeighbors.jl GPU paths")
    println("Julia: ", VERSION, "  threads: ", Threads.nthreads())
    println("GPU:   ", name(CUDA.device()))
    println()

    for sc in build_scenarios()
        println("=" ^ 100)
        println(sc.name, "   (D=", size(sc.pos, 1), ", r=", sc.radius, ")")
        println("-" ^ 100)
        gpu_pos = CuArray(sc.pos)

        ptns = profile_tns(gpu_pos, sc.radius)
        pn   = profile_pn(gpu_pos, sc.radius)

        total_tns = report_tns(ptns); println()
        total_pn  = report_pn(pn);    println()

        @printf("  TOTAL (median): TNS = %s ms,  PN = %s ms,  ratio TNS/PN = %.2fx\n",
                ms(total_tns), ms(total_pn), total_tns / total_pn)
        println()

        # If we're 2D, compute "what would removing the 3D-pad savings give us?"
        if size(sc.pos, 1) == 2
            tree_phases = ptns.run_phase + ptns.count_phase + ptns.write_phase
            other_phases = ptns.cumsum_phase + ptns.alloc_phase + ptns.pad2to3
            tree_frac = tree_phases / (tree_phases + other_phases)
            @printf("  Tree-shape-sensitive phases (run! + count + write) = %s ms = %.1f%% of TNS total\n",
                    ms(tree_phases), 100 * tree_frac)
            @printf("  Output/setup phases (cumsum + alloc + pad)         = %s ms = %.1f%% of TNS total\n",
                    ms(other_phases), 100 * (1 - tree_frac))
            println("  ^^^ This % is the upper bound on what native 2D could improve")
        end

        CUDA.reclaim()
        println()
    end
end

main()
