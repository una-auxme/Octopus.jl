#
# Copyright (c) 2026 Josef Jouaux
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Test
using Octopus
using Random
using CUDA

if !CUDA.functional()
    @info "CUDA unavailable; skipping GPU edge tests"
else
    # Sort an edge tuple by (receiver, sender). Ties are broken by sender; the
    # full row of rel_displacement and rel_dist_norm follows the same perm so
    # element-wise comparison is meaningful.
    function _sort_edges(senders, receivers, rdisp, rdist)
        s_h = Array(senders)
        r_h = Array(receivers)
        d_h = Array(rdisp)
        n_h = Array(rdist)
        perm = sortperm(collect(zip(r_h, s_h)))
        return s_h[perm], r_h[perm], d_h[:, perm], n_h[:, perm]
    end

    @testset "build_edges CPU↔GPU parity" begin
        Random.seed!(200)
        N = 1_000
        r = 0.1f0
        cpu_coords = rand(Float32, 3, N)
        gpu_coords = CuArray(cpu_coords)

        tns_cpu = TNS(Float32); set_search_radius!(tns_cpu, r)
        cid = add_point_set!(tns_cpu, cpu_coords)
        set_active_search!(tns_cpu, cid, cid); run!(tns_cpu)
        e_cpu = build_edges(tns_cpu, cid, cid)

        tns_gpu = TNS(Float32); set_search_radius!(tns_gpu, r)
        gid = add_point_set!(tns_gpu, gpu_coords)
        set_active_search!(tns_gpu, gid, gid); run!(tns_gpu)
        e_gpu = build_edges(tns_gpu, gid, gid)

        # Filter out points within an ε band of the radius — last-ULP FMA
        # differences (CPU vs PTX) flip the inequality on a handful of pairs.
        ε = r * 1f-4
        function in_band(i::Integer, j::Integer)
            dx = cpu_coords[1, i] - cpu_coords[1, j]
            dy = cpu_coords[2, i] - cpu_coords[2, j]
            dz = cpu_coords[3, i] - cpu_coords[3, j]
            d = sqrt(dx*dx + dy*dy + dz*dz)
            return abs(d - r) < ε
        end

        cpu_pairs = Set((Int(e_cpu.receivers[k]), Int(e_cpu.senders[k]))
                        for k in 1:length(e_cpu.senders)
                        if !in_band(Int(e_cpu.receivers[k]), Int(e_cpu.senders[k])))
        gpu_pairs_h = collect(zip(Array(e_gpu.receivers), Array(e_gpu.senders)))
        gpu_pairs = Set((Int(p[1]), Int(p[2])) for p in gpu_pairs_h
                        if !in_band(Int(p[1]), Int(p[2])))
        @test cpu_pairs == gpu_pairs

        # Element-wise check on a stable subset (intersect of pairs).
        common = sort!(collect(cpu_pairs ∩ gpu_pairs))
        cpu_idx = Dict((Int(e_cpu.receivers[k]), Int(e_cpu.senders[k])) => k
                      for k in 1:length(e_cpu.senders))
        gpu_idx = Dict((Int(p[1]), Int(p[2])) => k for (k, p) in enumerate(gpu_pairs_h))
        gpu_disp = Array(e_gpu.rel_displacement)
        gpu_dist = Array(e_gpu.rel_dist_norm)
        for pair in common
            i_cpu = cpu_idx[pair]
            i_gpu = gpu_idx[pair]
            @test e_cpu.rel_displacement[:, i_cpu] ≈ gpu_disp[:, i_gpu] atol=1f-4
            @test e_cpu.rel_dist_norm[1, i_cpu]    ≈ gpu_dist[1, i_gpu] atol=1f-4
        end
    end

    @testset "build_edges GPU empty point set" begin
        # Regression: a query set with N=0 used to crash the count-pass kernel
        # launch (blocks=cld(0,128)=0 is invalid). Should now no-op cleanly.
        empty_coords = CuArray(zeros(Float32, 3, 0))
        tns = TNS(Float32); set_search_radius!(tns, 0.1f0)
        id = add_point_set!(tns, empty_coords)
        set_active_search!(tns, id, id); run!(tns)
        e = build_edges(tns, id, id)

        @test length(e.senders)   == 0
        @test length(e.receivers) == 0
        @test size(e.rel_displacement) == (3, 0)
        @test size(e.rel_dist_norm)    == (1, 0)
    end

    @testset "build_edges GPU output shape & types" begin
        Random.seed!(201)
        N = 500
        r = 0.12f0
        gpu_coords = CuArray(rand(Float32, 3, N))

        tns = TNS(Float32); set_search_radius!(tns, r)
        gid = add_point_set!(tns, gpu_coords)
        set_active_search!(tns, gid, gid); run!(tns)
        e = build_edges(tns, gid, gid)

        n_edges = length(e.senders)
        @test e.senders   isa CuArray{Int32}
        @test e.receivers isa CuArray{Int32}
        @test e.rel_displacement isa CuArray{Float32}
        @test e.rel_dist_norm    isa CuArray{Float32}
        @test size(e.rel_displacement) == (3, n_edges)
        @test size(e.rel_dist_norm)    == (1, n_edges)
    end

    @testset "build_edges GPU sender/receiver row alignment" begin
        # Regression: row k of every output array must describe the SAME edge.
        # senders[k]/receivers[k] are one (target,query) pair, and
        # rel_displacement[:,k] / rel_dist_norm[k] are that pair's geometry.
        # Nothing in the GPU path may sort one array without co-permuting the
        # rest. Verified element-wise against the raw coords.
        Random.seed!(202)
        N = 800
        r = 0.1f0
        coords = rand(Float32, 3, N)
        tns = TNS(Float32); set_search_radius!(tns, r)
        gid = add_point_set!(tns, CuArray(coords))
        set_active_search!(tns, gid, gid); run!(tns)
        e = build_edges(tns, gid, gid)

        s = Array(e.senders); rcv = Array(e.receivers)
        disp = Array(e.rel_displacement); dist = Array(e.rel_dist_norm)
        @test length(s) == length(rcv)

        ok_pair = true; ok_geom = true; ok_self = true
        for k in 1:length(s)
            i = Int(rcv[k]); j = Int(s[k])     # receiver=query i, sender=target j
            (1 <= i <= N && 1 <= j <= N) || (ok_pair = false; continue)
            i == j && (ok_self = false)         # self-pair must be excluded
            d = coords[:, i] .- coords[:, j]    # query − target (sign convention)
            nd = sqrt(sum(abs2, d))
            nd <= r * (1 + 1f-4) || (ok_pair = false)
            all(abs.(disp[:, k] .- d ./ r) .< 1f-4) || (ok_geom = false)
            abs(dist[1, k] - nd / r) < 1f-4 || (ok_geom = false)
        end
        @test ok_pair    # every (sender,receiver) is a real in-radius neighbor
        @test ok_self    # no self-loops on a self-pair search
        @test ok_geom    # displacement/distance row k matches that exact pair
    end

    @testset "GPU build node-capacity overflow is memory-safe" begin
        # Regression for a memory-safety bug in the GPU octree topology build.
        # `_build_level_kernel!` bumps the device node counter with an atomic
        # add BEFORE the capacity check and never rolls it back, so an overflow
        # attempt ends a level with counter > capacity. The next BFS round then
        # mapped threads onto ghost node ids in (capacity, counter], doing OOB
        # reads of node_first/node_last and OOB writes of node_children past
        # the size-`est` arrays. The kernel now rejects `g > capacity` before
        # any such access; the counter overshoot stays harmless (the overflow
        # flag still trips and the host retries with a larger estimate).
        #
        # This config (D=2, N=40000, r=0.072, leaf=32) overshoots the attempt-1
        # estimate `est`. We drive the real build kernels with the node arrays
        # padded past `est` and the trailing slots set to a canary, telling the
        # kernels capacity == est. Any clobbered canary cell is a write the
        # real (size-est) arrays could not hold — i.e. an out-of-bounds write.
        ext = Base.get_extension(Octopus, :OctopusCUDAExt)
        @test ext !== nothing
        morton_k  = ext._morton_bin_kernel!
        init_k    = ext._init_build_kernel!
        level_k   = ext._build_level_kernel!
        advance_k = ext._advance_level_kernel!

        function jittered_lattice(D, N, radius; seed = 1)
            Random.seed!(seed); s = radius / 1.6f0; per = ceil(Int, N^(1 / D))
            grids = ntuple(_ -> 0:(per - 1), D)
            pts = Matrix{Float32}(undef, D, per^D); idx = 1
            for c in Iterators.product(grids...)
                for d in 1:D
                    pts[d, idx] = (c[d] + 0.25f0 * (rand(Float32) - 0.5f0)) * s
                end
                idx += 1
            end
            return pts[:, 1:min(N, size(pts, 2))]
        end

        D = 2; NCH = 1 << D; N = 40_000; r = 0.072f0; leaf = Int32(32)
        coords_h = jittered_lattice(D, N, r)
        coords = CuArray(coords_h)
        n = size(coords, 2)
        est = max(NCH * cld(n, Int(leaf)) + 64, 64)   # attempt-1 capacity

        # --- origin / morton / sort: mirrors _build_gpu_tree! attempt 1 ---
        mins = vec(Array(minimum(coords; dims = 2)))
        origin = ntuple(d -> mins[d] - 1f-6, D)
        morton = CuArray{UInt64}(undef, n)
        threads = 256; gridb = cld(n, threads)
        CUDA.@cuda threads=threads blocks=gridb morton_k(
            morton, coords, origin[1], origin[2], 0f0, inv(r), Int32(n), Val(D))
        perm = Int32.(sortperm(morton))

        # --- topology with a canary guard region past `est` ---
        maxlev = 31
        PAD = 8192
        CANARY = Int32(1234567)                       # never a real node value
        nf     = CUDA.fill(CANARY, est + PAD)
        nl     = CUDA.fill(CANARY, est + PAD)
        nchild = CUDA.fill(CANARY, NCH, est + PAD)
        counter   = CuArray{Int32}(undef, 1)
        lev_start = CuArray{Int32}(undef, maxlev + 2)
        lev_end   = CuArray{Int32}(undef, maxlev + 2)
        overflow  = CUDA.zeros(Int32, 1)

        CUDA.@cuda threads=1 init_k(nf, nl, lev_start, lev_end, counter, Int32(n))
        for L in 0:maxlev
            CUDA.@cuda threads=threads blocks=gridb level_k(
                nf, nl, nchild, morton, perm, counter, lev_start, lev_end,
                Int32(est), overflow, leaf, Int32(L), Val(D))   # capacity = est
            CUDA.@cuda threads=1 advance_k(lev_start, lev_end, counter, Int32(L))
        end
        CUDA.synchronize()

        # The config must genuinely exercise overflow, else the guard is untested.
        @test (CUDA.@allowscalar overflow[1]) == Int32(1)
        @test (CUDA.@allowscalar counter[1]) > est

        # No write may land in the guard region [est+1 .. est+PAD].
        guard = (est + 1):(est + PAD)
        @test all(==(CANARY), Array(@view nf[guard]))
        @test all(==(CANARY), Array(@view nl[guard]))
        @test all(==(CANARY), Array(@view nchild[:, guard]))

        # End-to-end: the same config reaches the retry path through the public
        # API and yields a correct tree. Spot-check GPU edges for a sample of
        # query points against an O(N) brute-force ground truth.
        tns = TNS(Float32; ndims = D); set_search_radius!(tns, r)
        pid = add_point_set!(tns, CuArray(coords_h))
        set_active_search!(tns, pid, pid); run!(tns)
        @test Int(tns.trees[pid].n_nodes) > est       # retry produced the tree

        e = build_edges(tns, pid, pid)
        rcv = Array(e.receivers); snd = Array(e.senders)
        gpu = Dict{Int,Set{Int}}()
        for k in 1:length(rcv)
            push!(get!(gpu, Int(rcv[k]), Set{Int}()), Int(snd[k]))
        end
        Random.seed!(99)
        sample = rand(1:N, 200)
        r2 = r * r; band = r * 1f-4
        function neighbors_match(coords_h, gpu, sample, r2, band, r, N)
            for i in sample
                truth = Set{Int}()
                xi = coords_h[1, i]; yi = coords_h[2, i]
                for j in 1:N
                    j == i && continue
                    dx = coords_h[1, j] - xi; dy = coords_h[2, j] - yi
                    dx * dx + dy * dy <= r2 && push!(truth, j)
                end
                g = get(gpu, i, Set{Int}())
                for j in union(setdiff(g, truth), setdiff(truth, g))
                    dx = coords_h[1, j] - xi; dy = coords_h[2, j] - yi
                    # tolerate last-ULP boundary flips, like the parity test
                    abs(sqrt(dx * dx + dy * dy) - r) < band || return false
                end
            end
            return true
        end
        @test neighbors_match(coords_h, gpu, sample, r2, band, r, N)
    end
end
