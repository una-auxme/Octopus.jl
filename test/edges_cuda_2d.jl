using Test
using TreeNSearch
using Random
using CUDA

if !CUDA.functional()
    @info "CUDA unavailable; skipping 2D GPU edge tests"
else
    function _sort_edges_2d(senders, receivers, rdisp, rdist)
        s_h = Array(senders)
        r_h = Array(receivers)
        d_h = Array(rdisp)
        n_h = Array(rdist)
        perm = sortperm(collect(zip(r_h, s_h)))
        return s_h[perm], r_h[perm], d_h[:, perm], n_h[:, perm]
    end

    @testset "2D build_edges CPU↔GPU parity" begin
        Random.seed!(500)
        N = 1_000
        r = 0.1f0
        cpu_coords = rand(Float32, 2, N)
        gpu_coords = CuArray(cpu_coords)

        tns_cpu = TNS(Float32; ndims=2); set_search_radius!(tns_cpu, r)
        cid = add_point_set!(tns_cpu, cpu_coords)
        set_symmetric_search!(tns_cpu, cid, cid); run!(tns_cpu)
        e_cpu = build_edges(tns_cpu, cid, cid)

        tns_gpu = TNS(Float32; ndims=2); set_search_radius!(tns_gpu, r)
        gid = add_point_set!(tns_gpu, gpu_coords)
        set_symmetric_search!(tns_gpu, gid, gid); run!(tns_gpu)
        e_gpu = build_edges(tns_gpu, gid, gid)

        ε = r * 1f-4
        function in_band(i::Integer, j::Integer)
            dx = cpu_coords[1, i] - cpu_coords[1, j]
            dy = cpu_coords[2, i] - cpu_coords[2, j]
            d = sqrt(dx*dx + dy*dy)
            return abs(d - r) < ε
        end

        cpu_pairs = Set((Int(e_cpu.receivers[k]), Int(e_cpu.senders[k]))
                        for k in 1:length(e_cpu.senders)
                        if !in_band(Int(e_cpu.receivers[k]), Int(e_cpu.senders[k])))
        gpu_pairs_h = collect(zip(Array(e_gpu.receivers), Array(e_gpu.senders)))
        gpu_pairs = Set((Int(p[1]), Int(p[2])) for p in gpu_pairs_h
                        if !in_band(Int(p[1]), Int(p[2])))
        @test cpu_pairs == gpu_pairs

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

    @testset "2D build_edges GPU output shape & types" begin
        Random.seed!(501)
        N = 500
        r = 0.12f0
        gpu_coords = CuArray(rand(Float32, 2, N))

        tns = TNS(Float32; ndims=2); set_search_radius!(tns, r)
        gid = add_point_set!(tns, gpu_coords)
        set_symmetric_search!(tns, gid, gid); run!(tns)
        e = build_edges(tns, gid, gid)

        n_edges = length(e.senders)
        @test e.senders   isa CuArray{Int32}
        @test e.receivers isa CuArray{Int32}
        @test e.rel_displacement isa CuArray{Float32}
        @test e.rel_dist_norm    isa CuArray{Float32}
        @test size(e.rel_displacement) == (2, n_edges)
        @test size(e.rel_dist_norm)    == (1, n_edges)
    end
end
