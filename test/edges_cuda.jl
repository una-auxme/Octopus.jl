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
end
