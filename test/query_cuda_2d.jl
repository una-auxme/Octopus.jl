using Test
using TreeNSearch
using Random
using CUDA

if !CUDA.functional()
    @info "CUDA unavailable; skipping 2D GPU tests"
else
    @testset "2D device_view + for_each_neighbor_device correctness" begin
        Random.seed!(400)
        N = 1_000
        r = 0.1f0

        cpu_coords = rand(Float32, 2, N)
        gpu_coords = CuArray(cpu_coords)

        tns_cpu = TNS(Float32; ndims=2)
        set_search_radius!(tns_cpu, r)
        cid = add_point_set!(tns_cpu, cpu_coords)
        set_active_search!(tns_cpu, cid, cid)
        run!(tns_cpu)

        tns_gpu = TNS(Float32; ndims=2)
        set_search_radius!(tns_gpu, r)
        gid = add_point_set!(tns_gpu, gpu_coords)
        set_active_search!(tns_gpu, gid, gid)
        run!(tns_gpu)

        dv = device_view(tns_gpu, gid, gid)
        counts = CUDA.zeros(Int32, N)

        function count_kernel_2d!(counts, dv, N)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            i > N && return nothing
            i32 = Int32(i)
            c = Int32(0)
            TreeNSearch.@for_each_neighbor_device_inline_2d dv i j begin
                if j != i32
                    c += Int32(1)
                end
            end
            @inbounds counts[i] = c
            return nothing
        end

        threads = 128
        blocks = cld(N, threads)
        @cuda threads=threads blocks=blocks count_kernel_2d!(counts, dv, Int32(N))
        CUDA.synchronize()

        host_counts = Array(counts)

        for i in 1:N
            cpu_list = get_neighborlist(tns_cpu, cid, cid, i)
            cpu_n = length(cpu_list)
            gpu_n = Int(host_counts[i])
            @test abs(gpu_n - cpu_n) <= 2  # tolerance band for radius boundary
        end
    end

    @testset "2D CPU↔GPU parity on neighbor membership (tolerance band)" begin
        Random.seed!(401)
        N = 500
        r = 0.12f0
        cpu_coords = rand(Float32, 2, N)
        gpu_coords = CuArray(cpu_coords)

        tns_cpu = TNS(Float32; ndims=2); set_search_radius!(tns_cpu, r)
        cid = add_point_set!(tns_cpu, cpu_coords)
        set_active_search!(tns_cpu, cid, cid); run!(tns_cpu)

        tns_gpu = TNS(Float32; ndims=2); set_search_radius!(tns_gpu, r)
        gid = add_point_set!(tns_gpu, gpu_coords)
        set_active_search!(tns_gpu, gid, gid); run!(tns_gpu)

        MAX_NBRS = 200
        lists = CUDA.fill(Int32(-1), MAX_NBRS, N)
        cnts  = CUDA.zeros(Int32, N)

        function collect_kernel_2d!(lists, cnts, dv, N, maxn)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            i > N && return nothing
            i32 = Int32(i)
            k = Int32(0)
            TreeNSearch.@for_each_neighbor_device_inline_2d dv i j begin
                if j != i32 && k < Int32(maxn)
                    k += Int32(1)
                    @inbounds lists[k, i] = j
                end
            end
            @inbounds cnts[i] = k
            return nothing
        end

        dv = device_view(tns_gpu, gid, gid)
        threads = 128
        blocks = cld(N, threads)
        @cuda threads=threads blocks=blocks collect_kernel_2d!(lists, cnts, dv, Int32(N), Int32(MAX_NBRS))
        CUDA.synchronize()

        host_lists = Array(lists)
        host_cnts = Array(cnts)
        ε = r * 1f-4

        for i in 1:N
            cpu_set = Set(Int32.(get_neighborlist(tns_cpu, cid, cid, i)))
            gpu_set = Set(host_lists[1:host_cnts[i], i])

            function _in_band(j)
                dx = cpu_coords[1,j] - cpu_coords[1,i]
                dy = cpu_coords[2,j] - cpu_coords[2,i]
                d2 = dx*dx + dy*dy
                return abs(sqrt(d2) - r) < ε
            end
            cpu_clean = filter(j -> !_in_band(Int(j)), collect(cpu_set))
            gpu_clean = filter(j -> !_in_band(Int(j)), collect(gpu_set))
            @test sort(cpu_clean) == sort(gpu_clean)
        end
    end
end
