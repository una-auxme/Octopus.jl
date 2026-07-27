#
# Copyright (c) 2026 Josef Jouaux, Chair of Mechatronics, University of Augsburg
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Test
using Octopus
using Random

@testset "Octopus" begin
    @testset "morton"        include("morton.jl")
    @testset "build"         include("build.jl")
    @testset "query_cpu"     include("query_cpu.jl")
    @testset "edges_cpu"     include("edges_cpu.jl")
    @testset "api"           include("api.jl")
    @testset "memory"        include("memory.jl")
    @testset "build_2d"      include("build_2d.jl")
    @testset "query_cpu_2d"  include("query_cpu_2d.jl")
    @testset "edges_cpu_2d"  include("edges_cpu_2d.jl")
    @testset "api_2d"        include("api_2d.jl")
    @testset "zsort"         include("zsort.jl")
    @testset "contracts"     include("contracts.jl")
    @testset "refit"         include("refit.jl")
    @testset "edge_cases"    include("edge_cases.jl")
    @testset "float64"       include("float64.jl")
    @testset "backend_errors_cpu" include("backend_errors_cpu.jl")
    if get(ENV, "JULIA_OCTOPUS_TEST_CUDA", "0") == "1"
        @testset "query_cuda"    include("query_cuda.jl")
        @testset "edges_cuda"    include("edges_cuda.jl")
        @testset "query_cuda_2d" include("query_cuda_2d.jl")
        @testset "edges_cuda_2d" include("edges_cuda_2d.jl")
        @testset "zsort_cuda"    include("zsort_cuda.jl")
    else
        @info "skipping CUDA tests; set JULIA_OCTOPUS_TEST_CUDA=1 to enable"
    end
    if get(ENV, "JULIA_OCTOPUS_TEST_CHAINRULES", "0") == "1"
        @testset "chainrules"    include("chainrules.jl")
    else
        @info "skipping ChainRulesCore tests; set JULIA_OCTOPUS_TEST_CHAINRULES=1 to enable"
    end
end
