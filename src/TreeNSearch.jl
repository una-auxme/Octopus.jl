module TreeNSearch

using StaticArrays
using Polyester: @batch
using Atomix
import Adapt
using GPUArraysCore: AbstractGPUArray

include("morton.jl")
include("types.jl")
include("memory.jl")
include("util.jl")
include("sort.jl")

include("cpu/bin_cpu.jl")
include("cpu/build_cpu.jl")
include("cpu/refit_cpu.jl")
include("cpu/query_cpu.jl")
include("cpu/iterator_cpu.jl")
include("cpu/edges_cpu.jl")
include("cpu/zsort_cpu.jl")

include("api.jl")
include("macros.jl")

# Public API
export TNS
export set_search_radius!, set_refit_mode!
export add_point_set!, resize_point_set!, update_point_set!
export set_active_search!, set_symmetric_search!
export run!
export for_each_neighbor, for_each_neighbor_device
export get_neighborlist, materialize_all_neighbors!
export build_edges, build_edges!, EdgeBuffer
export prepare_zsort!, apply_zsort!
export device_view
export @for_each_neighbor_device_inline
export @for_each_neighbor_device_inline_2d

# Hooks filled in by TreeNSearchCUDAExt when CUDA.jl is loaded.
function _run_cuda! end
function _device_view end
function _for_each_neighbor_device end
function _build_edges_cuda! end
function _apply_zsort_cuda end
function _cuda_error(msg::AbstractString)
    error("TreeNSearch: $msg")
end

end # module
