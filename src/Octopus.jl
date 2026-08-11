#
# Copyright (c) 2026 Josef Jouaux
# Copyright (c) 2022-present, CompactNSearch contributors
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module Octopus

using StaticArrays
using Polyester: @batch
import Adapt
using GPUArraysCore: AbstractGPUArray

include("morton.jl")
include("types.jl")
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
export set_active_search!
export run!
export for_each_neighbor, for_each_neighbor_device
export get_neighborlist, materialize_all_neighbors!
export build_edges, build_edges!, EdgeBuffer
export build_edges_diff
export prepare_zsort, apply_zsort
export prepare_zsort!, apply_zsort!   # deprecated bang spellings, see api.jl
export device_view
export @for_each_neighbor_device_inline
export @for_each_neighbor_device_inline_2d

# Hooks filled in by OctopusCUDAExt when CUDA.jl is loaded.
function _run_cuda! end
function _device_view end
function _for_each_neighbor_device end
function _build_edges_cuda! end
function _apply_zsort_cuda end
function _cuda_error(msg::AbstractString)
    error("Octopus: $msg")
end

"""
    build_edges_diff(coords, tns, id, radius) -> NamedTuple

Differentiable wrapper around `build_edges(tns, id, id)`. Returns a NamedTuple
with `(senders, receivers, rel_displacement, rel_dist_norm)` whose arrays are
fresh copies (so backprop closures are not affected by subsequent buffer
overwrites). Gradients flow back to `coords` (the tree topology and the
search radius are non-differentiable).

The user is responsible for keeping `coords` in sync with `tns` — typically
by calling `update_point_set!(tns, id, coords); run!(tns)` under
`Zygote.@ignore` before this function.

Loaded via the `OctopusChainRulesCoreExt` and
`OctopusCUDAChainRulesCoreExt` weak extensions; requires
`using ChainRulesCore` (and `using CUDA` for the GPU path).
"""
function build_edges_diff end

end # module
