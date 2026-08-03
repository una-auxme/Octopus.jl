#
# Copyright (c) 2026 Josef Jouaux
# Copyright (c) 2022-present, CompactNSearch contributors
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# CPU sort dispatch. GPU path lives in the CUDA extension.
#
# We want: perm[1..N] such that keys[perm[k]] is nondecreasing.
# Stdlib `sortperm!` allocates a temp and does a stable quicksort; for UInt64
# keys this is fast enough for the build phase (and not on the hot path of
# steady-state SPH runs). We can promote to SortingAlgorithms.RadixSort later
# if benchmarks demand it.

@inline function sort_by_key_cpu!(perm::Vector{Int32}, keys::Vector{UInt64})
    n = length(keys)
    @assert length(perm) == n
    @inbounds for i in 1:n
        perm[i] = Int32(i)
    end
    # sortperm! on a Vector{Int32} with a Vector{UInt64} key via lt closure.
    sort!(perm, by = i -> @inbounds(keys[i]), alg = QuickSort)
    return perm
end
