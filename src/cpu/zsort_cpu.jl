#
# Copyright (c) 2026 Josef Jouaux
# Copyright (c) 2022-present, CompactNSearch contributors
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# Apply the z-order permutation to user arrays.
# prepare_zsort ensures the permutation for every registered point set is available
# (requires refit_mode, or at least that the current build kept it).

function apply_zsort_cpu!(out::AbstractArray, perm::AbstractVector{Int32}, original::AbstractArray)
    # 1-D: permute the only axis. 2-D: permute columns. Higher rank is rejected
    # so the assertion (which checks the last axis) never disagrees with the
    # loop (which iterates the column axis).
    ndims(original) <= 2 || throw(ArgumentError(
        "apply_zsort supports 1-D or 2-D arrays only; got $(ndims(original))-D"))
    @assert length(perm) == size(original, ndims(original))
    if ndims(original) == 1
        @inbounds for i in eachindex(perm)
            out[i] = original[perm[i]]
        end
    else
        n = size(original, 2)
        @inbounds for i in 1:n
            src = perm[i]
            for r in axes(original, 1)
                out[r, i] = original[r, src]
            end
        end
    end
    return out
end
