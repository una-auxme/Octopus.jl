#
# Copyright (c) 2026 Josef Jouaux
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module OctopusChainRulesCoreExt

# Weak-extension: Zygote / ChainRules compatibility for Octopus.
#
# The package's core API is buffer-mutating, which Zygote rejects. We patch
# that by:
#   1. Declaring the setup ops (`run!`, `add_point_set!`, …) as
#      `@non_differentiable` so users can wrap them implicitly.
#   2. Exposing `build_edges_diff(coords, tns, id, radius)` whose rrule
#      computes ∂L/∂coords directly from the upstream Δrel_displacement /
#      Δrel_dist_norm tangents — mirroring the pattern in
#      GraphNetSim.jl/src/graph.jl:432.
#
# This file handles the CPU (Array) path. The matching GPU path lives in
# OctopusCUDAChainRulesCoreExt (loaded when CUDA + ChainRulesCore are
# both present), which uses `CUDA.atomic_add!` for the per-edge accumulation.

using Octopus
import Octopus: TNS, build_edges
import ChainRulesCore
using ChainRulesCore: NoTangent, ZeroTangent, AbstractZero, Tangent, @non_differentiable

# ---------------- non-differentiable setup ops ----------------------------
# These mutate `tns` and have no meaningful gradient. Declaring them keeps
# users from having to wrap each one in `Zygote.@ignore`.

@non_differentiable Octopus.run!(::Any)
@non_differentiable Octopus.set_search_radius!(::Any, ::Any)
@non_differentiable Octopus.set_refit_mode!(::Any, ::Any)
@non_differentiable Octopus.add_point_set!(::Any, ::Any)
@non_differentiable Octopus.resize_point_set!(::Any, ::Any, ::Any)
@non_differentiable Octopus.update_point_set!(::Any, ::Any, ::Any)
@non_differentiable Octopus.set_active_search!(::Any, ::Any, ::Any)
@non_differentiable Octopus.prepare_zsort(::Any)
# NB: `apply_zsort` is deliberately absent from this list — it is a gather and
# has a real gradient. See the rrule below.
@non_differentiable Octopus.get_neighborlist(::Any, ::Any, ::Any, ::Any)
@non_differentiable Octopus.materialize_all_neighbors!(::Any)
@non_differentiable Octopus.device_view(::Any, ::Any, ::Any)

# ---------------- apply_zsort ---------------------------------------------
#
# `apply_zsort` is a gather along the point axis:
#
#     out[i]    = user_array[perm[i]]        (1-D)
#     out[:, i] = user_array[:, perm[i]]     (2-D)
#
# so its adjoint is the matching scatter. `perm` is a permutation of 1:N, so
# every source slot lands in exactly one destination slot — no accumulation,
# and the scatter is itself just an indexed assignment.
#
# Without this rule the CPU path dies inside Zygote with "Mutating arrays is
# not supported -- called setindex!", because `apply_zsort_cpu!` fills its
# output with `setindex!` even though the function is externally pure. The
# CUDA path happened to work, being written as plain fancy indexing, so the
# two backends disagreed on whether z-sorted features were differentiable at
# all. One rule covers both: the scatter below is vectorised, so it stays on
# the device for `CuArray` inputs rather than falling into scalar indexing.

function ChainRulesCore.rrule(::typeof(Octopus.apply_zsort),
                              tns::TNS, set_id::Integer,
                              user_array::AbstractArray)
    y = Octopus.apply_zsort(tns, set_id, user_array)
    perm = tns.permutation[set_id]

    function apply_zsort_pullback(Δ)
        Δy = ChainRulesCore.unthunk(Δ)
        Δy isa AbstractZero &&
            return (NoTangent(), NoTangent(), NoTangent(), ZeroTangent())
        grad = _scatter_zsort(user_array, perm, Δy)
        return (NoTangent(), NoTangent(), NoTangent(), grad)
    end

    return y, apply_zsort_pullback
end

# Function barrier: `tns.permutation` is `Vector{Any}`, so `perm` reaches the
# pullback untyped. Splitting the scatter out lets it specialise on the
# concrete permutation and tangent types.
function _scatter_zsort(user_array::AbstractArray, perm::AbstractVector,
                        Δy::AbstractArray)
    grad = zero(user_array)
    if ndims(user_array) == 1
        grad[perm] = Δy
    else
        grad[:, perm] = Δy
    end
    return grad
end

# ---------------- build_edges_diff: CPU primal -----------------------------

# We snapshot the four output arrays so the pullback closes over stable
# values: the underlying EdgeBuffer is reused across calls and would
# otherwise be overwritten by a subsequent `build_edges(!)` / `build_edges_diff`
# call before the pullback runs.

function Octopus.build_edges_diff(coords::AbstractMatrix{T},
                                      tns::TNS,
                                      id::Integer,
                                      radius::Real) where {T<:AbstractFloat}
    e = build_edges(tns, id, id)
    return (
        senders          = copy(e.senders),
        receivers        = copy(e.receivers),
        rel_displacement = copy(e.rel_displacement),
        rel_dist_norm    = copy(e.rel_dist_norm),
    )
end

# Helper: extract a tangent component robustly from whatever Zygote passes.
# Δ might be a Tangent{<:NamedTuple}, a plain NamedTuple, ZeroTangent, or
# nothing. We return either an AbstractArray (the tangent) or `nothing`
# (treat as zero — caller short-circuits).
@inline _tangent_field(Δ::Nothing, ::Symbol)        = nothing
@inline _tangent_field(Δ::AbstractZero, ::Symbol)   = nothing
function _tangent_field(Δ, name::Symbol)
    v = try
        getproperty(Δ, name)
    catch
        return nothing
    end
    v === nothing && return nothing
    v isa AbstractZero && return nothing
    return ChainRulesCore.unthunk(v)
end

# ---------------- build_edges_diff: CPU rrule ------------------------------
# Self-pair only (qid == tid == id). Cross-pair could be added by accepting
# (coords_q, coords_t) and writing two grads; that's not part of v0.1.

function ChainRulesCore.rrule(::typeof(Octopus.build_edges_diff),
                              coords::AbstractMatrix{T},
                              tns::TNS,
                              id::Integer,
                              radius::Real) where {T<:AbstractFloat}
    e = Octopus.build_edges_diff(coords, tns, id, radius)
    senders          = e.senders
    receivers        = e.receivers
    rel_displacement = e.rel_displacement
    rel_dist_norm    = e.rel_dist_norm
    r = T(radius)
    D = size(coords, 1)

    function build_edges_diff_pullback(Δ)
        Δdisp = _tangent_field(Δ, :rel_displacement)
        Δdist = _tangent_field(Δ, :rel_dist_norm)

        grad_coords = zeros(T, size(coords))

        # Short-circuit if both upstream tangents are zero.
        if Δdisp === nothing && Δdist === nothing
            return (NoTangent(), grad_coords, NoTangent(), NoTangent(), NoTangent())
        end

        @inbounds for idx in eachindex(senders)
            i = Int(receivers[idx])
            j = Int(senders[idx])

            # ∂L/∂coords via rel_displacement[d, idx] = (coords[d,i] - coords[d,j]) / r
            if Δdisp !== nothing
                for d in 1:D
                    val = T(Δdisp[d, idx]) / r
                    grad_coords[d, i] += val
                    grad_coords[d, j] -= val
                end
            end

            # ∂L/∂coords via rel_dist_norm[1, idx] = ‖coords[:,i] - coords[:,j]‖ / r
            if Δdist !== nothing
                d_norm = rel_dist_norm[1, idx]
                if d_norm > T(1e-8)
                    inv_dnr = T(Δdist[1, idx]) / (d_norm * r)
                    for d in 1:D
                        g = rel_displacement[d, idx] * inv_dnr
                        grad_coords[d, i] += g
                        grad_coords[d, j] -= g
                    end
                end
            end
        end

        return (NoTangent(), grad_coords, NoTangent(), NoTangent(), NoTangent())
    end

    return e, build_edges_diff_pullback
end

end # module
