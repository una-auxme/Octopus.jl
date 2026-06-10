module TreeNSearchCUDAChainRulesCoreExt

# Weak extension: GPU rrule for build_edges_diff. Loads when both CUDA.jl
# and ChainRulesCore.jl are present. The CPU rrule lives in
# TreeNSearchChainRulesCoreExt; this file adds the matching GPU dispatch
# (atomic-add kernel mirroring GraphNetSim.jl/src/graph.jl:513).

using TreeNSearch
import TreeNSearch: TNS, build_edges
using CUDA
import ChainRulesCore
using ChainRulesCore: NoTangent, AbstractZero

# ---------------- build_edges_diff: GPU primal -----------------------------
# Snapshots the four output arrays so the pullback closes over stable values
# (the EdgeBuffer is reused across calls).

function TreeNSearch.build_edges_diff(coords::CUDA.CuMatrix{T},
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

# Local copy of the tangent helper (each ext is its own module; we duplicate
# this rather than reach across).
@inline _tangent_field(::Nothing, ::Symbol)      = nothing
@inline _tangent_field(::AbstractZero, ::Symbol) = nothing
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

# ---------------- pullback kernel -----------------------------------------
# One thread per edge. Each thread accumulates ∂L/∂coords[:,i] and
# ∂L/∂coords[:,j] via `CUDA.atomic_add!` because multiple edges can target
# the same particle. Mirrors GraphNetSim.jl/src/graph.jl:593.

function _build_edges_diff_pullback_kernel!(grad_coords::CuDeviceMatrix{T},
                                            Δdisp::CuDeviceMatrix{T},
                                            Δdist::CuDeviceMatrix{T},
                                            senders::CuDeviceVector{Int32},
                                            receivers::CuDeviceVector{Int32},
                                            rel_disp::CuDeviceMatrix{T},
                                            rel_dist::CuDeviceMatrix{T},
                                            radius::T,
                                            n_edges::Int32,
                                            use_disp::Bool,
                                            use_dist::Bool,
                                            D::Int32) where {T}
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    idx > n_edges && return nothing

    @inbounds i = receivers[idx]
    @inbounds j = senders[idx]
    rows = size(grad_coords, 1)  # == D

    if use_disp
        @inbounds for d in Int32(1):D
            val = Δdisp[d, idx] / radius
            CUDA.atomic_add!(pointer(grad_coords, (j - Int32(1)) * Int32(rows) + d), val)
            CUDA.atomic_add!(pointer(grad_coords, (i - Int32(1)) * Int32(rows) + d), -val)
        end
    end

    if use_dist
        @inbounds d_norm = rel_dist[1, idx]
        if d_norm > T(1e-8)
            @inbounds inv_dnr = Δdist[1, idx] / (d_norm * radius)
            @inbounds for d in Int32(1):D
                g = rel_disp[d, idx] * inv_dnr
                CUDA.atomic_add!(pointer(grad_coords, (j - Int32(1)) * Int32(rows) + d), -g)
                CUDA.atomic_add!(pointer(grad_coords, (i - Int32(1)) * Int32(rows) + d), g)
            end
        end
    end
    return nothing
end

# Convert an arbitrary upstream tangent to a CuArray matching `template`.
# Mirrors GraphNetSim's `ensure_cuda` defensiveness against Fill / Tangent
# wrappers Zygote sometimes hands us.
function _ensure_cuarray(amt, template::CUDA.CuMatrix{T}) where {T}
    if amt isa CUDA.CuArray
        return amt
    elseif amt isa AbstractArray
        return CUDA.CuArray{T}(amt)
    elseif amt isa AbstractZero
        return CUDA.zeros(T, size(template))
    else
        return convert(CUDA.CuArray{T}, amt)
    end
end

# ---------------- build_edges_diff: GPU rrule ------------------------------

function ChainRulesCore.rrule(::typeof(TreeNSearch.build_edges_diff),
                              coords::CUDA.CuMatrix{T},
                              tns::TNS,
                              id::Integer,
                              radius::Real) where {T<:AbstractFloat}
    e = TreeNSearch.build_edges_diff(coords, tns, id, radius)
    senders          = e.senders
    receivers        = e.receivers
    rel_displacement = e.rel_displacement
    rel_dist_norm    = e.rel_dist_norm
    r = T(radius)
    D = Int32(size(coords, 1))

    function build_edges_diff_pullback(Δ)
        Δdisp_raw = _tangent_field(Δ, :rel_displacement)
        Δdist_raw = _tangent_field(Δ, :rel_dist_norm)

        grad_coords = CUDA.zeros(T, size(coords))

        if Δdisp_raw === nothing && Δdist_raw === nothing
            return (NoTangent(), grad_coords, NoTangent(), NoTangent(), NoTangent())
        end

        # We always pass an array to the kernel, even when one of the two
        # upstream tangents is zero, so the kernel signature stays uniform.
        Δdisp = Δdisp_raw === nothing ?
                CUDA.zeros(T, size(rel_displacement)) :
                _ensure_cuarray(Δdisp_raw, rel_displacement)
        Δdist = Δdist_raw === nothing ?
                CUDA.zeros(T, size(rel_dist_norm)) :
                _ensure_cuarray(Δdist_raw, rel_dist_norm)

        n_edges = Int32(length(senders))
        if n_edges > 0
            threads = 256
            blocks  = cld(Int(n_edges), threads)
            @cuda threads=threads blocks=blocks _build_edges_diff_pullback_kernel!(
                grad_coords, Δdisp, Δdist,
                senders, receivers, rel_displacement, rel_dist_norm,
                r, n_edges,
                Δdisp_raw !== nothing, Δdist_raw !== nothing,
                D,
            )
        end

        return (NoTangent(), grad_coords, NoTangent(), NoTangent(), NoTangent())
    end

    return e, build_edges_diff_pullback
end

end # module
