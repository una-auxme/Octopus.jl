# Phase 1: bin every particle to a grid cell and compute its Morton code.
# Phase 3 (RLE of non-empty cells) also lives here since it reads the same
# permuted Morton array.
#
# Dispatch is by coord row count (3D vs 2D) via SVector{NDIMS,T} origin type.

# 3D origin
@inline function _point_origin_3d(coords::AbstractMatrix{T}) where {T}
    @inbounds begin
        ox = coords[1, 1]; oy = coords[2, 1]; oz = coords[3, 1]
        n = size(coords, 2)
        for i in 2:n
            ox = min(ox, coords[1, i])
            oy = min(oy, coords[2, i])
            oz = min(oz, coords[3, i])
        end
    end
    return SVector{3,T}(ox - T(1e-6), oy - T(1e-6), oz - T(1e-6))
end

# 2D origin
@inline function _point_origin_2d(coords::AbstractMatrix{T}) where {T}
    @inbounds begin
        ox = coords[1, 1]; oy = coords[2, 1]
        n = size(coords, 2)
        for i in 2:n
            ox = min(ox, coords[1, i])
            oy = min(oy, coords[2, i])
        end
    end
    return SVector{2,T}(ox - T(1e-6), oy - T(1e-6))
end

# Dispatch by row count of the coords matrix. The compiler doesn't know
# size(coords, 1) at compile time in general, so we keep the runtime branch
# at the orchestration layer (one call per build, not per point).
@inline function _point_origin(coords::AbstractMatrix{T}) where {T}
    nrows = size(coords, 1)
    if nrows == 3
        return _point_origin_3d(coords)
    elseif nrows == 2
        return _point_origin_2d(coords)
    else
        throw(DimensionMismatch("_point_origin: coords must have 2 or 3 rows, got $nrows"))
    end
end

function bin_cpu!(morton_codes::Vector{UInt64},
                  coords::AbstractMatrix{T},
                  origin::SVector{3,T},
                  cell_size::T) where {T<:AbstractFloat}
    n = size(coords, 2)
    ensure_capacity!(morton_codes, n)
    truncate!(morton_codes, n)
    ox, oy, oz = origin[1], origin[2], origin[3]
    inv_cs = one(T) / cell_size
    @batch minbatch = 4096 for i in 1:n
        @inbounds begin
            px = coords[1, i]; py = coords[2, i]; pz = coords[3, i]
            ix, iy, iz = bin_point(px, py, pz, ox, oy, oz, inv_cs)
            # Clamp to 21-bit range; SPH scenes never approach this.
            ix = clamp(ix, Int32(0), Int32(0x1fffff))
            iy = clamp(iy, Int32(0), Int32(0x1fffff))
            iz = clamp(iz, Int32(0), Int32(0x1fffff))
            morton_codes[i] = morton_encode3(ix, iy, iz)
        end
    end
    return morton_codes
end

function bin_cpu!(morton_codes::Vector{UInt64},
                  coords::AbstractMatrix{T},
                  origin::SVector{2,T},
                  cell_size::T) where {T<:AbstractFloat}
    n = size(coords, 2)
    ensure_capacity!(morton_codes, n)
    truncate!(morton_codes, n)
    ox, oy = origin[1], origin[2]
    inv_cs = one(T) / cell_size
    @batch minbatch = 4096 for i in 1:n
        @inbounds begin
            px = coords[1, i]; py = coords[2, i]
            ix, iy = bin_point(px, py, ox, oy, inv_cs)
            # Clamp to 31-bit range.
            ix = clamp(ix, Int32(0), Int32(0x7fffffff))
            iy = clamp(iy, Int32(0), Int32(0x7fffffff))
            morton_codes[i] = morton_encode2(ix, iy)
        end
    end
    return morton_codes
end

# After `sort_by_key_cpu!(perm, morton_codes)`, consecutive entries in
# `morton_codes[perm]` with equal code form a non-empty cell. We emit:
#   cell_morton[k]  = morton code of cell k
#   cell_first[k]   = first index into perm for cell k
#   cell_last[k]    = last index into perm for cell k  (inclusive)
function rle_cells_cpu!(cell_morton::Vector{UInt64},
                        cell_first::Vector{Int32},
                        cell_last::Vector{Int32},
                        morton_codes::Vector{UInt64},
                        perm::Vector{Int32})
    n = length(perm)
    empty!(cell_morton); empty!(cell_first); empty!(cell_last)
    n == 0 && return (cell_morton, cell_first, cell_last)

    @inbounds begin
        push!(cell_morton, morton_codes[perm[1]])
        push!(cell_first, Int32(1))
        prev = morton_codes[perm[1]]
        for i in 2:n
            m = morton_codes[perm[i]]
            if m != prev
                push!(cell_last, Int32(i - 1))
                push!(cell_morton, m)
                push!(cell_first, Int32(i))
                prev = m
            end
        end
        push!(cell_last, Int32(n))
    end
    return (cell_morton, cell_first, cell_last)
end
