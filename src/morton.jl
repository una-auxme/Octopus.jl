#
# Copyright (c) 2026 Josef Jouaux
# Copyright (c) 2022-present, CompactNSearch contributors
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# Morton (Z-order) encoding for cell coordinates.
# 3D: 21 bits per axis -> 63-bit code packed in UInt64.
# 2D: 31 bits per axis -> 62-bit code packed in UInt64 (top bit zero so
#     `lcp_tiebreak`'s 63-bit assumption is preserved).
# Magic-bits interleave in both dims.

@inline function _split_by_3(x::UInt64)::UInt64
    # Take the low 21 bits of x and spread them out so each bit occupies
    # every third position in a 64-bit word. Classic magic-bits:
    x &= 0x00000000001fffff
    x = (x | (x << 32)) & 0x001f00000000ffff
    x = (x | (x << 16)) & 0x001f0000ff0000ff
    x = (x | (x << 8))  & 0x100f00f00f00f00f
    x = (x | (x << 4))  & 0x10c30c30c30c30c3
    x = (x | (x << 2))  & 0x1249249249249249
    return x
end

@inline function _compact_by_3(x::UInt64)::UInt64
    x &= 0x1249249249249249
    x = (x | (x >> 2))  & 0x10c30c30c30c30c3
    x = (x | (x >> 4))  & 0x100f00f00f00f00f
    x = (x | (x >> 8))  & 0x001f0000ff0000ff
    x = (x | (x >> 16)) & 0x001f00000000ffff
    x = (x | (x >> 32)) & 0x00000000001fffff
    return x
end

@inline function morton_encode3(ix::Integer, iy::Integer, iz::Integer)::UInt64
    x = _split_by_3(UInt64(ix))
    y = _split_by_3(UInt64(iy))
    z = _split_by_3(UInt64(iz))
    return (z << 2) | (y << 1) | x
end

@inline function morton_decode3(code::UInt64)::NTuple{3,UInt32}
    x = _compact_by_3(code)
    y = _compact_by_3(code >> 1)
    z = _compact_by_3(code >> 2)
    return (UInt32(x), UInt32(y), UInt32(z))
end

# 2D magic-bits: spread 31 bits so each occupies every second position.
@inline function _split_by_2(x::UInt64)::UInt64
    x &= 0x000000007fffffff                      # keep low 31 bits
    x = (x | (x << 16)) & 0x0000ffff0000ffff
    x = (x | (x << 8))  & 0x00ff00ff00ff00ff
    x = (x | (x << 4))  & 0x0f0f0f0f0f0f0f0f
    x = (x | (x << 2))  & 0x3333333333333333
    x = (x | (x << 1))  & 0x5555555555555555
    return x
end

@inline function _compact_by_2(x::UInt64)::UInt64
    x &= 0x5555555555555555
    x = (x | (x >> 1))  & 0x3333333333333333
    x = (x | (x >> 2))  & 0x0f0f0f0f0f0f0f0f
    x = (x | (x >> 4))  & 0x00ff00ff00ff00ff
    x = (x | (x >> 8))  & 0x0000ffff0000ffff
    x = (x | (x >> 16)) & 0x000000007fffffff
    return x
end

@inline function morton_encode2(ix::Integer, iy::Integer)::UInt64
    x = _split_by_2(UInt64(ix))
    y = _split_by_2(UInt64(iy))
    return (y << 1) | x
end

@inline function morton_decode2(code::UInt64)::NTuple{2,UInt32}
    x = _compact_by_2(code)
    y = _compact_by_2(code >> 1)
    return (UInt32(x), UInt32(y))
end

# Bin a 3D point to integer cell coordinates, given origin and cell_size.
@inline function bin_point(px::T, py::T, pz::T, ox::T, oy::T, oz::T, inv_cs::T) where {T<:AbstractFloat}
    ix = floor(Int32, (px - ox) * inv_cs)
    iy = floor(Int32, (py - oy) * inv_cs)
    iz = floor(Int32, (pz - oz) * inv_cs)
    return (ix, iy, iz)
end

# Bin a 2D point to integer cell coordinates.
@inline function bin_point(px::T, py::T, ox::T, oy::T, inv_cs::T) where {T<:AbstractFloat}
    ix = floor(Int32, (px - ox) * inv_cs)
    iy = floor(Int32, (py - oy) * inv_cs)
    return (ix, iy)
end
