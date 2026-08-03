#
# Copyright (c) 2026 Josef Jouaux
# Copyright (c) 2022-present, CompactNSearch contributors
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Test
using Octopus: morton_encode3, morton_decode3, morton_encode2, morton_decode2, bin_point

@testset "encode/decode round-trip" begin
    for (x, y, z) in [(0, 0, 0), (1, 2, 3), (7, 11, 19), (0x1fffff, 0x1fffff, 0x1fffff),
                      (1_000_000, 500_000, 750_000)]
        c = morton_encode3(x, y, z)
        dx, dy, dz = morton_decode3(c)
        @test Int(dx) == x
        @test Int(dy) == y
        @test Int(dz) == z
    end
end

@testset "axis monotonicity along x" begin
    y = 42; z = 7
    prev = morton_encode3(0, y, z)
    for x in 1:64
        cur = morton_encode3(x, y, z)
        @test cur > prev
        prev = cur
    end
end

@testset "bin_point rounding" begin
    # origin = 0, cs = 1: floor(p) for each axis
    ix, iy, iz = bin_point(0.5f0, 1.5f0, 2.9f0, 0f0, 0f0, 0f0, 1f0)
    @test (ix, iy, iz) == (Int32(0), Int32(1), Int32(2))
    # offset origin, cs = 0.1 → inv_cs = 10
    ix, iy, iz = bin_point(3.2f0, 3.2f0, 3.2f0, 3.0f0, 3.0f0, 3.0f0, 1f0/0.1f0)
    @test (ix, iy, iz) == (Int32(2), Int32(2), Int32(2))
end

# ---- 2D Morton ----

@testset "2D encode/decode round-trip" begin
    for (x, y) in [(0, 0), (1, 2), (7, 11), (0x7fffffff, 0x7fffffff),
                    (1_000_000, 500_000), (123_456_789, 987_654_321)]
        c = morton_encode2(x, y)
        dx, dy = morton_decode2(c)
        @test Int(dx) == x
        @test Int(dy) == y
    end
end

@testset "2D top-bit invariant (62-bit code)" begin
    # With 31 bits/axis, the top 2 bits of the 64-bit code must be zero.
    for (x, y) in [(0x7fffffff, 0x7fffffff), (1_000_000, 999_999), (0, 0x7fffffff)]
        c = morton_encode2(x, y)
        @test (c >> 62) == UInt64(0)
    end
end

@testset "2D axis monotonicity along x" begin
    y = 42
    prev = morton_encode2(0, y)
    for x in 1:64
        cur = morton_encode2(x, y)
        @test cur > prev
        prev = cur
    end
end

@testset "2D bin_point rounding" begin
    ix, iy = bin_point(0.5f0, 1.5f0, 0f0, 0f0, 1f0)
    @test (ix, iy) == (Int32(0), Int32(1))
    ix, iy = bin_point(3.2f0, 3.2f0, 3.0f0, 3.0f0, 1f0/0.1f0)
    @test (ix, iy) == (Int32(2), Int32(2))
end
