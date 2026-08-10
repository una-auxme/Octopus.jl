#
# Copyright (c) 2026 Josef Jouaux
# Copyright (c) 2022-present, CompactNSearch contributors
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# Inlined-traversal macros for the device-side neighbor iterator.
#
# `for_each_neighbor_device(dv, i) do j ... end` is convenient but Julia boxes
# any local scalars the closure body mutates (e.g. a per-edge write cursor),
# which on GPU forces a global-memory round-trip per emitted edge. The macros
# below expand the same traversal directly into the user's kernel so cursors
# can stay in registers and constants like `1/radius` precompute. End-to-end
# measured speedup on a write-pass kernel (per-edge cursor in a thread-local):
#
#   2D N=1k  : 2.05x        (closure 1.62 ms -> macro 0.79 ms)
#   2D N=5k  : 1.75x        (closure 9.87 ms -> macro 5.63 ms)
#   2D N=20k : 1.43x        (closure 63.4 ms -> macro 44.4 ms)
#
# Edge sets are byte-for-byte identical between the two paths.
#
# Two macros are exposed: `@for_each_neighbor_device_inline` (3D, octree)
# and `@for_each_neighbor_device_inline_2d` (2D, quadtree). Macros expand at
# parse time, so a `Val{NDIMS}` argument is awkward — keep them separate.
#
# Usage (3D):
#
#   function my_kernel!(... , dv, coords, ...)
#       i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#       i > N && return nothing
#       i32 = Int32(i)
#       cursor = ...
#       Octopus.@for_each_neighbor_device_inline dv i j begin
#           if j != i32
#               # use j (Int32), do per-edge work; cursor is a register here
#               cursor += Int32(1)
#           end
#       end
#       cursors[i] = cursor
#       return nothing
#   end
#
# The body sees `j` as the neighbor index. The user is responsible for the
# self-pair filter (mirroring the closure form). The macro does NOT emit `j`
# for pairs outside the radius; only pairs with `‖coords[i] - coords[j]‖² ≤
# radius²` reach the body.

"""
    @for_each_neighbor_device_inline dv i j body

Inline expansion of the device-side neighbor traversal for use inside `@cuda`
kernels. Faster than `for_each_neighbor_device(dv, i) do j ... end` when the
body mutates local scalars (closure boxing forces a global-memory round-trip
per emitted edge). Emits `body` for every neighbor `j` whose distance to
point `i` is `≤ dv.radius`. The user is responsible for the self-pair filter
when `qid == tid`.

Requires the CUDA extension to be loaded (so `dv` is a `DeviceView`).
"""
macro for_each_neighbor_device_inline(dv_expr, i_expr, j_sym, body_expr)
    j_sym isa Symbol || error("@for_each_neighbor_device_inline: third argument must be a bare symbol naming the neighbor index variable")
    j = esc(j_sym)
    body = esc(body_expr)
    return quote
        let
            _dv  = $(esc(dv_expr))
            _i   = $(esc(i_expr))
            _r_sq = _dv.radius_sq
            _px = @inbounds _dv.coords_q[1, _i]
            _py = @inbounds _dv.coords_q[2, _i]
            _pz = @inbounds _dv.coords_q[3, _i]

            _stack = StaticArrays.MVector{STACK_DEPTH_3D, Int32}(undef)
            _sp = 1
            @inbounds _stack[_sp] = Int32(1)

            while _sp > 0
                _nid = @inbounds _stack[_sp]
                _sp -= 1

                if @inbounds(_dv.node_children[1, _nid]) == Int32(-1)
                    _f0 = @inbounds _dv.node_first[_nid]
                    _l0 = @inbounds _dv.node_last[_nid]
                    @inbounds for _k in _f0:_l0
                        $j = _dv.perm[_k]
                        _qx = _dv.coords_t[1, $j]
                        _qy = _dv.coords_t[2, $j]
                        _qz = _dv.coords_t[3, $j]
                        _dx = _qx - _px; _dy = _qy - _py; _dz = _qz - _pz
                        _d2 = _dx*_dx + _dy*_dy + _dz*_dz
                        if _d2 <= _r_sq
                            $body
                        end
                    end
                else
                    @inbounds for _c in 1:8
                        _cid = _dv.node_children[_c, _nid]
                        if _cid != Int32(-1)
                            _lo1 = _dv.bounds_min[1, _cid]; _lo2 = _dv.bounds_min[2, _cid]; _lo3 = _dv.bounds_min[3, _cid]
                            _hi1 = _dv.bounds_max[1, _cid]; _hi2 = _dv.bounds_max[2, _cid]; _hi3 = _dv.bounds_max[3, _cid]
                            _z = zero(_px)
                            _ddx = max(_lo1 - _px, _z); _ddx = max(_ddx, _px - _hi1)
                            _ddy = max(_lo2 - _py, _z); _ddy = max(_ddy, _py - _hi2)
                            _ddz = max(_lo3 - _pz, _z); _ddz = max(_ddz, _pz - _hi3)
                            if _ddx*_ddx + _ddy*_ddy + _ddz*_ddz <= _r_sq
                                _sp += 1
                                @inbounds _stack[_sp] = _cid
                            end
                        end
                    end
                end
            end
        end
    end
end

"""
    @for_each_neighbor_device_inline_2d dv i j body

2D quadtree variant of `@for_each_neighbor_device_inline`. Same shape and
purpose; expects `dv::DeviceView{T,2}` and a 4-way child fan-out. Use this
inside `@cuda` kernels for 2D `TNS{T,2}` workloads when closure boxing would
spill a per-edge cursor to global memory.
"""
macro for_each_neighbor_device_inline_2d(dv_expr, i_expr, j_sym, body_expr)
    j_sym isa Symbol || error("@for_each_neighbor_device_inline_2d: third argument must be a bare symbol naming the neighbor index variable")
    j = esc(j_sym)
    body = esc(body_expr)
    return quote
        let
            _dv  = $(esc(dv_expr))
            _i   = $(esc(i_expr))
            _r_sq = _dv.radius_sq
            _px = @inbounds _dv.coords_q[1, _i]
            _py = @inbounds _dv.coords_q[2, _i]

            _stack = StaticArrays.MVector{STACK_DEPTH_2D, Int32}(undef)
            _sp = 1
            @inbounds _stack[_sp] = Int32(1)

            while _sp > 0
                _nid = @inbounds _stack[_sp]
                _sp -= 1

                if @inbounds(_dv.node_children[1, _nid]) == Int32(-1)
                    _f0 = @inbounds _dv.node_first[_nid]
                    _l0 = @inbounds _dv.node_last[_nid]
                    @inbounds for _k in _f0:_l0
                        $j = _dv.perm[_k]
                        _qx = _dv.coords_t[1, $j]
                        _qy = _dv.coords_t[2, $j]
                        _dx = _qx - _px; _dy = _qy - _py
                        _d2 = _dx*_dx + _dy*_dy
                        if _d2 <= _r_sq
                            $body
                        end
                    end
                else
                    @inbounds for _c in 1:4
                        _cid = _dv.node_children[_c, _nid]
                        if _cid != Int32(-1)
                            _lo1 = _dv.bounds_min[1, _cid]; _lo2 = _dv.bounds_min[2, _cid]
                            _hi1 = _dv.bounds_max[1, _cid]; _hi2 = _dv.bounds_max[2, _cid]
                            _z = zero(_px)
                            _ddx = max(_lo1 - _px, _z); _ddx = max(_ddx, _px - _hi1)
                            _ddy = max(_lo2 - _py, _z); _ddy = max(_ddy, _py - _hi2)
                            if _ddx*_ddx + _ddy*_ddy <= _r_sq
                                _sp += 1
                                @inbounds _stack[_sp] = _cid
                            end
                        end
                    end
                end
            end
        end
    end
end
