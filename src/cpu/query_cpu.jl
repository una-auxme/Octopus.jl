# Opt-in CSR materialization. Two-pass:
#   A: count neighbors per querying point -> offsets (exclusive scan)
#   B: write into pre-sized flat[] at offset + cursor
# Self-pair (qid == tid) excludes j == i.
#
# 3D and 2D overloads dispatch on the tree type.

function materialize_cpu!(
    buf::NeighborBuffer,
    tree::Octree{T,3},
    perm_t::Vector{Int32},
    coords_q::AbstractMatrix{T},
    coords_t::AbstractMatrix{T},
    stack::Vector{Int32},
    r::T,
    exclude_self::Bool,
) where {T}
    n_q = size(coords_q, 2)
    r_sq = r * r

    # Pass A — counts
    resize!(buf.offsets, n_q + 1)
    fill!(buf.offsets, Int32(0))

    @inbounds for i in 1:n_q
        px = coords_q[1, i]; py = coords_q[2, i]; pz = coords_q[3, i]
        c = Int32(0)
        if exclude_self
            _traverse!(tree, coords_t, perm_t, stack, px, py, pz, r_sq) do j
                if Int32(j) != Int32(i)
                    c += Int32(1)
                end
            end
        else
            _traverse!(tree, coords_t, perm_t, stack, px, py, pz, r_sq) do _
                c += Int32(1)
            end
        end
        buf.offsets[i + 1] = c
    end

    # Exclusive scan -> CSR row offsets starting at 0.
    @inbounds for i in 1:n_q
        buf.offsets[i + 1] += buf.offsets[i]
    end

    total = @inbounds buf.offsets[n_q + 1]
    resize!(buf.flat, total)

    # Pass B — writes
    @inbounds for i in 1:n_q
        px = coords_q[1, i]; py = coords_q[2, i]; pz = coords_q[3, i]
        cursor = buf.offsets[i]
        if exclude_self
            _traverse!(tree, coords_t, perm_t, stack, px, py, pz, r_sq) do j
                if Int32(j) != Int32(i)
                    cursor += Int32(1)
                    buf.flat[cursor] = Int32(j)
                end
            end
        else
            _traverse!(tree, coords_t, perm_t, stack, px, py, pz, r_sq) do j
                cursor += Int32(1)
                buf.flat[cursor] = Int32(j)
            end
        end
    end

    buf.materialized = true
    return buf
end

function materialize_cpu!(
    buf::NeighborBuffer,
    tree::Octree{T,2},
    perm_t::Vector{Int32},
    coords_q::AbstractMatrix{T},
    coords_t::AbstractMatrix{T},
    stack::Vector{Int32},
    r::T,
    exclude_self::Bool,
) where {T}
    n_q = size(coords_q, 2)
    r_sq = r * r

    resize!(buf.offsets, n_q + 1)
    fill!(buf.offsets, Int32(0))

    @inbounds for i in 1:n_q
        px = coords_q[1, i]; py = coords_q[2, i]
        c = Int32(0)
        if exclude_self
            _traverse!(tree, coords_t, perm_t, stack, px, py, r_sq) do j
                if Int32(j) != Int32(i)
                    c += Int32(1)
                end
            end
        else
            _traverse!(tree, coords_t, perm_t, stack, px, py, r_sq) do _
                c += Int32(1)
            end
        end
        buf.offsets[i + 1] = c
    end

    @inbounds for i in 1:n_q
        buf.offsets[i + 1] += buf.offsets[i]
    end

    total = @inbounds buf.offsets[n_q + 1]
    resize!(buf.flat, total)

    @inbounds for i in 1:n_q
        px = coords_q[1, i]; py = coords_q[2, i]
        cursor = buf.offsets[i]
        if exclude_self
            _traverse!(tree, coords_t, perm_t, stack, px, py, r_sq) do j
                if Int32(j) != Int32(i)
                    cursor += Int32(1)
                    buf.flat[cursor] = Int32(j)
                end
            end
        else
            _traverse!(tree, coords_t, perm_t, stack, px, py, r_sq) do j
                cursor += Int32(1)
                buf.flat[cursor] = Int32(j)
            end
        end
    end

    buf.materialized = true
    return buf
end
