#
# Copyright (c) 2026 Josef Jouaux
# Copyright (c) 2022-present, CompactNSearch contributors
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# Threading helpers. We keep this intentionally small: Polyester on CPU via
# `@batch minbatch=...` inline, plain `for` for tiny loops.

# Geometric upsize for reused scratch buffers so we avoid repeated `resize!`
# churn at steady state.
@inline function ensure_capacity!(buf::Vector{T}, n::Integer) where {T}
    if length(buf) < n
        resize!(buf, max(n, 2 * length(buf) + 16))
    end
    return buf
end

@inline function truncate!(buf::Vector, n::Integer)
    resize!(buf, n)
    return buf
end

# ---------------- query-loop chunking --------------------------------------
#
# The two-pass query loops (build_edges, materialize) are parallel over query
# points: pass A writes only `counts[i+1]`, and pass B writes only the slice
# `[counts[i]+1 .. counts[i+1]]`, which is disjoint per point. What they cannot
# share is the traversal stack, so we split the point range into contiguous
# chunks and give each chunk its own stack out of `tns.query_stacks`.
#
# Chunk index — not `Threads.threadid()` — is the stack index. Polyester runs
# `@batch` bodies on its own worker pool, where `threadid()` is not a reliable
# per-task identity; indexing by chunk sidesteps that entirely.

# Contiguous split of 1:n into `nchunks` near-equal ranges (remainder spread
# over the leading chunks).
@inline function chunk_bounds(n::Int, nchunks::Int, c::Int)
    base, rem = divrem(n, nchunks)
    lo = (c - 1) * base + min(c - 1, rem) + 1
    hi = lo + base - 1 + (c <= rem ? 1 : 0)
    return lo, hi
end

# One chunk per thread at most, never more chunks than we have stacks, and
# never so many that a chunk is too small to pay for the scheduling.
@inline function n_query_chunks(n::Int, nstacks::Int, minwork::Int=1024)
    n <= minwork && return 1
    return clamp(cld(n, minwork), 1, min(nstacks, Threads.nthreads()))
end
