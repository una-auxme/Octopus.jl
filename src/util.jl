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
