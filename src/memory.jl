# Scratch-buffer accounting. `build_scratch` in TNS is a reused Int32 buffer
# handed to `build_edges_cpu!` for its Pass-A neighbor counts / exclusive-scan
# offsets, sized on demand and reused frame-to-frame so the differentiable edge
# build allocates nothing per call. Nothing here holds per-frame state across
# calls.

@inline function reset_scratch!(tns)
    empty!(tns.build_scratch)
    return nothing
end
