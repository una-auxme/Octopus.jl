# Scratch-buffer accounting. `build_scratch` in TNS is sized to the max needed
# across the current run's phases and reused frame-to-frame. Nothing here holds
# per-frame state across calls.

@inline function reset_scratch!(tns)
    empty!(tns.build_scratch)
    return nothing
end
