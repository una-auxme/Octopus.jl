```@meta
CurrentModule = Octopus
```

# Differentiable edges

Neighbor search is a discrete operation, but the *features* it produces —
relative displacements and distances — are smooth functions of the point
coordinates. [`build_edges_diff`](@ref) exposes that smooth part to a reverse-mode
AD system, so a radius graph can sit inside a learned model and gradients still
reach the coordinates that produced it.

## Loading the extension

The rules live in weak extensions, so nothing is loaded until you ask for it:

```julia
using Octopus
using ChainRulesCore    # activates OctopusChainRulesCoreExt (CPU)
using Zygote            # or any other ChainRules-compatible AD
```

For the GPU path, add `using CUDA` as well — that activates
`OctopusCUDAChainRulesCoreExt`, a separate hand-written kernel that accumulates
the pullback with atomic adds.

## Usage

```julia
using Octopus, ChainRulesCore, Zygote

radius = 0.25f0

function loss(coords)
    tns = TNS(Float32)
    set_search_radius!(tns, radius)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)

    e = build_edges_diff(coords, tns, id, radius)
    return sum(abs2, e.rel_displacement) + 0.5f0 * sum(abs2, e.rel_dist_norm)
end

coords = rand(Float32, 3, 1_000)
g = Zygote.gradient(loss, coords)[1]    # (3, 1000), same shape as coords
```

The return value has the same four fields as [`build_edges`](@ref) —
`senders`, `receivers`, `rel_displacement`, `rel_dist_norm` — but the arrays
are fresh copies rather than views onto the reused [`EdgeBuffer`](@ref). That
matters: without the copy, a later `build_edges` call would overwrite the
values the pullback closed over.

## What is and is not differentiable

Gradients flow to `coords` through `rel_displacement` and `rel_dist_norm`
only. The tree topology, the edge indices and the search radius are treated as
constants:

- `senders` and `receivers` are integer indices with no continuous gradient.
  A loss that depends only on them (e.g. the edge count) yields a zero or
  `nothing` gradient — which is correct, not a bug.
- The set of edges is held fixed at its value for the current coordinates.
  An infinitesimal coordinate perturbation that pushes a pair across the radius
  boundary is not modeled; the gradient is the derivative of the features of
  the *current* edge set.

The setup functions ([`run!`](@ref), [`add_point_set!`](@ref),
[`set_search_radius!`](@ref), [`set_active_search!`](@ref) and friends) are
declared `@non_differentiable`, so you can call them inside a differentiated
function without wrapping each one in `Zygote.@ignore`.

## Keeping `coords` and `tns` in sync

`build_edges_diff` takes `coords` as an explicit first argument because that is
the tensor the AD system tracks — but the *edges* come from whatever geometry
`tns` was last built against. Keeping the two consistent is the caller's job.
The simplest correct pattern is to rebuild inside the differentiated function,
as in the example above. In a training loop that reuses one handle:

```julia
function loss(coords)
    Zygote.@ignore begin
        update_point_set!(tns, id, coords)
        run!(tns)
    end
    e = build_edges_diff(coords, tns, id, radius)
    return my_model(e)
end
```

Pass the same `radius` you gave [`set_search_radius!`](@ref) — it is the
normalization constant for both feature arrays, and a mismatch silently
rescales the gradient rather than erroring.

## Verification

The CPU rules are checked against a 4-point central-difference reference in 2D
and 3D, to `rtol = 1e-4`, for displacement-only, distance-only and combined
losses. The GPU rules are checked element-wise against the CPU rules, which
pins the sign convention on the displacement term. Those tests live in
`test/chainrules.jl` and run in CI; the GPU half additionally requires
`JULIA_OCTOPUS_TEST_CUDA=1` and a working device.

To run them locally:

```console
$ JULIA_OCTOPUS_TEST_CHAINRULES=1 julia --project=. -e 'using Pkg; Pkg.test()'
```

---

Copyright (c) 2026 Josef Jouaux, Chair of Mechatronics, University of Augsburg.
Released under the MIT License.
