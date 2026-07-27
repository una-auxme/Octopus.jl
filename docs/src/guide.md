```@meta
CurrentModule = Octopus
```

# Guide

This page walks through the whole workflow. Everything here runs on the CPU;
see [GPU (CUDA)](gpu.md) for the device path and
[Differentiable edges](gnn.md) for gradients.

## The four-step workflow

Every Octopus program has the same shape:

1. Create a handle with [`TNS`](@ref) and set the radius with
   [`set_search_radius!`](@ref).
2. Register coordinates with [`add_point_set!`](@ref).
3. Declare which set queries which with [`set_active_search!`](@ref).
4. Build with [`run!`](@ref), then query.

```julia
using Octopus

xyz = rand(Float32, 3, 100_000)

tns = TNS(Float32; ndims = 3)     # 1. handle
set_search_radius!(tns, 0.02f0)
id = add_point_set!(tns, xyz)     # 2. point set
set_active_search!(tns, id, id)   # 3. self-search
run!(tns)                          # 4. build
```

### Coordinate layout

Coordinates are `(NDIMS, N)` matrices — one column per point, so each point's
components are contiguous. `TNS(Float32; ndims = 2)` builds a quadtree over
`(2, N)` input; the default `ndims = 3` builds an octree over `(3, N)`.

The element type of the coordinates must match the `TNS` element type.
`TNS{Float32}` is the tested and benchmarked configuration; `TNS{Float64}`
works but is unbenchmarked.

!!! note "Coordinates are referenced, not copied"
    [`add_point_set!`](@ref) keeps a reference to your matrix. Mutating it in
    place and calling [`run!`](@ref) again is the intended update path for a
    simulation loop — see [Updating points between steps](#Updating-points-between-steps).

### Point sets and active pairs

A `TNS` can hold several point sets. `add_point_set!` returns an integer id,
and an *active pair* `(qid, tid)` declares that points from set `qid` will look
for neighbors in set `tid`. Pairs are directed, so a mutual search between two
sets needs both directions registered:

```julia
fluid = add_point_set!(tns, fluid_xyz)
walls = add_point_set!(tns, wall_xyz)

set_active_search!(tns, fluid, fluid)   # fluid ↔ fluid
set_active_search!(tns, fluid, walls)   # fluid → walls
set_active_search!(tns, walls, fluid)   # walls → fluid
run!(tns)
```

Registering a pair twice is a no-op: the existing buffers are kept and refilled
by the next `run!`. All point sets on one handle must live on the same backend
— you cannot mix host and device arrays.

## Iterating neighbors

[`for_each_neighbor`](@ref) is the primitive. It visits neighbors during tree
traversal instead of collecting them, so the hot loop allocates nothing:

```julia
for i in 1:size(xyz, 2)
    for_each_neighbor(tns, id, id, i) do j
        # j is an Int32 index into the target set
    end
end
```

!!! warning "The self-pair is not filtered"
    When `qid == tid`, the callback also fires for `j == i`. This matches the
    paper's contract. Skip it yourself if your kernel needs to:

    ```julia
    i32 = Int32(i)
    for_each_neighbor(tns, id, id, i) do j
        j == i32 && return
        # ...
    end
    ```

The traversal stack is per-thread, so calling `for_each_neighbor` from inside
`Threads.@threads` or a `Polyester.@batch` loop is safe.

### Collecting a list instead

[`get_neighborlist`](@ref) allocates a fresh `Vector{Int32}` per call and
*does* exclude the self-pair. Convenient for tests and exploration, wasteful in
a hot loop:

```julia
neighbors = get_neighborlist(tns, id, id, 1)
```

### Materializing all lists at once

[`materialize_all_neighbors!`](@ref) fills CSR-style buffers for every active
pair in one pass. This is opt-in — the default iteration path stores nothing.
Read the result off the handle:

```julia
materialize_all_neighbors!(tns)

pair_idx = findfirst(==((Int32(id), Int32(id))), tns.active_pairs)
buf = tns.neighbor_buffers[pair_idx]

# offsets are 0-based; neighbors of point i are:
lo = buf.offsets[i] + 1
hi = buf.offsets[i + 1]
js = buf.flat[lo:hi]
```

Materializing costs memory — roughly 21 bytes per particle at steady state
versus 15 for the lazy path in the package's own benchmark — so reach for it
only when a downstream consumer genuinely needs a dense list. It is CPU-only;
on GPU use [`build_edges`](@ref) or in-kernel iteration.

## Edge buffers for graph models

[`build_edges`](@ref) produces a flat-COO edge list — the shape a message-passing
GNN wants — and returns it as a NamedTuple:

```julia
e = build_edges(tns, id, id)

e.senders           # Vector{Int32}, the target index j
e.receivers         # Vector{Int32}, the query index i
e.rel_displacement  # (NDIMS, n_edges), (q[:,i] - t[:,j]) / radius
e.rel_dist_norm     # (1, n_edges),     ‖q[:,i] - t[:,j]‖ / radius
```

Both feature arrays are normalized by the search radius, so they land in
`[-1, 1]` and `[0, 1]` respectively without further scaling. Unlike
`for_each_neighbor`, a self-pair search *does* exclude `j == i` here.

[`build_edges!`](@ref) is the same operation returning the underlying
[`EdgeBuffer`](@ref). Either way the storage is reused across calls: a second
`build_edges` on the same pair overwrites the arrays in place. Copy them if you
need the values to outlive the next call.

## Updating points between steps

For a simulation loop, mutate the coordinate matrix you already registered and
rebuild:

```julia
for step in 1:n_steps
    integrate!(xyz)        # your update, in place
    run!(tns)              # rebuild against the new positions
    # ... query ...
end
```

If you need to swap in a *different* matrix, the two entry points differ in
strictness:

- [`update_point_set!`](@ref) requires the same `N` and throws
  `DimensionMismatch` otherwise — the safer choice when the count is meant to
  be fixed.
- [`resize_point_set!`](@ref) accepts any `N`.

Both mark the set dirty; `run!` only rebuilds dirty sets, so untouched sets
carry their tree over.

!!! warning "Rebuilding invalidates materialized data"
    `run!` clears the materialized flags on all neighbor and edge buffers.
    Anything you read out of them must be refilled — or copied — after each
    rebuild.

## Refit mode

For time-stepping workloads where points move much less than one cell per
step, [`set_refit_mode!`](@ref) enables the paper's almost-sorted path:

```julia
set_refit_mode!(tns, true)
```

With refit mode on, the Morton codes from the previous build are retained so
the next `run!` can detect cell crossings instead of re-deriving the ordering
from scratch. With it off (the default) that scratch is released after every
`run!`, which keeps steady-state memory lower. Turn it on when you rebuild
every step and positions change slowly; leave it off for one-shot queries or
when points jump arbitrarily.

## Z-sorting user data

The build computes a Morton-order permutation. Reordering your per-point data
to match it improves cache locality for everything downstream.
[`apply_zsort!`](@ref) returns a permuted copy of any array whose last axis is
the point axis — a length-`N` vector or an `(F, N)` matrix:

```julia
run!(tns)
prepare_zsort!(tns)               # asserts the permutation is available

masses    = apply_zsort!(tns, id, masses)
velocities = apply_zsort!(tns, id, velocities)   # (3, N) works too
```

[`prepare_zsort!`](@ref) is a precondition check that mirrors the C++ API — it
gives a clear error if `run!` has not been called yet. It allocates nothing and
is safe (but unnecessary) to call every step. Note that `apply_zsort!` returns
a *new* array rather than permuting in place, despite the `!`.

## Which API is which?

These names mirror the paper and the C++ TreeNSearch reference implementation:
[`TNS`](@ref), [`set_search_radius!`](@ref), [`add_point_set!`](@ref),
[`resize_point_set!`](@ref), [`set_active_search!`](@ref), [`run!`](@ref),
[`for_each_neighbor`](@ref), [`get_neighborlist`](@ref),
[`prepare_zsort!`](@ref), [`apply_zsort!`](@ref).

These are Julia-only additions layered on top, for Julia idioms or GPU usage:
[`set_refit_mode!`](@ref), [`update_point_set!`](@ref),
[`materialize_all_neighbors!`](@ref), [`build_edges`](@ref),
[`build_edges!`](@ref), [`EdgeBuffer`](@ref), [`build_edges_diff`](@ref),
[`device_view`](@ref), [`for_each_neighbor_device`](@ref),
[`@for_each_neighbor_device_inline`](@ref) and its `_2d` variant.

---

Copyright (c) 2026 Josef Jouaux, Chair of Mechatronics, University of Augsburg.
Released under the MIT License.
