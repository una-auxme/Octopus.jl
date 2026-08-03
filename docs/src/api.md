```@meta
CurrentModule = Octopus
```

# API reference

Every exported symbol, grouped by role. See the [Guide](guide.md) for how they
fit together.

```@index
```

## Handle and configuration

```@docs
TNS
set_search_radius!
set_refit_mode!
```

## Point sets

```@docs
add_point_set!
resize_point_set!
update_point_set!
```

## Search registration and build

```@docs
set_active_search!
run!
```

## Neighbor iteration (host)

```@docs
for_each_neighbor
get_neighborlist
materialize_all_neighbors!
```

## Neighbor iteration (device)

These require the CUDA extension — see [GPU (CUDA)](gpu.md).

```@docs
device_view
for_each_neighbor_device
@for_each_neighbor_device_inline
@for_each_neighbor_device_inline_2d
```

## Edge construction

```@docs
EdgeBuffer
build_edges
build_edges!
build_edges_diff
```

## Z-sorting

```@docs
prepare_zsort!
apply_zsort!
```

---

Copyright (c) 2026 Josef Jouaux.
Released under the MIT License.
