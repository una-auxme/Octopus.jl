# GPU usage

TreeNSearch.jl runs on NVIDIA GPUs via the `TreeNSearchCUDAExt` extension that
loads when `using CUDA` is in scope.

```julia
using TreeNSearch
using CUDA

xyz = CUDA.rand(Float32, 3, 100_000)
tns = TNS(Float32; ndims = 3)
set_search_radius!(tns, 0.02f0)
id = add_point_set!(tns, xyz)
set_active_search!(tns, id, id)
run!(tns)
```

There are two ways to iterate neighbors from inside your own `@cuda` kernels:

1. **`for_each_neighbor_device(dv, i) do j ... end`** — closure form. Easiest
   to read, fine when the body is read-only or only writes to global memory
   slots indexed by `i`.
2. **`@for_each_neighbor_device_inline dv i j body`** — macro form. Inlines
   the tree traversal directly into your kernel. Use this when the body
   mutates per-thread local scalars (typical for write-pass kernels with a
   per-edge cursor).

The two paths have **identical semantics**: for every neighbor `j` of point
`i` whose distance is `≤ search_radius`, the body executes with `j` bound to
the neighbor index. The user is responsible for filtering the self-pair
`j == i` if their workflow needs it (mirroring the paper's
`for_each_neighbor` contract — the host's `for_each_neighbor` does not
filter self either).

## When the macro matters

If your kernel body mutates a captured scalar (e.g. a per-thread cursor), the
closure form forces Julia to box that scalar. On GPU the box lives in global
memory, so every emitted edge does a load + store of the cursor. The macro
form keeps the cursor in a register and writes it back once.

Measured on an NVIDIA A30 with a write-pass kernel that maintains a
per-thread edge cursor:

| Scene | Closure | Macro | Speedup |
|---|---:|---:|---:|
| 2D N=1k  | 1.62 ms | 0.79 ms | **2.05×** |
| 2D N=5k  | 9.87 ms | 5.63 ms | **1.75×** |
| 2D N=20k | 63.4 ms | 44.4 ms | **1.43×** |

End-to-end build + count + alloc + write (lower is better):

| Scene | TreeNSearch (macro) | PointNeighbors.jl | Speedup |
|---|---:|---:|---:|
| 2D dam-break N=1k  | 1.55 ms | 14.6 ms | 9.4× |
| 2D dam-break N=5k  | 9.3 ms  | 28.4 ms | 3.1× |
| 2D dam-break N=20k | 58 ms   | 73 ms   | 1.3× |
| 3D uniform N=200k  | 38 ms   | 209 ms  | 5.5× |

Edge sets are byte-for-byte identical to the closure path.

## Closure form (read-only body)

When all the body does is write to a per-point output slot (`array[i]`) or
read-only computation, the closure form is fine:

```julia
function count_neighbors!(counts, dv, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > N && return nothing
    @inbounds counts[i] = Int32(0)
    for_each_neighbor_device(dv, i) do j
        if j != Int32(i)
            @inbounds counts[i] += Int32(1)   # global-memory accumulator,
                                              # not a captured scalar
        end
    end
    return nothing
end
```

`counts[i]` is a global-memory slot, so no boxing happens. Each thread writes
to its own index, so there's no race.

## Macro form (per-thread cursor / accumulator)

If the body advances a per-thread cursor, use the macro:

```julia
using StaticArrays  # the macro uses MVector internally; no other deps

function write_edges!(senders, receivers, rd, rdn, cursors, dv, coords, N, radius)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > N && return nothing
    i32 = Int32(i)

    @inbounds oo = cursors[i]      # cursor lives in register from here on
    inv_r = 1f0 / radius
    px = @inbounds coords[1, i]
    py = @inbounds coords[2, i]
    pz = @inbounds coords[3, i]

    @for_each_neighbor_device_inline dv i j begin
        if j != i32
            qx = @inbounds coords[1, j]
            qy = @inbounds coords[2, j]
            qz = @inbounds coords[3, j]
            dx = qx - px; dy = qy - py; dz = qz - pz
            d2 = dx*dx + dy*dy + dz*dz
            @inbounds senders[oo]   = j
            @inbounds receivers[oo] = i32
            @inbounds rd[1, oo] = dx * inv_r
            @inbounds rd[2, oo] = dy * inv_r
            @inbounds rd[3, oo] = dz * inv_r
            @inbounds rdn[oo] = sqrt(d2) * inv_r
            oo += Int32(1)            # mutates a local register, not boxed
        end
    end

    @inbounds cursors[i] = oo
    return nothing
end
```

The macro provides:

- `j` (the neighbor index, `Int32`) bound for the body — pass the symbol you
  want to use as the third argument.
- Pre-checked distance: `body` only fires when `‖coords[i] - coords[j]‖² ≤ radius²`.
- A per-thread `MVector{40, Int32}` traversal stack (sufficient for trees up
  to ~40 levels deep, which covers 2³² particles in 3D).

## Launching the kernel

`device_view(tns, qid, tid)` returns an `isbits` struct safe to pass into a
`@cuda` kernel:

```julia
dv = device_view(tns, id, id)
counts = CUDA.zeros(Int32, N)
@cuda threads = 128 blocks = cld(N, 128) count_neighbors!(counts, dv, Int32(N))
CUDA.synchronize()
```

## Scope and limits

- 2D and 3D are both native. Construct `TNS(Float32; ndims=2)` for a
  quadtree; the GPU path uses a 4-way fan-out and the
  `@for_each_neighbor_device_inline_2d` macro for write-pass kernels.
  Coords are `(NDIMS, N)` matrices on either device — no padding.
- The macro requires `StaticArrays` to be loadable; that dependency is
  already in TreeNSearch's `Project.toml`, so calling code only needs
  `using TreeNSearch, CUDA` (and optionally `using StaticArrays` if you also
  use `MVector` directly in your own code).
- Stack depth is fixed at 40. If your tree is deeper (e.g. very-extreme leaf
  size ratios), increase the constant in [src/macros.jl](../../src/macros.jl)
  and rebuild — open an issue if you hit this in practice.
- `for_each_neighbor_device` is the public function — keep it for read-only
  bodies. The macro is the recommended path for write-pass kernels.

## Performance recipe

For a write-pass kernel modelled after `point_neighbor_ns`:

1. **Per-point cursor in a register**: hold `oo = cursors[i]` once, advance
   in-register inside the macro body, write back at the very end.
2. **Precompute `inv_r = 1f0 / radius`**: replace `dx / radius` with
   `dx * inv_r` in the body. On Float32 the compiler folds 4–6 divisions per
   edge into multiplies.
3. **Match `coords` shape to the body's actual dimensions**: when 2D-padded
   coords are passed to TreeNSearch, your kernel can still compute features
   in the original 2D space (the macro filters by 3D distance, which equals
   2D distance when z = 0).
4. **One thread per query point**: the simplest, most predictable mapping.
   Warp-cooperative descent is a v0.2 optimization.

Following this recipe gives you the macro-form numbers above on any modern
NVIDIA GPU.
