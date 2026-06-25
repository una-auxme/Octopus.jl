# Reproducer for the residual GPU illegal-address (CUDA code 700) in the native
# CUDA forward path at N=40000, 2D — the config where the node-buffer
# overflow-retry path runs (default leaf=32 ⇒ est = 4·⌈40000/32⌉+64 = 5064 <
# n_nodes = 5233, so the build overflows attempt 1 and retries).
#
# The package's "GPU build node-capacity overflow is memory-safe" test verifies
# the topology build and edge fill have no out-of-bounds WRITE. A separate fault
# remains: it is in the Octopus GPU BUILD path (the stack trace surfaces at the
# host sync near cuda_build.jl:253, i.e. a build-level kernel ran the bad access),
# it is an out-of-bounds READ (so the write-canary test misses it), and it is
# layout-dependent. Likely cause: when a parent node's child block straddles
# `capacity`, the bailing thread reserves the slots in the counter but writes
# none of them; the next BFS level then reads UNINITIALIZED node_first/node_last
# for the sub-`capacity` slots (the `g > capacity` guard does not catch g ≤
# capacity), and the garbage [f,l] drives an out-of-range perm[] read in the
# octant binary search. compute-sanitizer pinpoints the exact kernel/line:
#
# On an A30 + CUDA.jl pooled allocator the fragment!() churn below reproduces
# the fault reliably on a PLAIN run (code 700). compute-sanitizer additionally
# names the offending access and works regardless of allocator layout:
#
#   # one-time scratch env (CUDA + ChainRulesCore + Random on top of Octopus;
#   # ChainRulesCore activates the differentiable build_edges_diff GPU method):
#   julia --project=@octorepro -e 'using Pkg; \
#       Pkg.develop(path=raw"<abs path to this repo>"); \
#       Pkg.add(["CUDA", "ChainRulesCore"])'
#
#   compute-sanitizer --tool memcheck \
#     julia --project=@octorepro repro_gpu_oob.jl
#
# A plain `julia --project=@octorepro repro_gpu_oob.jl` may or may not fault
# depending on allocator layout; the fragment!() churn below tries to provoke it,
# but compute-sanitizer is the deterministic path.

using Octopus, CUDA, ChainRulesCore, Random  # ChainRulesCore ⇒ build_edges_diff GPU method

const D, N, r = 2, 40_000, 0.072f0

function jittered_lattice(D, N, radius; seed = 1)
    Random.seed!(seed)
    s = radius / 1.6f0
    per = ceil(Int, N^(1 / D))
    grids = ntuple(_ -> 0:(per - 1), D)
    pts = Matrix{Float32}(undef, D, per^D)
    idx = 1
    for c in Iterators.product(grids...)
        for d in 1:D
            pts[d, idx] = (c[d] + 0.25f0 * (rand(Float32) - 0.5f0)) * s
        end
        idx += 1
    end
    return pts[:, 1:min(N, size(pts, 2))]
end

# The full forward path a GNN caller runs: build + differentiable edges +
# self-loop append (mirrors GraphNetSim's tns_gpu).
function forward(pos, r)
    Dl = size(pos, 1)
    n = size(pos, 2)
    tns = TNS(eltype(pos); ndims = Dl)
    set_search_radius!(tns, r)
    id = add_point_set!(tns, pos)
    set_active_search!(tns, id, id)
    run!(tns)
    e = build_edges_diff(pos, tns, 1, r)
    s = similar(e.senders, n)
    copyto!(s, Int32.(1:n))
    senders = vcat(e.senders, s)
    receivers = vcat(e.receivers, copy(s))
    disp = hcat(e.rel_displacement, fill!(similar(e.rel_displacement, Dl, n), 0))
    dist = hcat(e.rel_dist_norm, fill!(similar(e.rel_dist_norm, 1, n), 0))
    return tns, senders, receivers, disp, dist
end

# Best-effort heap fragmentation: alloc a spread of differently-sized device
# blobs, free every other one, leaving holes. Approximates the allocator churn a
# prior consumer (e.g. a grid-based neighbour search) leaves behind, which is
# what makes the residual OOB access land on an unmapped page on a plain run.
function fragment!()
    blobs = CuArray{Float32}[]
    for k in 1:24
        push!(blobs, CUDA.rand(Float32, (k % 8 + 1) * 1_000_000))
    end
    for k in 1:2:length(blobs)
        CUDA.unsafe_free!(blobs[k])
    end
    GC.gc()
    CUDA.reclaim()
    return nothing
end

CUDA.functional() || error("CUDA not functional")
println("GPU = ", CUDA.name(CUDA.device()))
leaf = 32
NCH = 1 << D
println("config: D=$D N=$N r=$r  est(attempt-1 capacity)=", max(NCH * cld(N, leaf) + 64, 64))

# Warm up the smaller (non-overflowing) size, then churn the allocator.
warm = CuArray(jittered_lattice(D, 5_000, r))
forward(warm, r)
CUDA.synchronize()
warm = nothing
GC.gc()
CUDA.reclaim()
fragment!()

pos = CuArray(jittered_lattice(D, N, r))
for k in 1:16
    tns, sn, _, _, _ = forward(pos, r)
    CUDA.synchronize()
    if k == 1 || k == 16
        println("rep $k: n_nodes=", Int(tns.trees[1].n_nodes), "  edges=", length(sn),
                "  (overflow-retry path ", Int(tns.trees[1].n_nodes) > max(NCH * cld(N, leaf) + 64, 64) ? "TRIGGERED)" : "not hit)")
    end
    k % 4 == 0 && fragment!()
end
println(">>> reached end without a plain-run fault on this layout; ",
        "run under compute-sanitizer to flag the OOB access deterministically")
