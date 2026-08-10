#
# Copyright (c) 2026 Josef Jouaux
# Copyright (c) 2022-present, CompactNSearch contributors
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Test
using Octopus
using ChainRulesCore
using Zygote
using Random

# Pre-publication checks for the ChainRulesCore extension.
#
# Gated behind JULIA_OCTOPUS_TEST_CHAINRULES=1 (see runtests.jl) because
# Zygote + transitive deps add several seconds of precompile time. Run on
# release tags / before publication, not on every commit:
#
#     JULIA_OCTOPUS_TEST_CHAINRULES=1 julia --project=. -e 'using Pkg; Pkg.test()'

# 4-point central difference. Smaller systematic error than the usual
# 2-point form, important when the analytic gradient is tight to 1e-7.
function _central_diff(loss, coords::Matrix{Float64}, ε::Float64)
    g = zeros(Float64, size(coords))
    @inbounds for k in eachindex(coords)
        c1 = copy(coords); c1[k] += 2ε; f1 = loss(c1)
        c2 = copy(coords); c2[k] +=  ε; f2 = loss(c2)
        c3 = copy(coords); c3[k] -=  ε; f3 = loss(c3)
        c4 = copy(coords); c4[k] -= 2ε; f4 = loss(c4)
        # NB: `8f2` would be parsed as the Float32 literal 800.0f0 — keep the `*`.
        g[k] = (-f1 + 8.0 * f2 - 8.0 * f3 + f4) / (12 * ε)
    end
    return g
end

# ---- 1. @non_differentiable setup ops don't trip Zygote ------------------

@testset "build_edges_diff: Zygote does not crash on setup ops" begin
    Random.seed!(8001)
    coords = rand(Float64, 3, 30)
    radius = 0.2

    function loss(c)
        tns = TNS(Float64)
        set_search_radius!(tns, radius)
        i = add_point_set!(tns, c)
        set_active_search!(tns, i, i)
        run!(tns)
        e = build_edges_diff(c, tns, i, radius)
        return sum(abs2, e.rel_displacement)
    end

    @test loss(coords) ≥ 0
    g = Zygote.gradient(loss, coords)[1]
    @test g isa Matrix{Float64}
    @test size(g) == size(coords)
end

# ---- 2. CPU rrule matches central diff (3D) ------------------------------

@testset "build_edges_diff CPU 3D: gradient ≈ central-diff" begin
    Random.seed!(8002)
    coords = rand(Float64, 3, 40)
    radius = 0.25

    function loss(c)
        tns = TNS(Float64)
        set_search_radius!(tns, radius)
        i = add_point_set!(tns, c)
        set_active_search!(tns, i, i)
        run!(tns)
        e = build_edges_diff(c, tns, i, radius)
        return sum(abs2, e.rel_displacement) + 0.5 * sum(abs2, e.rel_dist_norm)
    end

    g_zygote = Zygote.gradient(loss, coords)[1]
    g_fd     = _central_diff(loss, coords, 1e-4)
    @test g_zygote ≈ g_fd rtol = 1e-4 atol = 1e-7
end

# ---- 3. CPU rrule matches central diff (2D quadtree) ---------------------

@testset "build_edges_diff CPU 2D: gradient ≈ central-diff" begin
    Random.seed!(8003)
    coords = rand(Float64, 2, 40)
    radius = 0.25

    function loss(c)
        tns = TNS(Float64; ndims = 2)
        set_search_radius!(tns, radius)
        i = add_point_set!(tns, c)
        set_active_search!(tns, i, i)
        run!(tns)
        e = build_edges_diff(c, tns, i, radius)
        return sum(abs2, e.rel_displacement) + 0.5 * sum(abs2, e.rel_dist_norm)
    end

    g_zygote = Zygote.gradient(loss, coords)[1]
    g_fd     = _central_diff(loss, coords, 1e-4)
    @test g_zygote ≈ g_fd rtol = 1e-4 atol = 1e-7
end

# ---- 4. rel_displacement-only loss ---------------------------------------

@testset "build_edges_diff: rel_displacement-only loss" begin
    Random.seed!(8004)
    coords = rand(Float64, 3, 40)
    radius = 0.25

    function loss(c)
        tns = TNS(Float64)
        set_search_radius!(tns, radius)
        i = add_point_set!(tns, c)
        set_active_search!(tns, i, i)
        run!(tns)
        e = build_edges_diff(c, tns, i, radius)
        return sum(abs2, e.rel_displacement)
    end

    g_zygote = Zygote.gradient(loss, coords)[1]
    g_fd     = _central_diff(loss, coords, 1e-4)
    @test g_zygote ≈ g_fd rtol = 1e-4 atol = 1e-7
end

# ---- 5. rel_dist_norm-only loss ------------------------------------------

@testset "build_edges_diff: rel_dist_norm-only loss" begin
    Random.seed!(8005)
    coords = rand(Float64, 3, 40)
    radius = 0.25

    function loss(c)
        tns = TNS(Float64)
        set_search_radius!(tns, radius)
        i = add_point_set!(tns, c)
        set_active_search!(tns, i, i)
        run!(tns)
        e = build_edges_diff(c, tns, i, radius)
        return sum(abs2, e.rel_dist_norm)
    end

    g_zygote = Zygote.gradient(loss, coords)[1]
    g_fd     = _central_diff(loss, coords, 1e-4)
    @test g_zygote ≈ g_fd rtol = 1e-4 atol = 1e-7
end

# ---- 6. Index-only loss has no continuous gradient -----------------------

@testset "build_edges_diff: index-only loss yields zero gradient" begin
    Random.seed!(8006)
    coords = rand(Float64, 3, 30)
    radius = 0.2

    function loss(c)
        tns = TNS(Float64)
        set_search_radius!(tns, radius)
        i = add_point_set!(tns, c)
        set_active_search!(tns, i, i)
        run!(tns)
        e = build_edges_diff(c, tns, i, radius)
        return Float64(length(e.senders))   # discrete output
    end

    g = Zygote.gradient(loss, coords)[1]
    @test g === nothing || all(iszero, g)
end

# ---- 7. Setup ops directly are non-differentiable ------------------------

@testset "build_edges_diff: setup-op rrules return NoTangent" begin
    tns = TNS(Float64)
    set_search_radius!(tns, 0.1)
    coords = rand(Float64, 3, 10)

    _, pb_add = ChainRulesCore.rrule(add_point_set!, tns, coords)
    out_add = pb_add(1.0)
    @test all(t -> t isa ChainRulesCore.NoTangent, out_add)

    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    _, pb_run = ChainRulesCore.rrule(run!, tns)
    out_run = pb_run(1.0)
    @test all(t -> t isa ChainRulesCore.NoTangent, out_run)
end

# ---- 7b. apply_zsort is differentiable (gather → scatter) ----------------
#
# Before the rrule, this path failed on CPU with "Mutating arrays is not
# supported -- called setindex!" (apply_zsort_cpu! fills its output with
# setindex! even though the function is externally pure), while the CUDA path
# worked because it is written as fancy indexing. These pin the CPU behaviour.

# Build a TNS whose permutation is non-trivial, and return it with the perm.
function _zsort_fixture(N, seed)
    Random.seed!(seed)
    coords = rand(Float32, 3, N)
    tns = TNS(Float32)
    set_search_radius!(tns, 0.15f0)
    id = add_point_set!(tns, coords)
    set_active_search!(tns, id, id)
    run!(tns)
    return tns, id, copy(tns.permutation[id])
end

@testset "apply_zsort: CPU gradient exists (1-D)" begin
    tns, id, perm = _zsort_fixture(200, 8200)
    x = rand(Float32, 200)
    # A permutation is a bijection, so sum(abs2, permuted) == sum(abs2, x)
    # and the gradient must come back as exactly 2x, unpermuted.
    g = Zygote.gradient(v -> sum(abs2, apply_zsort(tns, id, v)), x)[1]
    @test g ≈ 2 .* x
    @test size(g) == size(x)
    @test !all(==(perm[1]), perm)   # guard: the fixture's perm is non-trivial
end

@testset "apply_zsort: CPU gradient is the scatter, not the gather (1-D)" begin
    # The abs2 test above is invariant under inverting the permutation, so it
    # cannot catch a transposed adjoint. This one can: with a non-symmetric
    # weight vector, d/dx sum(w .* x[perm]) is w scattered through perm.
    tns, id, perm = _zsort_fixture(200, 8201)
    x = rand(Float32, 200)
    w = Float32.(1:200)

    g = Zygote.gradient(v -> sum(w .* apply_zsort(tns, id, v)), x)[1]

    expected = zeros(Float32, 200)
    expected[perm] = w
    @test g ≈ expected
    # Sanity: the gather (wrong direction) really would differ here.
    @test !(g ≈ w[perm])
end

@testset "apply_zsort: CPU gradient (2-D, permutes columns)" begin
    tns, id, perm = _zsort_fixture(150, 8202)
    X = rand(Float32, 4, 150)
    W = Float32.(reshape(1:(4 * 150), 4, 150))

    g = Zygote.gradient(V -> sum(W .* apply_zsort(tns, id, V)), X)[1]

    expected = zeros(Float32, 4, 150)
    expected[:, perm] = W
    @test size(g) == size(X)
    @test g ≈ expected
end

@testset "apply_zsort: CPU gradient ≈ central diff" begin
    tns, id, _ = _zsort_fixture(60, 8203)
    X64 = rand(Float64, 2, 60)
    loss(V) = sum(abs2, apply_zsort(tns, id, V) .* 1.7)

    g  = Zygote.gradient(loss, X64)[1]
    fd = _central_diff(loss, X64, 1e-5)
    @test g ≈ fd rtol = 1e-6
end

@testset "apply_zsort: zero upstream tangent yields zero gradient" begin
    tns, id, _ = _zsort_fixture(50, 8204)
    x = rand(Float32, 50)
    # Loss ignores the z-sorted values entirely.
    g = Zygote.gradient(v -> sum(abs2, apply_zsort(tns, id, v)) * 0.0f0, x)[1]
    @test g === nothing || all(iszero, g)
end

# ---- 8. GPU rrule matches the (FD-verified) CPU rrule --------------------
# The GPU `build_edges_diff` rrule is a SEPARATE hand-written atomic-add kernel
# (OctopusCUDAChainRulesCoreExt), so the finite-difference tests above —
# all CPU — do not exercise it. This guards the displacement/distance
# sign-convention regression: the kernel once applied the rel_displacement term
# with `+sender, −receiver` while the rel_dist_norm term used `+receiver,
# −sender`, so `g_gpu == −g_cpu` on the displacement contribution (full-loss
# cos(GPU,CPU) ≈ 0.77; disp-only ≈ −1). CPU is the reference because it is
# finite-difference-verified above; both run Float32 so the edge sets — and
# hence the element-wise comparison — are exact. The `:disp`-only case is the
# one that pins the sign. Requires JULIA_OCTOPUS_TEST_CUDA=1.
if get(ENV, "JULIA_OCTOPUS_TEST_CUDA", "0") == "1"
    using CUDA
    if !CUDA.functional()
        @info "CUDA unavailable; skipping GPU build_edges_diff gradient tests"
    else
        function _edge_loss(c, radius, ndims, mode)
            T = eltype(c)
            tns = TNS(T; ndims = ndims)
            set_search_radius!(tns, radius)
            i = add_point_set!(tns, c)
            set_active_search!(tns, i, i)
            run!(tns)
            e = build_edges_diff(c, tns, i, radius)
            if mode === :disp
                return sum(abs2, e.rel_displacement)
            elseif mode === :dist
                return sum(abs2, e.rel_dist_norm)
            else
                return sum(abs2, e.rel_displacement) + T(0.5) * sum(abs2, e.rel_dist_norm)
            end
        end

        @testset "GPU build_edges_diff: $(D)D $(mode) matches CPU rrule" for
                D in (2, 3), mode in (:full, :disp, :dist)
            Random.seed!(8100 + D)
            coords = rand(Float32, D, 64)
            radius = 0.25f0

            g_cpu = Zygote.gradient(c -> _edge_loss(c, radius, D, mode), coords)[1]
            g_gpu = Zygote.gradient(c -> _edge_loss(c, radius, D, mode), CuArray(coords))[1]

            @test Array(g_gpu) ≈ g_cpu rtol = 1.0f-3 atol = 1.0f-5
        end

        # The `apply_zsort` rrule is generic, so on GPU it now *replaces* the
        # native Zygote `getindex` adjoint the CUDA path relied on before. The
        # scatter is written vectorised (`grad[:, perm] = Δ`) specifically so it
        # stays on the device; a scalar-indexing regression would either throw
        # under CUDA's scalar-iteration guard or silently crawl. These pin both
        # the value and the device-residency.
        @testset "GPU apply_zsort gradient matches CPU ($(D)-D)" for D in (1, 2)
            Random.seed!(8300 + D)
            coords = rand(Float32, 3, 128)

            function _zs_loss(v, c)
                tns = TNS(Float32)
                set_search_radius!(tns, 0.15f0)
                i = add_point_set!(tns, c)
                set_active_search!(tns, i, i)
                run!(tns)
                z = apply_zsort(tns, i, v)
                return sum(abs2, z) + 2.0f0 * sum(z)
            end

            x_cpu = D == 1 ? rand(Float32, 128) : rand(Float32, 4, 128)
            x_gpu = CuArray(x_cpu)

            g_cpu = Zygote.gradient(v -> _zs_loss(v, coords), x_cpu)[1]
            g_gpu = Zygote.gradient(v -> _zs_loss(v, CuArray(coords)), x_gpu)[1]

            @test g_gpu isa CuArray            # stayed on the device
            @test size(g_gpu) == size(x_cpu)
            @test Array(g_gpu) ≈ g_cpu rtol = 1.0f-5
        end
    end
end
