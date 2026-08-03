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
    end
end
