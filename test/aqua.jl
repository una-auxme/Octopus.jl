#
# Copyright (c) 2026 Josef Jouaux
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# Package-hygiene checks. The one that earns its keep here is `stale_deps`:
# `SIMD` and `Atomix` sat in [deps] unused for the whole of v0.1, and the README
# advertised a SIMD fast path that did not exist. This test makes that class of
# drift fail CI instead of surviving to a release.

using Test
using Aqua
using Octopus

Aqua.test_all(
    Octopus;
    # Ambiguities are checked separately below so we can scope them to Octopus
    # and not inherit unrelated ones from Base/StaticArrays.
    ambiguities = false,
    # `persistent_tasks` generates a throwaway package, then runs a *nested*
    # `Pkg.precompile()` subprocess which itself forks parallel precompile
    # workers and a 7z to read the registry. Under a per-user process cap
    # (`ulimit -u` is 1024 on the LICCA login nodes, and a running test session
    # already holds ~750 threads) that fork fails:
    #
    #   IOError: could not spawn `.../7z x .../General.tar.gz -so`:
    #            resource temporarily unavailable (EAGAIN)
    #
    # Aqua reports the dead subprocess as "package has persistent tasks", so
    # the check fails intermittently — ~50% of in-suite runs here — for reasons
    # that have nothing to do with this package. Run standalone, with process
    # headroom, it passes 12/12.
    #
    # It is also close to vacuous for Octopus: `grep -rE '@async|@spawn|Timer\('
    # src/ ext/` finds nothing, so there is no construct here that could leave a
    # task alive. Re-enable it (drop this line) if the package ever gains one.
    persistent_tasks = false,
)

@testset "ambiguities (Octopus only)" begin
    @test isempty(Test.detect_ambiguities(Octopus; recursive = true))
end
