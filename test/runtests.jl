using ComradeBase, StaticArrays, JET
using StructArrays
using Pyehtim
using Test
using OhMyThreads
using Enzyme
using KernelAbstractions
using Polyester
using OffsetArrays

using FiniteDifferences
# using ChainRulesCore
# using ChainRulesTestUtils
import DimensionalData as DD

@testset "ComradeBase.jl" begin
    @testset "exported names are defined" begin
        # PolarizedTypes exports some names it does not define.
        upstream = filter(n -> !isdefined(ComradeBase.PolarizedTypes, n), names(ComradeBase.PolarizedTypes))
        @test setdiff(filter(n -> !isdefined(ComradeBase, n), names(ComradeBase)), upstream) == Symbol[]
    end
    include(joinpath(@__DIR__, "interface.jl"))
    include(joinpath(@__DIR__, "images.jl"))
    include(joinpath(@__DIR__, "visibilities.jl"))
    include(joinpath(@__DIR__, "structured.jl"))
    include(joinpath(@__DIR__, "structuredmap.jl"))
    include(joinpath(@__DIR__, "structuredexec.jl"))
    include(joinpath(@__DIR__, "executors.jl"))
    include(joinpath(@__DIR__, "multidomain.jl"))
    include(joinpath(@__DIR__, "reactant.jl"))
    include(joinpath(@__DIR__, "polarized.jl"))
end
