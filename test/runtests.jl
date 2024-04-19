using ComradeBase, StaticArrays, JET
using StructArrays
using Pyehtim
using Test

using FiniteDifferences
using ChainRulesCore
using ChainRulesTestUtils
import DimensionalData as DD


@testset "ComradeBase.jl" begin
    include(joinpath(@__DIR__, "interface.jl"))
    include(joinpath(@__DIR__, "images.jl"))
    include(joinpath(@__DIR__, "io.jl"))
    include(joinpath(@__DIR__, "visibilities.jl"))
    include(joinpath(@__DIR__, "executors.jl"))
end
